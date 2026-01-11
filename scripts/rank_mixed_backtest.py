from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from lib.market_data import normalize_ticker
from lib.price_cache import load_prices_with_cache
from lib.sp500_history import SP500History
from logger import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mixed strategy: short-term reversal + mid-term momentum on S&P500 (equal weight, SPY benchmark)."
    )
    p.add_argument("--start", default="2005-01-01", help="백테스트 시작일 (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="백테스트 종료일 (YYYY-MM-DD, 기본 today)")
    p.add_argument("--rebalance-months", type=int, default=1, help="리밸런스 주기(개월 단위)")
    p.add_argument("--short-lookback", type=int, default=10, help="단기 리버설 구간(거래일)")
    p.add_argument("--mid-lookback", type=int, default=126, help="중기 모멘텀 구간(거래일)")
    p.add_argument("--exclude-recent-days", type=int, default=21, help="중기 모멘텀 계산 시 최근 제외 일수 (0이면 제외 없음)")
    p.add_argument("--weight-mom", type=float, default=0.7, help="모멘텀 스코어 가중치 (0~1)")
    p.add_argument("--weight-rev", type=float, default=0.3, help="리버설 스코어 가중치 (0~1)")
    p.add_argument("--top", type=int, default=10, help="선택할 종목 수")
    p.add_argument("--max-daily-change", type=float, default=1.0, help="일별 수익률 절대값 상한")
    p.add_argument("--output", default=".output/rank_mix_backtest.csv", help="포트폴리오 수익률 저장 경로")
    p.add_argument("--monthly-output", default=".output/rank_mix_periods.csv", help="리밸런스 구간별 요약 저장 경로")
    p.add_argument("--report", default=".output/rank_mix_report.html", help="quantstats 리포트 경로")
    p.add_argument("--cache-dir", default=".cache/sp500", help="가격 캐시 디렉토리 (S&P500용 권장: .cache/sp500)")
    return p.parse_args()


def get_rebalance_dates(prices: pd.DataFrame, every_months: int = 1) -> List[pd.Timestamp]:
    by_month = prices.groupby(prices.index.to_period("M")).apply(lambda df: df.index.max())
    monthly_ends = list(by_month.sort_values())
    if every_months <= 1:
        return monthly_ends
    stepped = monthly_ends[::every_months] if monthly_ends else []
    if monthly_ends:
        last = monthly_ends[-1]
        if not stepped or last != stepped[-1]:
            stepped.append(last)
    return stepped


def zscore(series: pd.Series) -> pd.Series:
    if series.std(ddof=0) == 0 or series.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / series.std(ddof=0)


def compute_fetch_start(start: str | None, lookback_days: int) -> str | None:
    if not start:
        return start
    start_ts = pd.Timestamp(start)
    fetch_start = start_ts - pd.tseries.offsets.BDay(lookback_days)
    return fetch_start.strftime("%Y-%m-%d")


def select_portfolio(
    prices: pd.DataFrame,
    universe: List[str],
    rebalance_date: pd.Timestamp,
    short_lb: int,
    mid_lb: int,
    exclude_recent: int,
    w_mom: float,
    w_rev: float,
    top: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    window = prices.loc[:rebalance_date]
    short_window = window.tail(short_lb + 1)
    mid_window = window.tail(mid_lb + exclude_recent + 1)

    if len(short_window) < short_lb + 1 or len(mid_window) < mid_lb + exclude_recent + 1:
        return {}, pd.DataFrame()

    valid_cols = [c for c in universe if short_window[c].notna().all() and mid_window[c].notna().all()]
    if not valid_cols:
        return {}, pd.DataFrame()

    short_rets = short_window[valid_cols].iloc[-1] / short_window[valid_cols].iloc[0] - 1

    # 중기 모멘텀: 최근 exclude_recent 일을 제외한 mid_lb 기간 수익률
    end_idx = -(exclude_recent + 1) if exclude_recent > 0 else -1
    mid_start = mid_window[valid_cols].iloc[0]
    mid_end = mid_window[valid_cols].iloc[end_idx]
    mid_rets = mid_end / mid_start - 1

    # 스코어 결합
    mom_z = zscore(mid_rets)
    rev_z = zscore(-short_rets)  # 단기 낙폭이 클수록 높은 점수
    score = w_mom * mom_z + w_rev * rev_z
    score = score.dropna().sort_values(ascending=False)
    picks = score.head(top)
    if picks.empty:
        return {}, pd.DataFrame()

    weight = 1 / len(picks)
    weights = {t: weight for t in picks.index}

    detail = pd.DataFrame(
        {
            "score": picks,
            "short_return": short_rets.loc[picks.index],
            "mid_return": mid_rets.loc[picks.index],
        }
    )
    return weights, detail


def run_backtest(args: argparse.Namespace):
    history = SP500History()
    raw_tickers = history.df["ticker"].unique().tolist()
    tickers = [normalize_ticker(t) for t in raw_tickers]
    if "SPY" not in tickers:
        tickers.append("SPY")

    required_lb = max(
        args.short_lookback + 1,
        args.mid_lookback + args.exclude_recent_days + 1,
    )
    fetch_start = compute_fetch_start(args.start, required_lb)
    prices = load_prices_with_cache(tickers, fetch_start, args.end, cache_dir=Path(args.cache_dir))
    if prices.empty:
        raise RuntimeError("Price data is empty.")

    daily_rets = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], pd.NA)
    rebal_dates = get_rebalance_dates(prices, args.rebalance_months)
    start_ts = pd.Timestamp(args.start) if args.start else None
    if start_ts is not None:
        # 시작일 이후 리밸런스만 사용하되, 시작일(또는 이후 첫 거래일)을 리밸런스 기준으로 삽입해 초기 구간 수익이 0으로 찍히는 것을 방지
        if start_ts < prices.index.min():
            start_ts = prices.index.min()
        start_aligned = prices.index[prices.index.get_indexer([start_ts], method="bfill")[0]]
        rebal_dates = [d for d in rebal_dates if d >= start_ts]
        if not rebal_dates or start_aligned < rebal_dates[0]:
            rebal_dates = [start_aligned] + rebal_dates
    logger.info(f"Rebalance dates: {len(rebal_dates)} periods (every {args.rebalance_months} month(s))")

    port_returns: List[pd.Series] = []
    spy_returns: List[pd.Series] = []
    monthly_records: List[Dict[str, object]] = []

    for i, reb_date in enumerate(rebal_dates):
        universe = [normalize_ticker(t) for t in history.constituents(reb_date)]
        universe = [t for t in universe if t in prices.columns]
        weights, detail = select_portfolio(
            prices,
            universe,
            reb_date,
            args.short_lookback,
            args.mid_lookback,
            args.exclude_recent_days,
            args.weight_mom,
            args.weight_rev,
            args.top,
        )

        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else prices.index[-1]
        segment = prices.loc[(prices.index > reb_date) & (prices.index <= next_date)]
        if segment.empty:
            continue

        port_ret = pd.Series(0, index=segment.index)
        if weights:
            w = pd.Series(weights)
            valid_cols = [
                t
                for t in w.index
                if t in prices.columns
                and pd.notna(prices.loc[:reb_date, t].iloc[-1])
                and segment[t].notna().all()
            ]
            if valid_cols:
                w = w[valid_cols] / w[valid_cols].sum()
                seg_rets = daily_rets.loc[(daily_rets.index > reb_date) & (daily_rets.index <= next_date), valid_cols]
                seg_rets = seg_rets.replace([np.inf, -np.inf], pd.NA)
                mask_valid = seg_rets.abs().max() <= args.max_daily_change
                seg_rets = seg_rets.loc[:, mask_valid].fillna(0)
                if not seg_rets.empty:
                    valid_w = w[seg_rets.columns]
                    valid_w = valid_w / valid_w.sum()
                    port_ret = seg_rets.mul(valid_w, axis=1).sum(axis=1)

        port_returns.append(port_ret)

        spy_ret = pd.Series(0, index=segment.index)
        spy_month_ret = None
        if "SPY" in prices.columns:
            spy_seg = daily_rets.loc[(daily_rets.index > reb_date) & (daily_rets.index <= next_date), ["SPY"]]
            if not spy_seg.empty and spy_seg["SPY"].notna().all():
                spy_ret = spy_seg["SPY"]
                spy_month_ret = (1 + spy_ret).prod() - 1
        spy_returns.append(spy_ret)

        month_ret = (1 + port_ret).prod() - 1 if not port_ret.empty else 0.0
        active_vs_spy = month_ret - spy_month_ret if spy_month_ret is not None else None

        monthly_records.append(
            {
                "year_month": reb_date.strftime("%Y-%m"),
                "rule": "mixed_reversal_momentum",
                "lookback_short": args.short_lookback,
                "lookback_mid": args.mid_lookback,
                "weight_mom": args.weight_mom,
                "weight_rev": args.weight_rev,
                "buy_date": segment.index.min().date() if not segment.empty else None,
                "sell_date": segment.index.max().date() if not segment.empty else None,
                "return": month_ret,
                "spy_return": spy_month_ret,
                "active_return_vs_spy": active_vs_spy,
            }
        )

    if not port_returns:
        raise RuntimeError("No portfolio returns generated.")

    port_series = pd.concat(port_returns).sort_index()
    spy_series = pd.concat(spy_returns).sort_index() if spy_returns else None
    if spy_series is not None:
        spy_series = spy_series.reindex(port_series.index)

    return port_series, monthly_records, spy_series


def main():
    args = parse_args()
    port_ret, monthly_records, benchmark = run_backtest(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    port_ret.to_csv(args.output, header=["return"])
    logger.info(f"Saved portfolio returns to {args.output}")

    monthly_path = Path(args.monthly_output)
    monthly_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(monthly_records).to_csv(monthly_path, index=False)
    logger.info(f"Saved period holdings/returns to {args.monthly_output}")

    if benchmark is not None and benchmark.name is None:
        benchmark.name = "SPY"

    from quantstats import reports  # local import to avoid unused dependency if not generating report

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reports.html(
        port_ret,
        benchmark=benchmark if benchmark is not None and not benchmark.empty else None,
        output=report_path,
        title=(
            f"Mixed Reversal+Momentum "
            f"(short={args.short_lookback}, mid={args.mid_lookback}, "
            f"w_mom={args.weight_mom}, w_rev={args.weight_rev}, top={args.top})"
        ),
    )
    logger.info(f"Saved quantstats report to {args.report}")


if __name__ == "__main__":
    main()
