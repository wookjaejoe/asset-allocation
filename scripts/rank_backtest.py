from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import quantstats as qs
from lib.sp500_history import SP500History
from lib.market_data import normalize_ticker
from lib.price_cache import load_prices_with_cache
from logger import logger

STRATEGY_LABEL = {"head": "rank_head", "tail": "rank_tail"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank-based backtest on S&P500 with configurable rebalance frequency (head=top winners, tail=bottom losers).")
    p.add_argument("--start", default="2005-01-01", help="백테스트 시작일 (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="백테스트 종료일 (YYYY-MM-DD, 기본 today)")
    p.add_argument("--rebalance-months", type=int, default=1, help="리밸런스 주기(개월 단위, 기본 월말마다 1)")
    p.add_argument("--lookback", type=int, default=30, help="수익률 계산에 사용할 lookback 거래일 수 (n)")
    p.add_argument("--top", type=int, default=10, help="선택할 종목 수 (m)")
    p.add_argument("--mode", choices=["head", "tail"], default="head", help="수익률 상위(head) / 하위(tail) 선택")
    p.add_argument("--max-daily-change", type=float, default=1, help="일별 수익률 절대값 상한(초과 시 해당 종목 제외). 예: 0.5 = ±50%")
    p.add_argument("--output", default=".output/rank_backtest.csv", help="포트폴리오 수익률 저장 경로")
    p.add_argument("--monthly-output", default=".output/rank_monthly.csv", help="리밸런스 구간별 보유종목/수익률 저장 경로 (파일명은 기존 관례)")
    p.add_argument("--report", default=".output/rank_report.html", help="quantstats 리포트 경로")
    return p.parse_args()


def load_price_data(tickers: List[str], start: str, end: str | None) -> pd.DataFrame:
    # 가격 데이터는 캐시를 우선 사용하고, staleness 정책(1시간 부분 업데이트, 1일 전체 재다운로드)에 따라 갱신한다.
    prices = load_prices_with_cache(tickers, start, end)
    logger.info(f"Dropped all-NaN columns, remaining tickers: {len(prices.columns)}")
    logger.info(f"Price matrix shape: {prices.shape}")
    return prices


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


def pick_universe(history: SP500History, date: pd.Timestamp, prices: pd.DataFrame) -> List[str]:
    members = {normalize_ticker(t) for t in history.constituents(date)}
    available = members.intersection(prices.columns)
    return sorted(available)


def select_portfolio(
    prices: pd.DataFrame,
    universe: List[str],
    rebalance_date: pd.Timestamp,
    lookback: int,
    top: int,
    mode: str,
) -> Tuple[Dict[str, float], pd.Series]:
    window = prices.loc[:rebalance_date].tail(lookback + 1)
    if len(window) < lookback + 1:
        return {}, pd.Series(dtype=float)

    valid_cols = [c for c in universe if window[c].notna().all()]
    if len(valid_cols) < len(universe):
        missing = sorted(set(universe) - set(valid_cols))
        logger.warning(
            f"Rebalance {rebalance_date.date()} skipped {len(missing)} tickers with missing data in lookback: {', '.join(missing[:20])}"
            + ("..." if len(missing) > 20 else "")
        )
    if not valid_cols:
        return {}, pd.Series(dtype=float)

    rets = window[valid_cols].iloc[-1] / window[valid_cols].iloc[0] - 1
    candidates = rets.dropna()
    if candidates.empty:
        return {}, pd.Series(dtype=float)

    sorted_candidates = candidates.sort_values(ascending=(mode == "tail"))
    picks = sorted_candidates.head(top)
    if picks.empty:
        return {}, pd.Series(dtype=float)

    weight = 1 / len(picks)
    weights = {t: weight for t in picks.index}
    return weights, candidates.loc[picks.index]


def run_backtest(args: argparse.Namespace) -> tuple[pd.Series, List[Dict[str, object]], pd.Series | None]:
    history = SP500History()
    raw_tickers = history.df["ticker"].unique().tolist()
    tickers = [normalize_ticker(t) for t in raw_tickers]
    if "SPY" not in tickers:
        tickers.append("SPY")  # 벤치마크 비교용

    prices = load_price_data(tickers, args.start, args.end)
    if prices.empty:
        raise RuntimeError("Price data is empty; check date range or tickers.")

    # 일별 수익률은 한 번만 계산해 리밸런스마다 슬라이스해서 사용
    daily_rets = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], pd.NA)

    rebal_dates = get_rebalance_dates(prices, args.rebalance_months)
    logger.info(f"Rebalance dates: {len(rebal_dates)} periods (every {args.rebalance_months} month(s))")

    port_returns: List[pd.Series] = []
    bench_returns: List[pd.Series] = []
    spy_returns: List[pd.Series] = []
    monthly_records: List[Dict[str, object]] = []
    for i, reb_date in enumerate(rebal_dates):
        universe = pick_universe(history, reb_date, prices)
        weights, rank_rets = select_portfolio(prices, universe, reb_date, args.lookback, args.top, args.mode)

        next_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else prices.index[-1]
        segment = prices.loc[(prices.index > reb_date) & (prices.index <= next_date)]
        if segment.empty:
            continue

        holding_returns = pd.Series(dtype=float)
        if weights:
            w = pd.Series(weights)

            # 리밸런스 시점과 이후 구간 모두 데이터가 있는 종목만 사용
            valid_cols = [
                t for t in w.index
                if t in prices.columns
                and pd.notna(prices.loc[:reb_date, t].iloc[-1])
                and segment[t].notna().all()
            ]
            if not valid_cols:
                port_ret = pd.Series(0, index=segment.index)
            else:
                w = w[valid_cols]
                w = w / w.sum()
                seg_rets = daily_rets.loc[(daily_rets.index > reb_date) & (daily_rets.index <= next_date), valid_cols]
                # 과도한 변동 필터링
                mask_valid = seg_rets.abs().max() <= args.max_daily_change
                if (~mask_valid).any():
                    dropped = mask_valid[~mask_valid].index
                    stats = seg_rets[dropped].abs().max().sort_values(ascending=False)
                    sample = "; ".join(f"{k}={v:.2f}" for k, v in stats.head(10).items())
                    logger.warning(
                        f"Rebalance {reb_date.date()} dropped {len(dropped)} tickers over max_daily_change={args.max_daily_change}: {sample}"
                    )
                seg_rets = seg_rets.loc[:, mask_valid].fillna(0)

                if seg_rets.empty:
                    port_ret = pd.Series(0, index=segment.index)
                else:
                    valid_w = w[seg_rets.columns]
                    valid_w = valid_w / valid_w.sum()
                    port_ret = seg_rets.mul(valid_w, axis=1).sum(axis=1)
                    holding_returns = (seg_rets + 1).prod() - 1
        else:
            port_ret = pd.Series(0, index=segment.index)

        port_returns.append(port_ret)

        # 벤치마크: 리밸런스 시점의 유니버스를 균등 가중하여 월간 수익률 계산
        bench_ret = pd.Series(0, index=segment.index)
        bench_month_ret = 0.0
        bench_cols = [
            t for t in universe
            if t in prices.columns
            and pd.notna(prices.loc[:reb_date, t].iloc[-1])
            and segment.get(t) is not None
            and segment[t].notna().all()
        ]
        if bench_cols:
            bench_rets = daily_rets.loc[(daily_rets.index > reb_date) & (daily_rets.index <= next_date), bench_cols]
            mask_valid_bench = bench_rets.abs().max() <= args.max_daily_change
            bench_rets = bench_rets.loc[:, mask_valid_bench].fillna(0)
            if not bench_rets.empty:
                bench_ret = bench_rets.mean(axis=1)
                bench_month_ret = (1 + bench_ret).prod() - 1
        bench_returns.append(bench_ret)

        # SPY 벤치마크: 동일 기간 SPY 단일 종목 수익률
        spy_ret = pd.Series(0, index=segment.index)
        spy_month_ret = None
        if "SPY" in prices.columns:
            spy_seg = daily_rets.loc[(daily_rets.index > reb_date) & (daily_rets.index <= next_date), ["SPY"]]
            if not spy_seg.empty and spy_seg["SPY"].notna().all():
                spy_ret = spy_seg["SPY"]
                spy_month_ret = (1 + spy_ret).prod() - 1
        spy_returns.append(spy_ret)

        month_ret = (1 + port_ret).prod() - 1 if not port_ret.empty else 0.0
        active_ret = month_ret - bench_month_ret

        lookback_window = prices.loc[:reb_date].tail(args.lookback + 1)
        lookback_start = lookback_window.index[0].date()
        lookback_end = reb_date.date()
        buy_date = segment.index.min().date() if not segment.empty else None
        sell_date = segment.index.max().date() if not segment.empty else None

        holdings_detail = ""
        if isinstance(holding_returns, pd.Series) and not holding_returns.empty:
            cols = holding_returns.index
            # anchor 가격과 종료 가격
            anchor_prices = prices.loc[:reb_date, cols].iloc[-1]
            exit_prices = segment[cols].iloc[-1]
            parts = []
            for t in cols:
                lb_ret = rank_rets.get(t, np.nan)
                mret = holding_returns.get(t, np.nan)
                buy_p = anchor_prices.get(t, np.nan)
                sell_p = exit_prices.get(t, np.nan)
                parts.append(
                    f"{t}:lb={lb_ret:.4f};buy={buy_p:.2f}@{buy_date};sell={sell_p:.2f}@{sell_date};mret={mret:.4f}"
                )
            holdings_detail = " | ".join(parts)

        monthly_records.append({
            "year_month": reb_date.strftime("%Y-%m"),
            "rule": STRATEGY_LABEL.get(args.mode, args.mode),
            "lookback_start": lookback_start,
            "lookback_end": lookback_end,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "holdings": holdings_detail,
            "return": month_ret,
            "benchmark_return": bench_month_ret,
            "spy_return": spy_month_ret,
            "active_return": active_ret,
            "active_return_vs_spy": month_ret - spy_month_ret if spy_month_ret is not None else None,
        })

    if not port_returns:
        raise RuntimeError("No portfolio returns generated.")

    port_series = pd.concat(port_returns).sort_index()
    bench_series = pd.concat(bench_returns).sort_index() if bench_returns else None
    if bench_series is not None:
        bench_series = bench_series.reindex(port_series.index)

    spy_series = pd.concat(spy_returns).sort_index() if spy_returns else None
    if spy_series is not None:
        spy_series = spy_series.reindex(port_series.index)

    # quantstats용 기본 벤치마크는 유니버스 균등가중, SPY는 별도 반환
    return port_series, monthly_records, bench_series


def main():
    args = parse_args()
    port_ret, monthly_records, benchmark = run_backtest(args)

    # quantstats가 benchmark.name을 참조하므로 기본 이름을 부여
    if benchmark is not None and benchmark.name is None:
        benchmark.name = "Benchmark"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    port_ret.to_csv(args.output, header=["return"])
    logger.info(f"Saved portfolio returns to {args.output}")

    monthly_path = Path(args.monthly_output)
    monthly_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(monthly_records).to_csv(monthly_path, index=False)
    logger.info(f"Saved period holdings/returns to {args.monthly_output}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        port_ret,
        benchmark=benchmark if benchmark is not None and not benchmark.empty else None,
        output=report_path,
        title=f"S&P500 Rank Strategy ({STRATEGY_LABEL.get(args.mode, args.mode)}, lookback={args.lookback}, top={args.top})",
    )
    logger.info(f"Saved quantstats report to {args.report}")


if __name__ == "__main__":
    main()
