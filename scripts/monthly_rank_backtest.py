from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import quantstats as qs
import yfinance as yf

from lib.sp500_history import SP500History
from lib.market_data import normalize_ticker
from logger import logger

STRATEGY_LABEL = {"head": "rank_head", "tail": "rank_tail"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly rank-based backtest on S&P500 (head=top winners, tail=bottom losers).")
    p.add_argument("--start", default="2005-01-01", help="백테스트 시작일 (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="백테스트 종료일 (YYYY-MM-DD, 기본 today)")
    p.add_argument("--lookback", type=int, default=30, help="수익률 계산에 사용할 lookback 거래일 수 (n)")
    p.add_argument("--top", type=int, default=10, help="선택할 종목 수 (m)")
    p.add_argument("--mode", choices=["head", "tail"], default="head", help="수익률 상위(head) / 하위(tail) 선택")
    p.add_argument("--max-daily-change", type=float, default=1, help="일별 수익률 절대값 상한(초과 시 해당 종목 제외). 예: 0.5 = ±50%")
    p.add_argument("--output", default=".output/rank_backtest.csv", help="포트폴리오 수익률 저장 경로")
    p.add_argument("--monthly-output", default=".output/rank_monthly.csv", help="월별 보유종목/수익률 저장 경로")
    p.add_argument("--report", default=".output/rank_report.html", help="quantstats 리포트 경로")
    return p.parse_args()


def load_price_data(tickers: List[str], start: str, end: str | None) -> pd.DataFrame:
    logger.info(f"Downloading prices for {len(tickers)} tickers from {start} to {end or 'today'}")
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=True)
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    closes = closes.dropna(how="all").dropna(axis=1, how="all").sort_index()
    logger.info(f"Dropped all-NaN columns, remaining tickers: {len(closes.columns)}")
    logger.info(f"Price matrix shape: {closes.shape}")
    return closes


def get_rebalance_dates(prices: pd.DataFrame) -> List[pd.Timestamp]:
    by_month = prices.groupby(prices.index.to_period("M")).apply(lambda df: df.index.max())
    return list(by_month.sort_values())


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
) -> Dict[str, float]:
    window = prices.loc[:rebalance_date].tail(lookback + 1)
    if len(window) < lookback + 1:
        return {}

    # lookback 구간에 결측이 없는 종목만 고려
    valid_cols = [c for c in universe if window[c].notna().all()]
    if not valid_cols:
        return {}

    rets = window[valid_cols].iloc[-1] / window[valid_cols].iloc[0] - 1
    candidates = rets.dropna()
    if candidates.empty:
        return {}

    sorted_candidates = candidates.sort_values(ascending=(mode == "tail"))
    picks = sorted_candidates.head(top)
    if picks.empty:
        return {}

    weight = 1 / len(picks)
    return {t: weight for t in picks.index}


def run_backtest(args: argparse.Namespace) -> tuple[pd.Series, List[Dict[str, object]], pd.Series | None]:
    history = SP500History()
    raw_tickers = history.df["ticker"].unique().tolist()
    tickers = [normalize_ticker(t) for t in raw_tickers]
    if "SPY" not in tickers:
        tickers.append("SPY")  # 벤치마크용

    prices = load_price_data(tickers, args.start, args.end)
    if prices.empty:
        raise RuntimeError("Price data is empty; check date range or tickers.")

    rebal_dates = get_rebalance_dates(prices)
    logger.info(f"Rebalance dates: {len(rebal_dates)} months")

    port_returns: List[pd.Series] = []
    monthly_records: List[Dict[str, object]] = []
    for i, reb_date in enumerate(rebal_dates):
        universe = pick_universe(history, reb_date, prices)
        weights = select_portfolio(prices, universe, reb_date, args.lookback, args.top, args.mode)

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
                anchor = prices.loc[:reb_date, valid_cols].iloc[-1:]
                seg_with_anchor = pd.concat([anchor, segment[valid_cols]])
                seg_rets = seg_with_anchor.pct_change(fill_method=None).iloc[1:]
                seg_rets = seg_rets.replace([np.inf, -np.inf], pd.NA)
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

        month_ret = (1 + port_ret).prod() - 1 if not port_ret.empty else 0.0
        monthly_records.append({
            "year_month": reb_date.strftime("%Y-%m"),
            "rule": STRATEGY_LABEL.get(args.mode, args.mode),
            "holdings": ";".join(f"{k}={v:.4f}" for k, v in holding_returns.items()) if isinstance(holding_returns, pd.Series) and not holding_returns.empty else "",
            "return": month_ret,
        })

    if not port_returns:
        raise RuntimeError("No portfolio returns generated.")

    port_series = pd.concat(port_returns).sort_index()
    bench = None
    if "SPY" in prices.columns:
        bench = prices["SPY"].pct_change().reindex(port_series.index)

    return port_series, monthly_records, bench


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
    logger.info(f"Saved monthly holdings/returns to {args.monthly_output}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    qs.reports.html(
        port_ret,
        benchmark=benchmark,
        output=report_path,
        title=f"S&P500 Rank Strategy ({STRATEGY_LABEL.get(args.mode, args.mode)}, lookback={args.lookback}, top={args.top})",
    )
    logger.info(f"Saved quantstats report to {args.report}")


if __name__ == "__main__":
    main()
