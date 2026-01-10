from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import numpy as np
import pandas as pd

# Ensure repo root is importable when running as a script (e.g., GitHub Actions)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.market_data import normalize_ticker
from lib.price_cache import load_prices_with_cache
from lib.sp500_history import SP500History
from logger import logger


KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class StrategyRun:
    label: str
    mode: str  # head|tail
    lookback: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate daily head/tail rank signals + HTML email body.")
    p.add_argument("--top", type=int, default=50, help="선택할 종목 수(동일가중)")
    p.add_argument("--head-lookbacks", default="60,120,250", help="모멘텀 lookback 후보(영업일), 예: 60,120,250")
    p.add_argument("--tail-lookbacks", default="10,20,40", help="리버설 lookback 후보(영업일), 예: 10,20,40")
    p.add_argument("--max-daily-change", type=float, default=1.0, help="lookback 구간 내 일별수익률 절대값 상한(초과 시 제외)")
    p.add_argument("--output-dir", default=".output/daily", help="산출물 디렉토리")
    p.add_argument("--asof-kst", default=None, help="메일 기준시각(KST, ISO8601). 기본 now.")
    return p.parse_args()


def _parse_int_list(csv: str) -> list[int]:
    items = [s.strip() for s in csv.split(",") if s.strip()]
    return [int(x) for x in items]


def _compute_fetch_start(max_lookback: int) -> str:
    # yfinance start는 달력일이므로, 영업일 기준 lookback에 충분한 버퍼를 준다.
    start = (pd.Timestamp.utcnow().normalize() - pd.tseries.offsets.BDay(max_lookback + 10)).date()
    return start.isoformat()


def _pick_universe(history: SP500History, ref_date: pd.Timestamp, prices: pd.DataFrame) -> list[str]:
    members = {normalize_ticker(t) for t in history.constituents(ref_date)}
    available = members.intersection(prices.columns)
    return sorted(available)


def _select_ranked(
    prices: pd.DataFrame,
    universe: list[str],
    ref_date: pd.Timestamp,
    lookback: int,
    top: int,
    mode: str,
    max_daily_change: float,
) -> pd.DataFrame:
    window = prices.loc[:ref_date].tail(lookback + 1)
    if len(window) < lookback + 1:
        return pd.DataFrame()

    valid_cols = [c for c in universe if c in window.columns and window[c].notna().all()]
    if not valid_cols:
        return pd.DataFrame()

    # lookback 구간 내 극단 일변동 필터
    daily = window[valid_cols].pct_change(fill_method=None).iloc[1:].replace([np.inf, -np.inf], np.nan)
    if not daily.empty and max_daily_change is not None:
        mask_ok = daily.abs().max() <= max_daily_change
        valid_cols = [c for c in valid_cols if bool(mask_ok.get(c, False))]
        if not valid_cols:
            return pd.DataFrame()

    start_px = window[valid_cols].iloc[0]
    end_px = window[valid_cols].iloc[-1]
    rets = end_px / start_px - 1
    rets = rets.dropna()
    if rets.empty:
        return pd.DataFrame()

    ranked = rets.sort_values(ascending=(mode == "tail")).head(top)
    if ranked.empty:
        return pd.DataFrame()

    df = ranked.rename("lookback_return").to_frame().reset_index().rename(columns={"index": "ticker"})
    df.insert(0, "rank", range(1, len(df) + 1))
    df["lookback_price"] = start_px.loc[df["ticker"]].values
    df["current_price"] = end_px.loc[df["ticker"]].values
    df["lookback_date"] = window.index[0].date()
    df["current_date"] = window.index[-1].date()
    return df


def _render_html(
    asof_kst: datetime,
    data_date: pd.Timestamp,
    top: int,
    head_runs: list[tuple[StrategyRun, pd.DataFrame]],
    tail_runs: list[tuple[StrategyRun, pd.DataFrame]],
    max_daily_change: float | None,
) -> str:
    def _slug(run: StrategyRun) -> str:
        return f"{run.mode}-lk{run.lookback}"

    def _table(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p><i>No picks (insufficient data)</i></p>"
        out = df.copy()
        lb_date = out["lookback_date"].iloc[0]
        cur_date = out["current_date"].iloc[0]
        out["lookback_price"] = out["lookback_price"].map(lambda x: f"{x:,.2f}")
        out["current_price"] = out["current_price"].map(lambda x: f"{x:,.2f}")
        out["lookback_return"] = out["lookback_return"].map(lambda x: f"{x:.2%}")
        out = out[["rank", "ticker", "lookback_price", "current_price", "lookback_return"]]
        out = out.rename(columns={"lookback_price": f"{lb_date:%Y-%m-%d}", "current_price": f"{cur_date:%Y-%m-%d}", "lookback_return": "LookbackReturn"})
        return out.to_html(index=False, escape=True, border=0)

    style = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35; margin: 0; }
.container { margin-left: 230px; padding: 18px 24px 32px; }
.sidebar { position: fixed; top: 12px; left: 12px; width: 200px; border: 1px solid #e5e5e5; border-radius: 8px; padding: 12px; background: #fbfbfb; }
.sidebar h3 { margin: 0 0 8px; font-size: 14px; }
.sidebar ul { list-style: none; padding-left: 0; margin: 0; }
.sidebar li { margin: 4px 0; font-size: 13px; }
h1 { margin: 0 0 10px; }
h2 { margin: 18px 0 8px; }
h3 { margin: 12px 0 6px; }
.meta { color: #555; font-size: 13px; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; max-width: 1000px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #e5e5e5; text-align: left; font-size: 13px; }
th { background: #fafafa; position: sticky; top: 0; }
.section { margin-bottom: 22px; }
</style>
""".strip()

    parts: list[str] = [
        "<html><head><meta charset='utf-8'/>",
        style,
        "</head><body>",
    ]

    # Sidebar TOC
    parts.append("<div class='sidebar'>")
    parts.append("<h3>목차</h3>")
    parts.append("<ul>")
    for run, _ in head_runs:
        parts.append(f"<li><a href='#{_slug(run)}'>Head lk={run.lookback}</a></li>")
    for run, _ in tail_runs:
        parts.append(f"<li><a href='#{_slug(run)}'>Tail lk={run.lookback}</a></li>")
    parts.append("</ul></div>")

    parts.append("<div class='container'>")
    parts.append("<h1>Daily Holdings (Rank Head/Tail)</h1>")
    parts.append(
        f"<div class='meta'>As-of (KST): {asof_kst.strftime('%Y-%m-%d %H:%M %Z')}<br/>"
        f"Data date (latest close): {data_date.date().isoformat()}<br/>"
        f"Equal-weight universe pick, top={top}</div>"
    )

    head_lks = ", ".join(str(run.lookback) for run, _ in head_runs)
    tail_lks = ", ".join(str(run.lookback) for run, _ in tail_runs)
    parts.append("<div class='section'>")
    parts.append("<h2>요약</h2>")
    parts.append("<ul>")
    parts.append("<li>목적: 08:00 KST 기준 S&P500 편입 종목 중 모멘텀/리버설 상·하위 top 동일가중 보유 리스트</li>")
    parts.append("<li>계산: lookback 영업일 전 종가 대비 누적수익률을 정렬해 Head(상위) / Tail(하위)을 추출</li>")
    parts.append("<li>유니버스: 해당 데이터 기준일의 실제 S&P500 편입 종목만 사용, 결측 제거</li>")
    parts.append(
        "<li>필터: "
        + (f"일별 변동폭 |r| <= {max_daily_change:.0%} 종목만 포함" if max_daily_change is not None else "극단 변동 필터 미적용")
        + "</li>"
    )
    parts.append("</ul>")
    parts.append(
        f"<div class='meta'>Head lookback: {head_lks or '-'} | Tail lookback: {tail_lks or '-'} | top={top}</div>"
    )
    parts.append("</div>")

    parts.append("<div class='section'><h2>Head (Momentum)</h2>")
    for run, df in head_runs:
        parts.append(f"<h3 id='{_slug(run)}'>lookback={run.lookback} (business days)</h3>")
        parts.append(_table(df if not df.empty else df))
    parts.append("</div>")

    parts.append("<div class='section'><h2>Tail (Reversal)</h2>")
    for run, df in tail_runs:
        parts.append(f"<h3 id='{_slug(run)}'>lookback={run.lookback} (business days)</h3>")
        parts.append(_table(df if not df.empty else df))
    parts.append("</div>")

    parts.append("</div>")  # container

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()

    asof_kst = (
        datetime.fromisoformat(args.asof_kst).astimezone(KST)
        if args.asof_kst
        else datetime.now(tz=KST)
    )
    head_lbs = _parse_int_list(args.head_lookbacks)
    tail_lbs = _parse_int_list(args.tail_lookbacks)
    all_lbs = sorted(set(head_lbs + tail_lbs))
    max_lb = max(all_lbs) if all_lbs else 0

    history = SP500History()
    raw_tickers = history.df["ticker"].unique().tolist()
    tickers = [normalize_ticker(t) for t in raw_tickers]

    fetch_start = _compute_fetch_start(max_lb + 1)
    prices = load_prices_with_cache(tickers, fetch_start, end=None)
    if prices.empty:
        raise RuntimeError("Price data is empty; cannot generate signals.")

    data_date = prices.index.max()
    universe = _pick_universe(history, data_date, prices)
    logger.info(f"Universe at {data_date.date()}: {len(universe)} tickers")

    head_runs: list[tuple[StrategyRun, pd.DataFrame]] = []
    tail_runs: list[tuple[StrategyRun, pd.DataFrame]] = []

    for lb in head_lbs:
        run = StrategyRun(label=f"rank_head_lk{lb}_top{args.top}", mode="head", lookback=lb)
        df = _select_ranked(prices, universe, data_date, lb, args.top, "head", args.max_daily_change)
        head_runs.append((run, df))

    for lb in tail_lbs:
        run = StrategyRun(label=f"rank_tail_lk{lb}_top{args.top}", mode="tail", lookback=lb)
        df = _select_ranked(prices, universe, data_date, lb, args.top, "tail", args.max_daily_change)
        tail_runs.append((run, df))

    out_root = Path(args.output_dir)
    out_dir = out_root / asof_kst.strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for run, df in head_runs + tail_runs:
        if df.empty:
            continue
        weight = 1.0 / float(args.top) if args.top else float("nan")
        for _, row in df.iterrows():
            records.append(
                {
                    "asof_kst": asof_kst.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "data_date": data_date.date().isoformat(),
                    "strategy": "rank_head" if run.mode == "head" else "rank_tail",
                    "mode": run.mode,
                    "lookback": run.lookback,
                    "top": args.top,
                    "ticker": row["ticker"],
                    "rank": int(row["rank"]),
                    "weight": weight,
                    "lookback_return": float(row["lookback_return"]),
                    "lookback_price": float(row["lookback_price"]),
                    "current_price": float(row["current_price"]),
                    "lookback_date": str(row["lookback_date"]),
                    "current_date": str(row["current_date"]),
                }
            )

    signals_path = out_dir / "signals.csv"
    signals_df = pd.DataFrame.from_records(records).sort_values(["mode", "lookback", "rank", "ticker"])
    # Keep a stable, cross-strategy-friendly column order (asset allocation report will emit the same superset).
    col_order = [
        "asof_kst",
        "data_date",
        "strategy",
        "mode",
        "lookback",
        "top",
        "ticker",
        "rank",
        "weight",
        "lookback_return",
        "lookback_price",
        "current_price",
        "lookback_date",
        "current_date",
    ]
    existing = [c for c in col_order if c in signals_df.columns]
    remaining = [c for c in signals_df.columns if c not in existing]
    signals_df = signals_df[existing + remaining]
    signals_df.to_csv(signals_path, index=False)

    html = _render_html(asof_kst, data_date, args.top, head_runs, tail_runs, args.max_daily_change)
    html_path = out_dir / "email.html"
    html_path.write_text(html, encoding="utf-8")

    logger.info(f"Wrote {signals_path}")
    logger.info(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
