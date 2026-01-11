from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
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


@dataclass(frozen=True)
class RunDiagnostics:
    universe_size: int
    valid_data_tickers: int
    dropped_missing_lookback: int
    dropped_extreme_change: int
    picks: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate daily head/tail rank signals + HTML email body.")
    p.add_argument("--top", type=int, default=50, help="선택할 종목 수(동일가중)")
    p.add_argument("--head-lookbacks", default="60,120,250", help="모멘텀 lookback 후보(영업일), 예: 60,120,250")
    p.add_argument("--tail-lookbacks", default="10,20,40", help="리버설 lookback 후보(영업일), 예: 10,20,40")
    p.add_argument("--max-daily-change", type=float, default=1.0, help="lookback 구간 내 일별수익률 절대값 상한(초과 시 제외)")
    p.add_argument("--output-dir", default=".output/daily", help="산출물 디렉토리")
    p.add_argument("--cache-dir", default=".cache/sp500", help="가격 캐시 디렉토리 (S&P500용 권장: .cache/sp500)")
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

    # Robustly name the index column as "ticker" regardless of the Series index name.
    df = ranked.rename("lookback_return").reset_index()
    if df.shape[1] >= 2:
        df.columns = ["ticker", "lookback_return"] + list(df.columns[2:])
    df.insert(0, "rank", range(1, len(df) + 1))
    df["lookback_price"] = start_px.loc[df["ticker"]].values
    df["current_price"] = end_px.loc[df["ticker"]].values
    df["lookback_date"] = window.index[0].date()
    df["current_date"] = window.index[-1].date()
    return df


def _select_ranked_with_diagnostics(
    prices: pd.DataFrame,
    universe: list[str],
    ref_date: pd.Timestamp,
    lookback: int,
    top: int,
    mode: str,
    max_daily_change: float,
) -> tuple[pd.DataFrame, RunDiagnostics]:
    window = prices.loc[:ref_date].tail(lookback + 1)
    if len(window) < lookback + 1:
        return (
            pd.DataFrame(),
            RunDiagnostics(
                universe_size=len(universe),
                valid_data_tickers=0,
                dropped_missing_lookback=len(universe),
                dropped_extreme_change=0,
                picks=0,
            ),
        )

    valid_missing = [c for c in universe if c in window.columns and window[c].notna().all()]
    dropped_missing = max(0, len(universe) - len(valid_missing))

    # lookback 구간 내 극단 일변동 필터
    dropped_extreme = 0
    valid_cols = list(valid_missing)
    daily = window[valid_cols].pct_change(fill_method=None).iloc[1:].replace([np.inf, -np.inf], np.nan)
    if not daily.empty and max_daily_change is not None:
        mask_ok = daily.abs().max() <= max_daily_change
        before = len(valid_cols)
        valid_cols = [c for c in valid_cols if bool(mask_ok.get(c, False))]
        dropped_extreme = max(0, before - len(valid_cols))

    df = _select_ranked(prices, universe, ref_date, lookback, top, mode, max_daily_change)
    diag = RunDiagnostics(
        universe_size=len(universe),
        valid_data_tickers=len(valid_missing),
        dropped_missing_lookback=dropped_missing,
        dropped_extreme_change=dropped_extreme,
        picks=int(len(df)) if df is not None else 0,
    )
    return df, diag


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

    # Keep CSS email-friendly: avoid position:fixed and sticky headers (often stripped by email clients).
    style = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35; margin: 0; }
.container { padding: 18px 24px 32px; }
h1 { margin: 0 0 10px; }
h2 { margin: 18px 0 8px; }
h3 { margin: 12px 0 6px; }
.meta { color: #555; font-size: 13px; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; max-width: 1000px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #e5e5e5; text-align: left; font-size: 13px; }
th { background: #fafafa; }
.section { margin-bottom: 22px; }
.toc { font-size: 13px; margin: 10px 0 14px; }
.toc a { text-decoration: none; }
</style>
""".strip()

    parts: list[str] = [
        "<html><head><meta charset='utf-8'/>",
        style,
        "</head><body>",
    ]

    parts.append("<div class='container'>")
    parts.append("<h1>Daily Holdings (Rank Head/Tail)</h1>")
    parts.append(
        f"<div class='meta'>As-of (KST): {asof_kst.strftime('%Y-%m-%d %H:%M %Z')}<br/>"
        f"Data date (latest close): {data_date.date().isoformat()}<br/>"
        f"Equal-weight universe pick, top={top}</div>"
    )

    # TOC
    parts.append("<div class='toc'><b>목차</b>: ")
    toc_parts: list[str] = []
    for run, _ in head_runs:
        toc_parts.append(f"<a href='#{_slug(run)}'>Head lk={run.lookback}</a>")
    for run, _ in tail_runs:
        toc_parts.append(f"<a href='#{_slug(run)}'>Tail lk={run.lookback}</a>")
    parts.append(" | ".join(toc_parts) if toc_parts else "-")
    parts.append("</div>")

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


def _safe_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{float(x):+.2%}"
    except Exception:
        return "NA"


def _safe_price(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{float(x):,.2f}"
    except Exception:
        return "NA"


def _build_decision_notes(
    asof_kst: datetime,
    data_date: pd.Timestamp,
    top: int,
    head_runs: list[tuple[StrategyRun, pd.DataFrame]],
    tail_runs: list[tuple[StrategyRun, pd.DataFrame]],
    diag_rows: list[dict[str, object]],
) -> str:
    """
    Human-readable notes explaining *why* tickers were selected (audit/validation).
    """
    diag_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for d in diag_rows:
        try:
            diag_by_key[(str(d["mode"]), int(d["lookback"]))] = d
        except Exception:
            continue

    lines: list[str] = []
    lines.append(f"As-of (KST): {asof_kst.strftime('%Y-%m-%d %H:%M %Z')}")
    lines.append(f"Data date (latest close): {data_date.date().isoformat()}")
    lines.append(f"Rule: select top={top} by lookback return; equal weight={1/top:.4%}" if top else "Rule: top=0 (invalid)")
    lines.append("")
    lines.append("Filters:")
    lines.append("- Missing data: exclude tickers missing any price in the lookback window")
    lines.append("- Extreme move: exclude tickers whose |daily return| exceeds max_daily_change during lookback")
    lines.append("")

    def _section(mode: str, runs: list[tuple[StrategyRun, pd.DataFrame]]) -> None:
        title = "Head (Momentum)" if mode == "head" else "Tail (Reversal)"
        lines.append(f"[{title}]")
        if not runs:
            lines.append("  (no runs)")
            lines.append("")
            return
        for run, df in runs:
            lines.append(f"- Run: mode={run.mode}, lookback={run.lookback}, top={top}")
            diag = diag_by_key.get((run.mode, run.lookback))
            if diag:
                lines.append(
                    "  Universe="
                    + f"{diag.get('universe_size','NA')}, "
                    + f"valid_data={diag.get('valid_data_tickers','NA')}, "
                    + f"dropped_missing={diag.get('dropped_missing_lookback','NA')}, "
                    + f"dropped_extreme={diag.get('dropped_extreme_change','NA')}, "
                    + f"picks={diag.get('picks','NA')}"
                )
            if df is None or df.empty:
                lines.append("  Picks: (none)")
                lines.append("")
                continue

            # df columns: rank,ticker,lookback_price,current_price,lookback_return,lookback_date,current_date
            lb_date = df["lookback_date"].iloc[0] if "lookback_date" in df.columns and not df.empty else "NA"
            cur_date = df["current_date"].iloc[0] if "current_date" in df.columns and not df.empty else "NA"
            lines.append(f"  Window: {lb_date} -> {cur_date}")
            lines.append("  Picks:")
            for _, row in df.iterrows():
                lines.append(
                    "    "
                    + f"{int(row.get('rank', 0)):>3d}. {row.get('ticker')} "
                    + f"ret={_safe_pct(row.get('lookback_return'))} "
                    + f"px({lb_date})={_safe_price(row.get('lookback_price'))} "
                    + f"px({cur_date})={_safe_price(row.get('current_price'))}"
                )
            lines.append("")

    _section("head", head_runs)
    _section("tail", tail_runs)
    return "\n".join(lines).strip() + "\n"


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
    # Daily report only needs the *current* S&P500 membership (not every historical constituent),
    # which keeps the download/universe smaller and avoids lots of delisted tickers.
    ref = pd.Timestamp.utcnow().normalize()
    if getattr(ref, "tzinfo", None) is not None:
        ref = ref.tz_localize(None)
    tickers = [normalize_ticker(t) for t in history.constituents(ref)]

    fetch_start = _compute_fetch_start(max_lb + 1)
    prices = load_prices_with_cache(tickers, fetch_start, end=None, cache_dir=Path(args.cache_dir))
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
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    for run, df in head_runs + tail_runs:
        if df.empty:
            continue
        weight = 1.0 / float(args.top) if args.top else float("nan")
        # Save per-run picks for audit/debug
        pick_path = details_dir / f"picks_{run.mode}_lk{run.lookback}_top{args.top}.csv"
        df.to_csv(pick_path, index=False)
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

        # Diagnostics for the run (counts)
        df2, diag = _select_ranked_with_diagnostics(
            prices=prices,
            universe=universe,
            ref_date=data_date,
            lookback=run.lookback,
            top=args.top,
            mode=run.mode,
            max_daily_change=args.max_daily_change,
        )
        _ = df2  # keep the function side-effect free; df already saved above
        diag_rows.append(
            {
                "mode": run.mode,
                "lookback": run.lookback,
                "top": args.top,
                "universe_size": diag.universe_size,
                "valid_data_tickers": diag.valid_data_tickers,
                "dropped_missing_lookback": diag.dropped_missing_lookback,
                "dropped_extreme_change": diag.dropped_extreme_change,
                "picks": diag.picks,
                "max_daily_change": args.max_daily_change,
            }
        )

    (details_dir / "meta.json").write_text(
        json.dumps(
            {
                "asof_kst": asof_kst.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "data_date": data_date.date().isoformat(),
                "top": args.top,
                "head_lookbacks": head_lbs,
                "tail_lookbacks": tail_lbs,
                "max_daily_change": args.max_daily_change,
                "universe_size": len(universe),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if diag_rows:
        pd.DataFrame.from_records(diag_rows).sort_values(["mode", "lookback"]).to_csv(details_dir / "diagnostics.csv", index=False)

    decision_notes = _build_decision_notes(asof_kst, data_date, args.top, head_runs, tail_runs, diag_rows)
    (details_dir / "decision_notes.txt").write_text(decision_notes, encoding="utf-8")

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
