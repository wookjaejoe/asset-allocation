from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

import numpy as np
import pandas as pd

# Ensure repo root is importable when running as a script (e.g., local shell / GitHub Actions)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.price_cache import load_prices_with_cache
from logger import logger
from strategy import InvestmentStrategyIntegration
from strategy.baa import Assets as BAAAssets, InvestmentStrategyBAA
from strategy.bda import Assets as BDAAssets, InvestmentStrategyBDA
from strategy.laa import Assets as LAAAssets, InvestmentStrategyLAA
from strategy.mdm import Assets as MDMAssets, InvestmentStrategyMDM


KST = ZoneInfo("Asia/Seoul")


def _parse_asof_kst(value: str) -> datetime:
    text = value.strip()
    if text.endswith("KST"):
        text = text[:-3].strip()
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(KST)
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate daily asset-allocation (integration) holdings + HTML email body.")
    p.add_argument("--output-dir", default=".output/daily_asset_allocation", help="산출물 디렉토리")
    p.add_argument("--cache-dir", default=".cache/asset_allocation", help="가격 캐시 디렉토리 (ETF/자산배분용 권장)")
    p.add_argument("--asof-kst", default=None, help="메일 기준시각(KST, ISO8601). 기본 now.")
    p.add_argument("--start", default=None, help="가격 데이터 시작일(YYYY-MM-DD). 기본: 최근 3년")
    return p.parse_args()


def _render_html(asof_kst: datetime, data_date: pd.Timestamp, weights: pd.Series) -> str:
    out = weights.copy()
    out = out.sort_values(ascending=False)
    df = out.rename("weight").to_frame().reset_index().rename(columns={"index": "ticker"})
    df.insert(0, "rank", range(1, len(df) + 1))
    df["weight"] = df["weight"].map(lambda x: f"{x:.2%}")

    style = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35; margin: 0; }
.container { padding: 18px 24px 32px; }
h1 { margin: 0 0 8px; }
.meta { color: #555; font-size: 13px; margin-bottom: 12px; }
table { border-collapse: collapse; width: 100%; max-width: 820px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #e5e5e5; text-align: left; font-size: 13px; }
th { background: #fafafa; position: sticky; top: 0; }
</style>
""".strip()

    parts: list[str] = [
        "<html><head><meta charset='utf-8'/>",
        style,
        "</head><body>",
        "<div class='container'>",
        "<h1>Daily Holdings (Asset Allocation)</h1>",
        (
            "<div class='meta'>"
            f"As-of (KST): {asof_kst.strftime('%Y-%m-%d %H:%M %Z')}<br/>"
            f"Data date (latest close): {data_date.date().isoformat()}<br/>"
            "Strategy: Integration (BAA/BDA/LAA/MDM merged, equal-weight across sub-strategies)"
            "</div>"
        ),
        df.to_html(index=False, escape=True, border=0),
        "</div></body></html>",
    ]
    return "\n".join(parts)


def _compute_breakdown(
    chart: pd.DataFrame,
    month_chart: pd.DataFrame,
    data_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - long_df: per-strategy holdings (strategy, ticker, weight_strategy, weight_final)
      - matrix_df: per-ticker contribution matrix with overlap count and final_weight
    """
    strategy_types = list(InvestmentStrategyIntegration.strategy_types)
    n_strategies = len(strategy_types) or 1

    long_rows: list[dict[str, object]] = []
    contrib: dict[str, dict[str, float]] = {}

    for st in strategy_types:
        s = st(chart=chart, month_chart=month_chart)
        port_series = s.portfolio
        if port_series.empty:
            continue
        port = port_series.iloc[-1]
        alloc = pd.Series(port.allocations, dtype=float)
        alloc = alloc.replace([np.inf, -np.inf], np.nan).dropna()
        alloc = alloc / alloc.sum() if float(alloc.sum() or 0) else alloc

        strategy_name = st.get_name()
        for ticker, w in alloc.items():
            w_final = float(w) / float(n_strategies)
            long_rows.append(
                {
                    "strategy": strategy_name,
                    "ticker": ticker,
                    "weight_strategy": float(w),
                    "weight_final": w_final,
                }
            )
            contrib.setdefault(ticker, {})[strategy_name] = w_final

    long_df = pd.DataFrame.from_records(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    # Wide matrix: ticker x strategy contribution
    matrix = (
        long_df.pivot_table(index="ticker", columns="strategy", values="weight_final", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    # Prevent pandas "columns.name" (='strategy') from becoming a confusing extra header column in CSV/Excel.
    matrix.columns.name = None
    matrix_df = matrix.copy()
    matrix_df["overlap_count"] = (matrix > 0).sum(axis=1).astype(int)
    matrix_df["final_weight"] = matrix.sum(axis=1)
    matrix_df = matrix_df.sort_values(["final_weight", "overlap_count"], ascending=[False, False]).reset_index()
    matrix_df.columns.name = None
    matrix_df.insert(0, "data_date", data_date.date().isoformat())

    long_df = long_df.sort_values(["strategy", "weight_final", "ticker"], ascending=[True, False, True]).reset_index(drop=True)
    long_df.insert(0, "data_date", data_date.date().isoformat())
    return long_df, matrix_df


def _render_html_with_breakdown(
    asof_kst: datetime,
    data_date: pd.Timestamp,
    weights: pd.Series,
    long_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    decision_notes: str,
) -> str:
    base = _render_html(asof_kst, data_date, weights)
    if long_df.empty or matrix_df.empty:
        if not decision_notes.strip():
            return base
        insert = "\n".join(
            [
                "<hr/>",
                "<h2>Decision Notes (검산용)</h2>",
                "<pre style='white-space:pre-wrap; font-size: 12px; line-height: 1.35; background:#fafafa; border:1px solid #e5e5e5; padding:10px; border-radius:8px;'>",
                decision_notes.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                "</pre>",
            ]
        )
        return base.replace("</div></body></html>", insert + "\n</div></body></html>")

    long_view = long_df.copy()
    long_view["weight_strategy"] = long_view["weight_strategy"].map(lambda x: f"{x:.2%}")
    long_view["weight_final"] = long_view["weight_final"].map(lambda x: f"{x:.2%}")
    long_view = long_view[["strategy", "ticker", "weight_strategy", "weight_final"]]

    matrix_view = matrix_df.copy()
    for c in matrix_view.columns:
        if c in {"data_date", "ticker", "overlap_count"}:
            continue
        matrix_view[c] = matrix_view[c].map(lambda x: f"{float(x):.2%}")
    matrix_view = matrix_view.drop(columns=["data_date"])

    insert = "\n".join(
        [
            "<hr/>",
            "<h2>Breakdown (검산용)</h2>",
            "<h3>0) Decision Notes</h3>",
            "<pre style='white-space:pre-wrap; font-size: 12px; line-height: 1.35; background:#fafafa; border:1px solid #e5e5e5; padding:10px; border-radius:8px;'>"
            + decision_notes.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            + "</pre>",
            "<ul>",
            "<li>최종 비중 = (각 서브전략 비중) / 서브전략 개수(현재 4개)</li>",
            "<li>동일 티커가 여러 서브전략에 포함되면 최종 비중이 합산됩니다</li>",
            "</ul>",
            "<h3>1) 서브전략별 보유/비중</h3>",
            long_view.to_html(index=False, escape=True, border=0),
            "<h3>2) 티커별 합성 기여도</h3>",
            matrix_view.to_html(index=False, escape=True, border=0),
        ]
    )

    # Insert before closing tags
    return base.replace("</div></body></html>", insert + "\n</div></body></html>")


def _safe_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{float(x):+.2%}"
    except Exception:
        return "NA"


def _safe_float(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{float(x):.4f}"
    except Exception:
        return "NA"


def _build_decision_notes(chart: pd.DataFrame, month_chart: pd.DataFrame) -> str:
    """
    Human-readable, compact notes that explain *why* each sub-strategy picked its holdings.
    Intended for audit/validation, not for machine parsing.
    """
    lines: list[str] = []
    ref = month_chart.index.max() if not month_chart.empty else chart.index.max()
    lines.append(f"Reference date (month context): {ref.date().isoformat() if hasattr(ref, 'date') else str(ref)}")
    lines.append("")

    # BAA
    try:
        baa = InvestmentStrategyBAA(chart=chart, month_chart=month_chart)
        mmt = baa.month_chart[list(set(BAAAssets.canaria + BAAAssets.aggressive))].apply(baa.momentum_score, axis=1).dropna()
        last = mmt.iloc[-1]
        canary = last[BAAAssets.canaria].sort_values(ascending=False)
        aggressive_signal = bool((last[BAAAssets.canaria] >= 0).all())
        lines.append("[BAA] Bold Asset Allocation")
        lines.append(f"- Canary momentum >= 0 for all? {aggressive_signal}  (SPY,VWO,VEA,BND)")
        lines.append("  Canary scores: " + ", ".join(f"{k}={_safe_float(v)}" for k, v in canary.items()))
        if aggressive_signal:
            ranked = last[BAAAssets.aggressive].sort_values(ascending=False).head(6)
            lines.append("  Aggressive top6 by momentum score: " + ", ".join(f"{k}={_safe_float(v)}" for k, v in ranked.items()))
        else:
            # Defensive selection: above 12M MA filter and 12M return top3
            pos = baa.month_chart.index.get_loc(mmt.index[-1])
            current = baa.month_chart.iloc[pos][BAAAssets.defensive]
            ma12 = baa.month_chart[BAAAssets.defensive].iloc[pos - 11:pos + 1].mean()
            ret12 = baa.rate_of_return(current, 12).sort_values(ascending=False).head(3)
            picks = []
            for name, v in ret12.items():
                picks.append(f"{'BIL' if current[name] < ma12[name] else name}(12m_ret={_safe_pct(v)}, above_ma12={current[name] >= ma12[name]})")
            lines.append("  Defensive top3 by 12M return (with MA12 filter): " + ", ".join(picks))
        lines.append("")
    except Exception as e:  # noqa: BLE001
        lines.append("[BAA] Failed to compute decision notes: " + str(e))
        lines.append("")

    # BDA
    try:
        bda = InvestmentStrategyBDA(chart=chart, month_chart=month_chart)
        returns_6m = bda.month_chart.pct_change(6)
        last = returns_6m.iloc[-1][BDAAssets.bonds].dropna().sort_values(ascending=False)
        top3 = list(last.head(3).index)
        kept = [t for t in top3 if float(returns_6m.iloc[-1].get(t, np.nan)) > 0]
        cash_fill = [BDAAssets.cash for _ in range(3 - len(kept))]
        final = kept + cash_fill
        lines.append("[BDA] Bond Dynamic Asset Allocation")
        lines.append("  6M returns (top8 shown): " + ", ".join(f"{k}={_safe_pct(v)}" for k, v in last.head(8).items()))
        lines.append("  Pick top3; replace non-positive with BIL: " + ", ".join(final))
        lines.append("")
    except Exception as e:  # noqa: BLE001
        lines.append("[BDA] Failed to compute decision notes: " + str(e))
        lines.append("")

    # LAA
    try:
        laa = InvestmentStrategyLAA(chart=chart, month_chart=month_chart)
        unemployment = laa._load_unemployment()  # reuse the normalized loader
        spy_200_ma = laa.chart["SPY"].rolling(window=200).mean().resample("ME").last()
        spy_200_ma = spy_200_ma.rename(index={spy_200_ma.index[-1]: laa.chart.index[-1]})
        base = spy_200_ma.rename("spy_200_ma").to_frame().reset_index().rename(columns={"index": "Date"})
        merged = pd.merge_asof(base.sort_values("Date"), unemployment.sort_values("Date"), on="Date", direction="backward").set_index("Date")
        merged["unemployment_12_ma"] = merged["unemployment"].rolling(window=12).mean()
        merged["SPY"] = laa.month_chart["SPY"].reindex(merged.index)
        merged = merged.dropna(subset=["SPY", "spy_200_ma", "unemployment", "unemployment_12_ma"])
        row = merged.iloc[-1]
        cond1 = bool(row["SPY"] > row["spy_200_ma"])
        cond2 = bool(row["unemployment"] > row["unemployment_12_ma"])
        pick = LAAAssets.static + (["QQQ"] if (cond1 or cond2) else ["SHY"])
        lines.append("[LAA] Lethargic Asset Allocation")
        lines.append(f"  Condition1: SPY > 200DMA? {cond1}  (SPY={row['SPY']:.2f}, 200DMA={row['spy_200_ma']:.2f})")
        lines.append(
            f"  Condition2: UNRATE > 12M MA? {cond2}  (UNRATE={row['unemployment']:.2f}, UNRATE_12M_MA={row['unemployment_12_ma']:.2f})"
        )
        lines.append("  Holdings = static(IWD,GLD,IEF) + (QQQ if cond1|cond2 else SHY): " + ", ".join(pick))
        lines.append("")
    except Exception as e:  # noqa: BLE001
        lines.append("[LAA] Failed to compute decision notes: " + str(e))
        lines.append("")

    # MDM
    try:
        mdm = InvestmentStrategyMDM(chart=chart, month_chart=month_chart)
        returns_12m = mdm.month_chart.pct_change(12)
        returns_6m = mdm.month_chart.pct_change(6)
        r_spy = float(returns_12m.iloc[-1].get("SPY", np.nan))
        aggressive = bool(r_spy > 0)
        lines.append("[MDM] Modified Dual Momentum")
        lines.append(f"  SPY 12M return > 0? {aggressive}  (SPY_12m={_safe_pct(r_spy)})")
        if aggressive:
            cand = returns_12m.iloc[-1][MDMAssets.aggressive].dropna().sort_values(ascending=False)
            pick = str(cand.index[0]) if not cand.empty else "NA"
            lines.append("  Pick best of (SPY,EFA) by 12M return: " + ", ".join(f"{k}={_safe_pct(v)}" for k, v in cand.items()))
            lines.append(f"  Holdings: {pick}")
        else:
            cand = returns_6m.iloc[-1][MDMAssets.defensive].dropna().sort_values(ascending=False).head(3)
            lines.append("  Defensive top3 by 6M return: " + ", ".join(f"{k}={_safe_pct(v)}" for k, v in cand.items()))
            lines.append("  Holdings: " + ", ".join(list(cand.index)))
        lines.append("")
    except Exception as e:  # noqa: BLE001
        lines.append("[MDM] Failed to compute decision notes: " + str(e))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()

    # Strategies may write intermediate CSVs under ./output (legacy path).
    Path("output").mkdir(parents=True, exist_ok=True)

    asof_kst = _parse_asof_kst(args.asof_kst) if args.asof_kst else datetime.now(tz=KST)

    tickers = sorted(InvestmentStrategyIntegration.get_assets())
    if args.start:
        start = args.start
    else:
        start_ts = pd.Timestamp.utcnow().normalize()
        if getattr(start_ts, "tzinfo", None) is not None:
            start_ts = start_ts.tz_localize(None)
        start = (start_ts - pd.DateOffset(years=3)).date().isoformat()

    prices = load_prices_with_cache(tickers, start=start, end=None, cache_dir=Path(args.cache_dir))
    if prices.empty:
        raise RuntimeError("Price data is empty; cannot generate signals.")

    data_date = prices.index.max()
    month_prices = prices.resample("ME").last()
    if not month_prices.empty:
        month_prices = month_prices.rename(index={month_prices.index[-1]: data_date})

    strategy = InvestmentStrategyIntegration(chart=prices, month_chart=month_prices, run_backtests=False)
    port_series = strategy.portfolio
    if port_series.empty:
        raise RuntimeError("Portfolio series is empty; cannot generate signals.")

    latest_port = port_series.iloc[-1]
    weights = pd.Series(latest_port.allocations, dtype=float)
    weights = weights.replace([np.inf, -np.inf], np.nan).dropna()
    weights = weights / weights.sum() if weights.sum() else weights

    out_root = Path(args.output_dir)
    out_dir = out_root / asof_kst.strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)

    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    long_df, matrix_df = _compute_breakdown(prices, month_prices, data_date)
    # Always materialize these files so downstream workflows can attach them without conditional logic.
    long_df.to_csv(details_dir / "strategy_holdings.csv", index=False)
    matrix_df.to_csv(details_dir / "merge_contributions.csv", index=False)

    decision_notes = _build_decision_notes(prices, month_prices)
    (details_dir / "decision_notes.txt").write_text(decision_notes, encoding="utf-8")

    # Emit a superset schema compatible with scripts/daily_rank_report.py
    ordered = weights.sort_values(ascending=False)
    records: list[dict[str, object]] = []
    for i, (ticker, weight) in enumerate(ordered.items(), start=1):
        records.append(
            {
                "asof_kst": asof_kst.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "data_date": data_date.date().isoformat(),
                "strategy": "asset_allocation_integration",
                "mode": None,
                "lookback": None,
                "top": None,
                "ticker": ticker,
                "rank": i,
                "weight": float(weight),
                "lookback_return": None,
                "lookback_price": None,
                "current_price": float(prices.loc[data_date, ticker]) if ticker in prices.columns else None,
                "lookback_date": None,
                "current_date": data_date.date().isoformat(),
            }
        )

    signals_df = pd.DataFrame.from_records(records)
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

    signals_path = out_dir / "signals.csv"
    signals_df.to_csv(signals_path, index=False)

    html = _render_html_with_breakdown(asof_kst, data_date, weights, long_df, matrix_df, decision_notes)
    html_path = out_dir / "email.html"
    html_path.write_text(html, encoding="utf-8")

    logger.info(f"Wrote {signals_path}")
    logger.info(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
