from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


KST = ZoneInfo("Asia/Seoul")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a concise weekly rank backtest summary HTML for email body.")
    p.add_argument("--summary-csv", default=".output/rank_summary.csv", help="Path to rank_summary.csv")
    p.add_argument("--out", default=".output/weekly_rank_email.html", help="Where to write the HTML email body")
    p.add_argument("--top-n", type=int, default=10, help="Rows to show per mode (sorted by sharpe_like)")
    return p.parse_args()


def _fmt_pct(x: object) -> str:
    try:
        if x is None:
            return "NA"
        v = float(x)
        if np.isnan(v):
            return "NA"
        return f"{v:+.2%}"
    except Exception:
        return "NA"


def _fmt_pct0(x: object) -> str:
    try:
        if x is None:
            return "NA"
        v = float(x)
        if np.isnan(v):
            return "NA"
        return f"{v:.0%}"
    except Exception:
        return "NA"


def _fmt_num(x: object, digits: int = 2) -> str:
    try:
        if x is None:
            return "NA"
        v = float(x)
        if np.isnan(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return "NA"


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><i>No rows</i></p>"

    view = df.copy()
    view["cagr"] = view["cagr"].map(_fmt_pct)
    view["ann_vol"] = view["ann_vol"].map(_fmt_pct)
    view["max_drawdown"] = view["max_drawdown"].map(_fmt_pct)
    view["period_active_mean"] = view["period_active_mean"].map(_fmt_pct)
    view["sharpe_like"] = view["sharpe_like"].map(lambda x: _fmt_num(x, 2))
    view["last_period_active_return"] = view["last_period_active_return"].map(_fmt_pct)

    cols = [
        "label",
        "lookback",
        "top",
        "rebalance_months",
        "cagr",
        "ann_vol",
        "max_drawdown",
        "period_active_mean",
        "last_period_active_return",
        "sharpe_like",
    ]
    existing = [c for c in cols if c in view.columns]
    view = view[existing]
    return view.to_html(index=False, escape=True, border=0, classes="data-table")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"Summary CSV is empty: {summary_path}")

    # Derive a simple risk-adjusted score.
    df = df.copy()
    df["sharpe_like"] = df.apply(
        lambda row: float(row["cagr"]) / float(row["ann_vol"]) if float(row.get("ann_vol", 0) or 0) > 1e-9 else 0.0,
        axis=1,
    )

    head = df[df["mode"] == "head"].sort_values(["sharpe_like", "cagr"], ascending=[False, False]).head(args.top_n)
    tail = df[df["mode"] == "tail"].sort_values(["sharpe_like", "cagr"], ascending=[False, False]).head(args.top_n)

    now_kst = datetime.now(tz=KST)
    daily_start = str(df.get("daily_start", pd.Series(dtype=object)).min()) if "daily_start" in df.columns else "NA"
    daily_end = str(df.get("daily_end", pd.Series(dtype=object)).max()) if "daily_end" in df.columns else "NA"

    style = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35; margin: 0; }
.container { padding: 18px 24px 32px; }
h1 { margin: 0 0 10px; }
h2 { margin: 18px 0 8px; }
.meta { color: #555; font-size: 13px; margin-bottom: 10px; }
.data-table { border-collapse: collapse; width: 100%; max-width: 1100px; }
.data-table th, .data-table td { padding: 6px 8px; border-bottom: 1px solid #e5e5e5; text-align: left; font-size: 13px; }
.data-table th { background: #fafafa; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
</style>
""".strip()

    html = "\n".join(
        [
            "<html><head><meta charset='utf-8'/>",
            style,
            "</head><body>",
            "<div class='container'>",
            "<h1>Weekly Rank Backtest Summary</h1>",
            (
                "<div class='meta'>"
                f"Generated at (KST): {now_kst.strftime('%Y-%m-%d %H:%M %Z')}<br/>"
                f"Backtest daily range: {daily_start} ~ {daily_end}<br/>"
                "Attachments: <code>rank_summary.csv</code>, <code>rank_summary.md</code>, <code>rank_analysis.html</code>"
                "</div>"
            ),
            "<h2>Top candidates (Head / Momentum)</h2>",
            _table(head),
            "<h2>Top candidates (Tail / Reversal)</h2>",
            _table(tail),
            "<div class='meta'>Notes: <code>sharpe_like=cagr/ann_vol</code> (simple score). Always review drawdown and stability.</div>",
            "</div></body></html>",
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

