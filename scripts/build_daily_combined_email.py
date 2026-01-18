from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


KST = ZoneInfo("Asia/Seoul")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a single combined HTML email for daily asset-allocation + daily rank reports."
    )
    p.add_argument("--date", default=None, help="KST date folder (YYYYMMDD). Default: now in KST.")
    p.add_argument("--rank-dir", default=".output/daily", help="Root dir produced by scripts/daily_rank_report.py")
    p.add_argument(
        "--aa-dir",
        default=".output/daily_asset_allocation",
        help="Root dir produced by scripts/daily_asset_allocation_report.py",
    )
    p.add_argument("--out", default=None, help="Where to write combined HTML (default: .output/daily_reports/YYYYMMDD/email.html)")
    return p.parse_args()


def _fmt_pct(x: object, digits: int = 2) -> str:
    try:
        if x is None:
            return "NA"
        v = float(x)
        if np.isnan(v):
            return "NA"
        return f"{v:.{digits}%}"
    except Exception:
        return "NA"


def _fmt_price(x: object) -> str:
    try:
        if x is None:
            return "NA"
        v = float(x)
        if np.isnan(v):
            return "NA"
        return f"{v:,.2f}"
    except Exception:
        return "NA"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><i>No rows</i></p>"
    return df.to_html(index=False, escape=True, border=0, classes="data-table")


def main() -> None:
    args = parse_args()

    date = args.date or datetime.now(tz=KST).strftime("%Y%m%d")
    rank_root = Path(args.rank_dir) / date
    aa_root = Path(args.aa_dir) / date

    rank_signals_path = rank_root / "signals.csv"
    aa_signals_path = aa_root / "signals.csv"

    rank_df = _read_csv(rank_signals_path)
    aa_df = _read_csv(aa_signals_path)

    # Meta (best-effort)
    asof_kst = None
    data_date = None
    for df in (aa_df, rank_df):
        if df is None or df.empty:
            continue
        if asof_kst is None and "asof_kst" in df.columns:
            val = str(df["asof_kst"].dropna().iloc[0]) if df["asof_kst"].dropna().shape[0] else ""
            asof_kst = val or None
        if data_date is None and "data_date" in df.columns:
            val = str(df["data_date"].dropna().iloc[0]) if df["data_date"].dropna().shape[0] else ""
            data_date = val or None

    style = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.35; margin: 0; }
.container { padding: 18px 24px 32px; }
h1 { margin: 0 0 10px; }
h2 { margin: 18px 0 8px; }
h3 { margin: 12px 0 6px; }
.meta { color: #555; font-size: 13px; margin-bottom: 10px; }
.data-table { border-collapse: collapse; width: 100%; max-width: 1100px; }
.data-table th, .data-table td { padding: 6px 8px; border-bottom: 1px solid #e5e5e5; text-align: left; font-size: 13px; }
.data-table th { background: #fafafa; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }

/* Tab styles */
.tab-container { margin: 16px 0; }
.tab-buttons { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 2px solid #e5e5e5; padding-bottom: 0; }
.tab-btn { 
    padding: 8px 16px; 
    border: 1px solid #e5e5e5; 
    border-bottom: none;
    background: #f8f8f8; 
    cursor: pointer; 
    font-size: 13px;
    border-radius: 6px 6px 0 0;
    margin-bottom: -2px;
    transition: background 0.2s;
}
.tab-btn:hover { background: #eee; }
.tab-btn.active { 
    background: #fff; 
    border-bottom: 2px solid #fff;
    font-weight: 600;
    color: #0066cc;
}
.tab-content { display: none; padding: 16px 0; }
.tab-content.active { display: block; }
</style>
""".strip()

    script = """
<script>
function showTab(containerId, tabId) {
    const container = document.getElementById(containerId);
    container.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    container.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    container.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
}
</script>
""".strip()

    parts: list[str] = [
        "<html><head><meta charset='utf-8'/>",
        style,
        script,
        "</head><body>",
        "<div class='container'>",
        "<h1>Daily Reports (Asset Allocation + Rank)</h1>",
        "<div class='meta'>"
        + f"KST folder: <code>{date}</code><br/>"
        + (f"As-of (KST): {asof_kst}<br/>" if asof_kst else "")
        + (f"Data date (latest close): {data_date}<br/>" if data_date else "")
        + "Attachments: <code>signals.csv</code> (rank/asset allocation)"
        + "</div>",
    ]

    # Asset allocation section
    parts.append("<h2>Asset Allocation (Integration)</h2>")
    if aa_df.empty:
        parts.append(f"<p><i>Missing: {aa_signals_path}</i></p>")
    else:
        view = aa_df.copy()
        if "weight" in view.columns:
            view["weight"] = view["weight"].map(lambda x: _fmt_pct(x, 2))
        if "current_price" in view.columns:
            view["current_price"] = view["current_price"].map(_fmt_price)
        cols = [c for c in ["rank", "ticker", "weight", "current_price"] if c in view.columns]
        if cols:
            view = view[cols]
        parts.append(_table(view))

    # Rank section with tabs
    parts.append("<h2>Rank (Head/Tail)</h2>")
    if rank_df.empty:
        parts.append(f"<p><i>Missing: {rank_signals_path}</i></p>")
    else:
        view = rank_df.copy()
        
        # Format prices
        if "lookback_price" in view.columns:
            view["lookback_price"] = view["lookback_price"].map(_fmt_price)
        if "current_price" in view.columns:
            view["current_price"] = view["current_price"].map(_fmt_price)
        if "lookback_return" in view.columns:
            view["lookback_return"] = view["lookback_return"].map(lambda x: _fmt_pct(x, 2))

        for mode in ["head", "tail"]:
            sub = view[view.get("mode", pd.Series(dtype=object)) == mode] if "mode" in view.columns else pd.DataFrame()
            title = "Head (Momentum)" if mode == "head" else "Tail (Reversal)"
            parts.append(f"<h3>{title}</h3>")
            if sub.empty:
                parts.append("<p><i>No rows</i></p>")
                continue
            
            lookbacks = sorted({int(x) for x in sub.get("lookback", pd.Series(dtype=float)).dropna().tolist()})
            
            if len(lookbacks) <= 1:
                # No tabs needed for single lookback
                for lb in lookbacks:
                    block = sub[sub["lookback"] == lb].copy()
                    block = block.sort_values(["rank", "ticker"])
                    
                    # Get dates for column renaming
                    lookback_date_val = block["lookback_date"].iloc[0] if "lookback_date" in block.columns and not block.empty else "Lookback"
                    current_date_val = block["current_date"].iloc[0] if "current_date" in block.columns and not block.empty else "Current"
                    
                    # Build display dataframe with date column names
                    display_cols = ["rank", "ticker"]
                    rename_map = {}
                    if "lookback_price" in block.columns:
                        display_cols.append("lookback_price")
                        rename_map["lookback_price"] = str(lookback_date_val)
                    if "current_price" in block.columns:
                        display_cols.append("current_price")
                        rename_map["current_price"] = str(current_date_val)
                    if "lookback_return" in block.columns:
                        display_cols.append("lookback_return")
                        rename_map["lookback_return"] = "Return"
                    
                    block = block[[c for c in display_cols if c in block.columns]]
                    block = block.rename(columns=rename_map)
                    
                    parts.append(f"<div class='meta'>lookback={lb} business days</div>")
                    parts.append(_table(block))
            else:
                # Use tabs for multiple lookbacks
                container_id = f"tabs-{mode}"
                parts.append(f"<div class='tab-container' id='{container_id}'>")
                
                # Tab buttons
                parts.append("<div class='tab-buttons'>")
                for i, lb in enumerate(lookbacks):
                    active_class = " active" if i == 0 else ""
                    tab_id = f"tab-{mode}-{lb}"
                    parts.append(f"<button class='tab-btn{active_class}' data-tab='{tab_id}' onclick=\"showTab('{container_id}', '{tab_id}')\">LB={lb}</button>")
                parts.append("</div>")
                
                # Tab contents
                for i, lb in enumerate(lookbacks):
                    active_class = " active" if i == 0 else ""
                    tab_id = f"tab-{mode}-{lb}"
                    block = sub[sub["lookback"] == lb].copy()
                    block = block.sort_values(["rank", "ticker"])
                    
                    # Get dates for column renaming
                    lookback_date_val = block["lookback_date"].iloc[0] if "lookback_date" in block.columns and not block.empty else "Lookback"
                    current_date_val = block["current_date"].iloc[0] if "current_date" in block.columns and not block.empty else "Current"
                    
                    # Build display dataframe with date column names
                    display_cols = ["rank", "ticker"]
                    rename_map = {}
                    if "lookback_price" in block.columns:
                        display_cols.append("lookback_price")
                        rename_map["lookback_price"] = str(lookback_date_val)
                    if "current_price" in block.columns:
                        display_cols.append("current_price")
                        rename_map["current_price"] = str(current_date_val)
                    if "lookback_return" in block.columns:
                        display_cols.append("lookback_return")
                        rename_map["lookback_return"] = "Return"
                    
                    block = block[[c for c in display_cols if c in block.columns]]
                    block = block.rename(columns=rename_map)
                    
                    parts.append(f"<div class='tab-content{active_class}' id='{tab_id}'>")
                    parts.append(f"<div class='meta'>lookback={lb} business days</div>")
                    parts.append(_table(block))
                    parts.append("</div>")
                
                parts.append("</div>")  # close tab-container

    parts.append("</div></body></html>")

    out_path = Path(args.out) if args.out else (Path(".output/daily_reports") / date / "email.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
