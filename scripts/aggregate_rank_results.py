from __future__ import annotations

import argparse
import re
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


LABEL_RE = re.compile(r"rank_(head|tail)_lk(\d+)_top(\d+)(?:_rbm(\d+))?")


def parse_label(name: str) -> Optional[Dict[str, object]]:
    m = LABEL_RE.fullmatch(name)
    if not m:
        return None
    mode, lb, top, rbm = m.groups()
    return {
        "label": name,
        "mode": mode,
        "lookback": int(lb),
        "top": int(top),
        "rebalance_months": int(rbm) if rbm is not None else 1,
    }


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return dd.min()


def load_daily_metrics(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    rets = df.iloc[:, 0].rename("return").fillna(0)
    if rets.empty:
        return {
            "daily_start": None,
            "daily_end": None,
            "days": 0,
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "max_drawdown": float("nan"),
        }
    days = len(rets)
    cagr = (1 + rets).prod() ** (252 / days) - 1 if days > 0 else float("nan")
    ann_vol = rets.std() * sqrt(252)
    mdd = max_drawdown(rets)
    return {
        "daily_start": rets.index.min().date(),
        "daily_end": rets.index.max().date(),
        "days": days,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
    }


def load_period_metrics(path: Path) -> Dict[str, object]:
    df = pd.read_csv(path, parse_dates=["lookback_start", "lookback_end", "buy_date", "sell_date"])
    if df.empty:
        return {
            "months": 0,
            "period_return_mean": float("nan"),
            "period_return_std": float("nan"),
            "period_active_mean": float("nan"),
            "period_benchmark_mean": float("nan"),
            "period_benchmark_std": float("nan"),
            "last_period_return": float("nan"),
            "last_period_active_return": float("nan"),
            "last_period_benchmark_return": float("nan"),
        }
    return {
        "months": len(df),
        "period_return_mean": df["return"].mean(),
        "period_return_std": df["return"].std(),
        "period_active_mean": df["active_return"].mean() if "active_return" in df else float("nan"),
        "period_benchmark_mean": df["benchmark_return"].mean() if "benchmark_return" in df else float("nan"),
        "period_benchmark_std": df["benchmark_return"].std() if "benchmark_return" in df else float("nan"),
        "last_period_return": df["return"].iloc[-1],
        "last_period_active_return": df["active_return"].iloc[-1] if "active_return" in df else float("nan"),
        "last_period_benchmark_return": df["benchmark_return"].iloc[-1] if "benchmark_return" in df else float("nan"),
    }


def collect_results(output_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(output_root.glob("rank_*")):
        if not path.is_dir():
            continue
        meta = parse_label(path.name)
        if not meta:
            continue

        daily_path = path / "rank_backtest.csv"
        monthly_path = path / "rank_monthly.csv"
        if not daily_path.exists() or not monthly_path.exists():
            continue

        metrics = {**meta}
        metrics["daily_path"] = str(daily_path)
        metrics["monthly_path"] = str(monthly_path)
        metrics.update(load_daily_metrics(daily_path))
        metrics.update(load_period_metrics(monthly_path))
        rows.append(metrics)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate multiple rank backtest outputs into a summary CSV/Markdown.")
    parser.add_argument(
        "--output-root",
        default=".output",
        help="폴더를 스캔할 루트 경로 (rank_* 하위 폴더들이 위치).",
    )
    parser.add_argument(
        "--summary-csv",
        default=".output/rank_summary.csv",
        help="집계 CSV 저장 경로.",
    )
    parser.add_argument(
        "--summary-md",
        default=".output/rank_summary.md",
        help="요약 Markdown 저장 경로.",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    df = collect_results(root)
    if df.empty:
        print("No rank_* results found under", root)
        return

    df_sorted = df.sort_values(["mode", "lookback", "top", "rebalance_months"])
    out_csv = Path(args.summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(out_csv, index=False)
    print(f"Saved summary CSV to {out_csv}")

    out_md = Path(args.summary_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    preview_cols = [
        "label",
        "mode",
        "lookback",
        "top",
        "rebalance_months",
        "cagr",
        "ann_vol",
        "max_drawdown",
        "period_return_mean",
        "period_active_mean",
        "period_benchmark_mean",
        "last_period_return",
        "last_period_active_return",
        "last_period_benchmark_return",
        "months",
        "days",
    ]
    df_sorted[preview_cols].to_markdown(out_md, index=False)
    print(f"Saved summary Markdown to {out_md}")


if __name__ == "__main__":
    main()
