from __future__ import annotations

import argparse
import re
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


LABEL_RE = re.compile(r"rank_(head|tail)_lk(\d+)_top(\d+)(?:_rbm(\d+))?")
LABEL_MIX_RE = re.compile(r"rankmix_s(\d+)_m(\d+)_w(\d+)_rbm(\d+)")


def parse_label(name: str) -> Optional[Dict[str, object]]:
    m = LABEL_RE.fullmatch(name)
    if m:
        mode, lb, top, rbm = m.groups()
        return {
            "label": name,
            "mode": mode,
            "lookback": int(lb),
            "top": int(top),
            "rebalance_months": int(rbm) if rbm is not None else 1,
        }

    m2 = LABEL_MIX_RE.fullmatch(name)
    if m2:
        s_lb, m_lb, w_raw, rbm = m2.groups()
        # label에서 w08, w07 형태이므로 10으로 나눠 복원(0.8, 0.7 등)
        try:
            weight_mom = int(w_raw) / 10
        except Exception:
            weight_mom = float("nan")
        weight_rev = 1 - weight_mom if isinstance(weight_mom, float) else float("nan")
        return {
            "label": name,
            "mode": "mix",
            "lookback": int(m_lb),  # 중기 창을 대표 룩백으로 기록
            "lookback_short": int(s_lb),
            "lookback_mid": int(m_lb),
            "top": 10,  # 현재 스크립트 기본값
            "rebalance_months": int(rbm),
            "weight_mom": weight_mom,
            "weight_rev": weight_rev,
        }
    return None


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
    # 일부 전략 파일은 lookback_start/lookback_end가 없을 수 있으므로 존재하는 컬럼만 파싱
    parse_cols = [c for c in ["lookback_start", "lookback_end", "buy_date", "sell_date"] if c in pd.read_csv(path, nrows=0).columns]
    df = pd.read_csv(path, parse_dates=parse_cols) if parse_cols else pd.read_csv(path)
    bench_cagr = float("nan")
    spy_cagr = float("nan")
    if not df.empty:
        start_dates = df["buy_date"].dropna()
        end_dates = df["sell_date"].dropna()
        total_days = None
        if not start_dates.empty and not end_dates.empty:
            total_days = (end_dates.max() - start_dates.min()).days + 1

        def _cagr(total_ret: float) -> float:
            if total_days and total_days > 0:
                return (1 + total_ret) ** (365 / total_days) - 1
            return float("nan")

        bench_total = (1 + df["benchmark_return"]).prod() - 1 if "benchmark_return" in df else float("nan")
        spy_total = (1 + df["spy_return"]).prod() - 1 if "spy_return" in df else float("nan")
        bench_cagr = _cagr(bench_total)
        spy_cagr = _cagr(spy_total)
    if df.empty:
        return {
            "months": 0,
            "period_return_mean": float("nan"),
            "period_active_mean": float("nan"),
            "period_benchmark_mean": float("nan"),
            "period_spy_mean": float("nan"),
            "period_active_spy_mean": float("nan"),
            "period_benchmark_cagr": float("nan"),
            "period_spy_cagr": float("nan"),
            "last_period_return": float("nan"),
            "last_period_active_return": float("nan"),
            "last_period_benchmark_return": float("nan"),
            "last_period_spy_return": float("nan"),
            "last_period_active_spy_return": float("nan"),
        }
    return {
        "months": len(df),
        "period_return_mean": df["return"].mean(),
        "period_active_mean": df["active_return"].mean() if "active_return" in df else float("nan"),
        "period_benchmark_mean": df["benchmark_return"].mean() if "benchmark_return" in df else float("nan"),
        "period_spy_mean": df["spy_return"].mean() if "spy_return" in df else float("nan"),
        "period_active_spy_mean": df["active_return_vs_spy"].mean() if "active_return_vs_spy" in df else float("nan"),
        "period_benchmark_cagr": bench_cagr,
        "period_spy_cagr": spy_cagr,
        "last_period_return": df["return"].iloc[-1],
        "last_period_active_return": df["active_return"].iloc[-1] if "active_return" in df else float("nan"),
        "last_period_benchmark_return": df["benchmark_return"].iloc[-1] if "benchmark_return" in df else float("nan"),
        "last_period_spy_return": df["spy_return"].iloc[-1] if "spy_return" in df else float("nan"),
        "last_period_active_spy_return": df["active_return_vs_spy"].iloc[-1] if "active_return_vs_spy" in df else float("nan"),
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

    for path in sorted(output_root.glob("rankmix_*")):
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
        "lookback_short",
        "lookback_mid",
        "top",
        "rebalance_months",
        "weight_mom",
        "weight_rev",
        "cagr",
        "ann_vol",
        "max_drawdown",
        "period_return_mean",
        "period_active_mean",
        "period_benchmark_mean",
        "period_spy_mean",
        "period_active_spy_mean",
        "period_benchmark_cagr",
        "period_spy_cagr",
        "last_period_return",
        "last_period_active_return",
        "last_period_benchmark_return",
        "last_period_spy_return",
        "last_period_active_spy_return",
        "months",
        "days",
    ]
    for col in preview_cols:
        if col not in df_sorted.columns:
            df_sorted[col] = pd.NA
    df_sorted[preview_cols].to_markdown(out_md, index=False)
    print(f"Saved summary Markdown to {out_md}")


if __name__ == "__main__":
    main()
