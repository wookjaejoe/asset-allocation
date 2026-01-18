#!/usr/bin/env bash
set -euo pipefail

# 4가지 자산배분 전략(BAA, BDA, LAA, MDM) 백테스트 실행
# Integration(동일가중 통합) 백테스트도 함께 생성
# 
# 출력 파일:
#   .output/asset_allocation/{strategy_name}.csv  - 일별 수익률
#   .output/asset_allocation/{strategy_name}.html - quantstats 리포트
#   .output/asset_allocation/final.csv            - 통합 포트폴리오 일별 수익률
#   .output/asset_allocation/final.html           - 통합 quantstats 리포트
#   .output/asset_allocation/summary.csv          - 전략별 성과 요약

OUTPUT_DIR="${OUTPUT_DIR:-.output/asset_allocation}"
CACHE_DIR="${CACHE_DIR:-.cache/asset_allocation}"
START_DATE="${START_DATE:-2007-01-01}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

echo "Running asset allocation backtests..."
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Cache dir: ${CACHE_DIR}"
echo "  Start date: ${START_DATE}"

python - <<'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path
from math import sqrt

import pandas as pd
import numpy as np

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.price_cache import load_prices_with_cache
from logger import logger
from strategy import InvestmentStrategyIntegration
from strategy.baa import InvestmentStrategyBAA
from strategy.bda import InvestmentStrategyBDA
from strategy.laa import InvestmentStrategyLAA
from strategy.mdm import InvestmentStrategyMDM


def max_drawdown(series: pd.Series) -> float:
    """Calculate maximum drawdown from a return series."""
    if series.empty:
        return float("nan")
    cum = (1 + series.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    return dd.min()


def compute_metrics(returns: pd.Series, name: str) -> dict:
    """Compute performance metrics from daily returns."""
    if returns.empty:
        return {
            "strategy": name,
            "start_date": None,
            "end_date": None,
            "days": 0,
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe_like": float("nan"),
            "total_return": float("nan"),
        }
    
    days = len(returns)
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (252 / days) - 1 if days > 0 else float("nan")
    ann_vol = returns.std() * sqrt(252)
    mdd = max_drawdown(returns)
    sharpe_like = cagr / ann_vol if ann_vol > 1e-9 else 0.0
    
    return {
        "strategy": name,
        "start_date": returns.index.min().date().isoformat(),
        "end_date": returns.index.max().date().isoformat(),
        "days": days,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "sharpe_like": sharpe_like,
        "total_return": total_return,
    }


def main():
    output_dir = Path(os.environ.get("OUTPUT_DIR", ".output/asset_allocation"))
    cache_dir = Path(os.environ.get("CACHE_DIR", ".cache/asset_allocation"))
    start_date = os.environ.get("START_DATE", "2007-01-01")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Legacy output path for strategy internal CSVs
    Path("output").mkdir(parents=True, exist_ok=True)
    
    # Fetch price data
    assets = InvestmentStrategyIntegration.get_assets()
    logger.info(f"Fetching prices for {len(assets)} assets from {start_date}...")
    
    chart = load_prices_with_cache(list(assets), start=start_date, end=None, cache_dir=cache_dir)
    chart = chart.dropna(how="all").dropna(axis=1, how="all").sort_index()
    
    if chart.empty:
        raise RuntimeError("Price data is empty; cannot run backtests.")
    
    month_chart = chart.resample("ME").last()
    month_chart = month_chart.rename(index={month_chart.index[-1]: chart.index[-1]})
    
    logger.info(f"Price data: {chart.index.min().date()} to {chart.index.max().date()}, {len(chart)} days")
    
    # Run backtests for each sub-strategy
    strategy_types = [
        InvestmentStrategyBAA,
        InvestmentStrategyBDA,
        InvestmentStrategyLAA,
        InvestmentStrategyMDM,
    ]
    
    summary_rows = []
    
    for strategy_cls in strategy_types:
        name = strategy_cls.get_name()
        logger.info(f"Running backtest for {name}...")
        
        try:
            strategy = strategy_cls(chart=chart, month_chart=month_chart)
            output_prefix = str(output_dir / name)
            strategy.backtest(output_prefix)
            
            # Read back the returns CSV to compute metrics
            returns_path = Path(f"{output_prefix}.csv")
            if returns_path.exists():
                df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
                if "return" in df.columns:
                    returns = df["return"].fillna(0)
                    metrics = compute_metrics(returns, name)
                    summary_rows.append(metrics)
                    logger.info(f"  {name}: CAGR={metrics['cagr']:.2%}, Vol={metrics['ann_vol']:.2%}, MDD={metrics['max_drawdown']:.2%}")
            
            logger.info(f"  Wrote {output_prefix}.csv, {output_prefix}.html")
        except Exception as e:
            logger.error(f"  Failed to run {name}: {e}")
            summary_rows.append({"strategy": name, "error": str(e)})
    
    # Run integration (combined) backtest
    logger.info("Running Integration (combined) backtest...")
    try:
        integration = InvestmentStrategyIntegration(chart=chart, month_chart=month_chart, run_backtests=False)
        output_prefix = str(output_dir / "final")
        integration.backtest(output_prefix)
        
        # Read back the returns CSV to compute metrics
        returns_path = Path(f"{output_prefix}.csv")
        if returns_path.exists():
            df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            if "return" in df.columns:
                returns = df["return"].fillna(0)
                metrics = compute_metrics(returns, "Integration")
                summary_rows.append(metrics)
                logger.info(f"  Integration: CAGR={metrics['cagr']:.2%}, Vol={metrics['ann_vol']:.2%}, MDD={metrics['max_drawdown']:.2%}")
        
        logger.info(f"  Wrote {output_prefix}.csv, {output_prefix}.html")
    except Exception as e:
        logger.error(f"  Failed to run Integration: {e}")
        summary_rows.append({"strategy": "Integration", "error": str(e)})
    
    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Wrote summary to {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Asset Allocation Backtest Summary")
    print("=" * 80)
    if not summary_df.empty:
        display_cols = ["strategy", "cagr", "ann_vol", "max_drawdown", "sharpe_like", "total_return"]
        display_cols = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[display_cols].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
PYTHON_SCRIPT

echo "Asset allocation backtests completed."
echo "Results in: ${OUTPUT_DIR}"
