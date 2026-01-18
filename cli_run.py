import argparse
from datetime import datetime
import os

import pandas as pd

from logger import logger
from strategy import InvestmentStrategyMerged, fetch_charts


def run_cli(output_prefix: str, summary_path: str) -> int:
    logger.info("Fetching res...")
    chart, month_chart = fetch_charts()

    logger.info("Backtesting...")
    strategy = InvestmentStrategyMerged(chart=chart, month_chart=month_chart)
    portfolio = strategy.portfolio
    portfolio = portfolio.apply(lambda x: pd.Series(x).value_counts(normalize=True).to_dict())

    ref_date = portfolio.index[-1].date()
    weights_str = ", ".join([f"{key}={value * 100:.2f}%" for key, value in portfolio.iloc[-1].items()])
    output = f"{ref_date} final portfolio: {weights_str}"
    logger.info(output)
    print(output)

    logger.info("Making backtest reports...")
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    strategy.backtest(output_prefix)
    logger.info("Backtest completed successfully.")

    created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_lines = [
        f"Created Date: {created_date}",
        f"Reference Date: {ref_date}",
        "",
        "Final Portfolio (%):",
    ]
    summary_lines.extend([f"{k}: {v * 100:.2f}%" for k, v in portfolio.iloc[-1].items()])

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Summary saved to {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run asset allocation strategy without UI.")
    parser.add_argument("--output-prefix", default="output/final", help="Backtest output prefix.")
    parser.add_argument("--summary-path", default="output/final_summary.txt", help="Summary output path.")
    args = parser.parse_args()

    return run_cli(args.output_prefix, args.summary_path)


if __name__ == "__main__":
    raise SystemExit(main())
