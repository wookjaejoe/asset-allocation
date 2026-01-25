#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Set INSTALL_DEPS=0 to skip dependency install.
if [[ "${INSTALL_DEPS:-1}" == "1" ]]; then
  python -m pip install -r requirements-actions.txt
fi

bash scripts/run_aa_backtests.sh
python scripts/daily_asset_allocation_report.py --start "2007-01-01"

bash scripts/run_rank_backtests.sh
python scripts/daily_rank_report.py --top 20 --head-lookbacks "120,252" --tail-lookbacks "20,40"
python scripts/analyze_aa_summary.py

TODAY_KST="$(TZ=Asia/Seoul date +%Y%m%d)"
python scripts/build_daily_combined_email.py --date "${TODAY_KST}"

echo "Done. Outputs are in .output/."
