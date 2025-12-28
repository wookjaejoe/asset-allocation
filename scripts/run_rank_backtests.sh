#!/usr/bin/env bash
set -euo pipefail

# 반복 실행용: 여러 lookback/mode/top 조합으로 월간 랭크 백테스트 실행
# 기본 python 인터프리터를 사용하므로 필요시 .venv 활성화 후 실행

LOOKBACKS=("20" "60" "120" "252")
MODES=("head" "tail")
TOPS=("10" "20" "50")

mkdir -p .output

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for mode in "${MODES[@]}"; do
  for lb in "${LOOKBACKS[@]}"; do
    for top in "${TOPS[@]}"; do
      label="rank_${mode}_lk${lb}_top${top}"
      out_dir=".output/${label}"
      mkdir -p "${out_dir}"
      echo "Running ${label}..."
      python scripts/monthly_rank_backtest.py \
        --lookback "${lb}" \
        --mode "${mode}" \
        --top "${top}" \
        --output "${out_dir}/rank_backtest.csv" \
        --monthly-output "${out_dir}/rank_monthly.csv" \
        --report "${out_dir}/rank_report.html"
    done
  done
done

echo "Done."
