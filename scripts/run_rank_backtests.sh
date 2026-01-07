#!/usr/bin/env bash
set -euo pipefail

# 반복 실행용: 여러 lookback/mode/top 조합으로 월간 랭크 백테스트 실행
# 기본 python 인터프리터를 사용하므로 필요시 .venv 활성화 후 실행

LOOKBACKS=("20" "60" "120" "252")
MODES=("head" "tail")
TOPS=("10" "50")
REBAL_MONTHS=("1" "3" "6")
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

mkdir -p .output

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for mode in "${MODES[@]}"; do
  for lb in "${LOOKBACKS[@]}"; do
    for top in "${TOPS[@]}"; do
      for rbm in "${REBAL_MONTHS[@]}"; do
        label="rank_${mode}_lk${lb}_top${top}_rbm${rbm}"
        out_dir=".output/${label}"
      mkdir -p "${out_dir}"
      echo "Running ${label}..."
      python scripts/rank_backtest.py \
        --lookback "${lb}" \
        --mode "${mode}" \
        --top "${top}" \
        --rebalance-months "${rbm}" \
          --output "${out_dir}/rank_backtest.csv" \
          --monthly-output "${out_dir}/rank_monthly.csv" \
          --report "${out_dir}/rank_report.html"
      done
    done
  done
done

if [ "${SKIP_AGGREGATE}" != "1" ]; then
  echo "Aggregating results..."
  python scripts/aggregate_rank_results.py
else
  echo "Skipping aggregation (set SKIP_AGGREGATE=1)."
fi

echo "Done."
