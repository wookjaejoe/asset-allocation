#!/usr/bin/env bash
set -euo pipefail

# 반복 실행용: 여러 lookback/mode 조합으로 월간 랭크 백테스트 실행
# 기본 python 인터프리터를 사용하므로 필요시 .venv 활성화 후 실행

LOOKBACKS=("7" "30" "90")
MODES=("head" "tail")

mkdir -p .output

for lb in "${LOOKBACKS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "Running lookback=${lb}, mode=${mode}..."
    python scripts/monthly_rank_backtest.py \
      --lookback "${lb}" \
      --mode "${mode}" \
      --output ".output/rank_${lb}_${mode}.csv" \
      --monthly-output ".output/rank_${lb}_${mode}_monthly.csv" \
      --report ".output/rank_${lb}_${mode}_report.html"
  done
done

echo "Done."
