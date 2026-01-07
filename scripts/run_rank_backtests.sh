#!/usr/bin/env bash
set -euo pipefail

# 반복 실행용: 여러 lookback/mode/top 조합으로 월간 랭크 백테스트 실행
# 기본 python 인터프리터를 사용하므로 필요시 .venv 활성화 후 실행

LOOKBACKS_CSV_DEFAULT="20,60,120,252"
MODES_CSV_DEFAULT="head,tail"
TOPS_CSV_DEFAULT="10,50"
REBAL_MONTHS_CSV_DEFAULT="1,3,6"

# Optional overrides (useful for CI/GitHub Actions)
# - LOOKBACKS_CSV: fallback when HEAD_LOOKBACKS_CSV/TAIL_LOOKBACKS_CSV not set
# - HEAD_LOOKBACKS_CSV / TAIL_LOOKBACKS_CSV: per-mode lookbacks
# - MODES_CSV / TOPS_CSV / REBAL_MONTHS_CSV
LOOKBACKS_CSV="${LOOKBACKS_CSV:-${LOOKBACKS_CSV_DEFAULT}}"
HEAD_LOOKBACKS_CSV="${HEAD_LOOKBACKS_CSV:-}"
TAIL_LOOKBACKS_CSV="${TAIL_LOOKBACKS_CSV:-}"
MODES_CSV="${MODES_CSV:-${MODES_CSV_DEFAULT}}"
TOPS_CSV="${TOPS_CSV:-${TOPS_CSV_DEFAULT}}"
REBAL_MONTHS_CSV="${REBAL_MONTHS_CSV:-${REBAL_MONTHS_CSV_DEFAULT}}"

IFS=',' read -r -a MODES <<< "${MODES_CSV}"
IFS=',' read -r -a TOPS <<< "${TOPS_CSV}"
IFS=',' read -r -a REBAL_MONTHS <<< "${REBAL_MONTHS_CSV}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

mkdir -p .output

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for mode in "${MODES[@]}"; do
  lbs_csv="${LOOKBACKS_CSV}"
  if [ "${mode}" = "head" ] && [ -n "${HEAD_LOOKBACKS_CSV}" ]; then
    lbs_csv="${HEAD_LOOKBACKS_CSV}"
  elif [ "${mode}" = "tail" ] && [ -n "${TAIL_LOOKBACKS_CSV}" ]; then
    lbs_csv="${TAIL_LOOKBACKS_CSV}"
  fi
  IFS=',' read -r -a LOOKBACKS <<< "${lbs_csv}"

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
  echo "Aggregating and analyzing results..."
  python scripts/analyze_rank_summary.py --force-aggregate --mode head --mode tail
else
  echo "Skipping aggregation/analysis (set SKIP_AGGREGATE=1)."
fi

echo "Done."
