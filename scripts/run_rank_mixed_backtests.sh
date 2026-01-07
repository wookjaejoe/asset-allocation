#!/usr/bin/env bash
set -euo pipefail

# 단기 리버설 + 중기 모멘텀 혼합 전략 그리드 실행
# 기본 python 인터프리터를 사용하므로 필요시 .venv 활성화 후 실행

SHORT_LB=("5" "10" "20")
MID_LB=("60" "120" "252")
WEIGHTS=("0.8:0.2" "0.7:0.3" "0.6:0.4") # w_mom:w_rev
TOP=("10" "50")
REBAL_MONTHS=("1" "3" "6")
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
START_DATE="${START_DATE:-2005-01-01}"
END_DATE="${END_DATE:-}"

mkdir -p .output

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for s in "${SHORT_LB[@]}"; do
  for m in "${MID_LB[@]}"; do
    for wp in "${WEIGHTS[@]}"; do
      w_mom="${wp%%:*}"
      w_rev="${wp##*:}"
      for top in "${TOP[@]}"; do
        for rbm in "${REBAL_MONTHS[@]}"; do
          label="rankmix_s${s}_m${m}_w${w_mom//./}_top${top}_rbm${rbm}"
          out_dir=".output/${label}"
          mkdir -p "${out_dir}"
          echo "Running ${label}..."
          python scripts/rank_mixed_backtest.py \
            --short-lookback "${s}" \
            --mid-lookback "${m}" \
            --weight-mom "${w_mom}" \
            --weight-rev "${w_rev}" \
            --top "${top}" \
            --rebalance-months "${rbm}" \
            --start "${START_DATE}" \
            ${END_DATE:+--end "${END_DATE}"} \
            --output "${out_dir}/rank_backtest.csv" \
            --monthly-output "${out_dir}/rank_monthly.csv" \
            --report "${out_dir}/rank_report.html"
        done
      done
    done
  done
done

if [ "${SKIP_AGGREGATE}" != "1" ]; then
  echo "Aggregating results..."
  # NOTE: analyze_rank_summary.py currently doesn't fully support mix mode analysis.
  # Mix mode has additional columns (lookback_short, lookback_mid, weight_mom, weight_rev)
  # that require separate analysis logic. For now, only aggregate to CSV.
  python scripts/aggregate_rank_results.py
  echo "[info] Mix mode analysis not yet implemented. See rank_summary.csv for raw data."
else
  echo "Skipping aggregation (set SKIP_AGGREGATE=1)."
fi

echo "Done."
