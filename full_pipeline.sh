#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "racmed" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Activate the racmed environment before running this pipeline." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate racmed
fi

: "${RACMED_DATA_ROOT:?Set RACMED_DATA_ROOT to the dataset store.}"
: "${RACMED_MODEL_ROOT:?Set RACMED_MODEL_ROOT to the external teacher store.}"

RESUME_ARGS=()
if [[ "${RACMED_RESUME:-1}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

racmed_timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
RESULTS_ROOT="outputs/experiments"
CONFIGS=(
  configs/tasks/classification/rsna.yaml
  configs/tasks/segmentation2d/kvasir.yaml
  configs/tasks/segmentation2d/acdc.yaml
  configs/tasks/segmentation3d/brats2018.yaml
)
echo "[$(racmed_timestamp)] RACMed pipeline starting (resume=${RACMED_RESUME:-1})"
echo "[$(racmed_timestamp)] Progress log: $RESULTS_ROOT/pipeline.log"

python -u scripts/08_run_pipeline.py \
  --config "${CONFIGS[0]}" \
  --config "${CONFIGS[1]}" \
  --config "${CONFIGS[2]}" \
  --config "${CONFIGS[3]}" \
  --results-dir "$RESULTS_ROOT" \
  "${RESUME_ARGS[@]}"

echo "[$(racmed_timestamp)] Profiling persisted compression FLOPs"
for config in "${CONFIGS[@]}"; do
  python -u scripts/04_compress.py --config "$config" --backfill-flops
done

run_lesion_curve() {
  local config="$1" run_dir="$2" method="$3" seed="$4"
  local result="$run_dir/lesion_operating_curve_summary.csv"
  if [[ "${RACMED_RESUME:-1}" == "1" && -f "$result" ]]; then
    echo "[$(racmed_timestamp)] SKIP lesion-FROC method=$method seed=$seed result=$result"
    return
  fi
  echo "[$(racmed_timestamp)] START lesion-FROC method=$method seed=$seed"
  python -u scripts/03_evaluate.py \
    --config "$config" \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --method-label "$method" \
    --lesion-operating-curve \
    --override "seed=$seed" \
    --override "output_dir=$run_dir"
}

echo "[$(racmed_timestamp)] Evaluating threshold-independent lesion reliability"
for seed in 0 1 2; do
  for method in standalone vanilla_kd racmed; do
    run_lesion_curve \
      "${CONFIGS[1]}" \
      "$RESULTS_ROOT/kvasir/training/$method/seed_$seed" "$method" "$seed"
    run_lesion_curve \
      "${CONFIGS[3]}" \
      "$RESULTS_ROOT/brats2018/training/$method/seed_$seed" "$method" "$seed"
  done
done

python -u scripts/07_analyze.py --results-dir "$RESULTS_ROOT"

echo "[$(racmed_timestamp)] RACMed pipeline completed"
