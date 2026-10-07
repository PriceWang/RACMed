#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
racmed_timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
trap 'status=$?; echo "[$(racmed_timestamp)] RACMed real-device pipeline FAILED status=$status target=${TARGET:-unset}" >&2; exit "$status"' ERR

TARGET="${1:-}"
CONFIG="${2:-configs/tasks/segmentation2d/kvasir.yaml}"
RUN_DIR="${3:-}"
if [[ -z "$TARGET" || -z "$RUN_DIR" ]]; then
  echo "Usage: $0 {raspi5|jetson-cpu|jetson-cuda|apple-m4} CONFIG EXPERIMENT_DIR" >&2
  exit 2
fi
case "$TARGET" in
  raspi5|jetson-cpu|jetson-cuda|apple-m4) ;;
  *) echo "Unsupported target: $TARGET" >&2; exit 2 ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "racmed" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Activate the racmed environment before running real-device benchmarks." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate racmed
fi

CHECKPOINT="$RUN_DIR/compression/pruned.pt"
if [[ ! -f "$CHECKPOINT" ]]; then CHECKPOINT="$RUN_DIR/checkpoints/best.pt"; fi

COMMON=(--config "$CONFIG" --override "output_dir=$RUN_DIR")
BENCHMARK_FLAGS=(--energy)
if [[ "${RACMED_MEMORY_ONLY:-0}" == "1" ]]; then
  BENCHMARK_FLAGS=(--memory-only)
  echo "[$(racmed_timestamp)] Memory-only mode: one inference per backend; existing latency is preserved"
fi
echo "[$(racmed_timestamp)] RACMed real-device pipeline starting target=$TARGET"
echo "[$(racmed_timestamp)] Experiment directory: $RUN_DIR"
case "$TARGET" in
  apple-m4)
    if [[ ! -d "$RUN_DIR/deployment/model.mlpackage" ]]; then
      if [[ ! -f "$CHECKPOINT" ]]; then echo "CoreML export requires pruned.pt or best.pt under $RUN_DIR" >&2; exit 1; fi
      python -u scripts/05_export.py "${COMMON[@]}" --checkpoint "$CHECKPOINT" --coreml --coreml-only
    fi
    APPLE_MODE="${RACMED_APPLE_MODE:-all}"
    benchmark_apple_variant() {
      local artifact="$1" backend="$2" label="$3"
      if [[ -n "${RACMED_DEVICE_REPETITION:-}" ]]; then label="${label}_rep${RACMED_DEVICE_REPETITION}"; fi
      if [[ "$backend" == "coreml" ]]; then
        [[ -d "$artifact" ]] || { echo "Missing CoreML artifact: $artifact" >&2; exit 1; }
        python -u scripts/06_benchmark.py "${COMMON[@]}" \
          --target apple-m4 --backend coreml --artifact "$artifact" --output-label "$label" "${BENCHMARK_FLAGS[@]}"
      else
        [[ -f "$artifact" ]] || { echo "Missing ONNX artifact: $artifact" >&2; exit 1; }
        python -u scripts/06_benchmark.py "${COMMON[@]}" \
          --target apple-m4 --backend onnx --onnx-provider cpu \
          --artifact "$artifact" --output-label "$label" "${BENCHMARK_FLAGS[@]}"
      fi
    }
    case "$APPLE_MODE" in
      coreml) benchmark_apple_variant "$RUN_DIR/deployment/model.mlpackage" coreml coreml ;;
      cpu-fp32) benchmark_apple_variant "$RUN_DIR/deployment/model.onnx" onnx onnx_fp32_cpu ;;
      cpu-uniform-int8) benchmark_apple_variant "$RUN_DIR/deployment/model.uniform.int8.onnx" onnx onnx_uniform_int8_cpu ;;
      cpu-risk-aware-int8) benchmark_apple_variant "$RUN_DIR/deployment/model.risk_aware.int8.onnx" onnx onnx_risk_aware_int8_cpu ;;
      all)
        benchmark_apple_variant "$RUN_DIR/deployment/model.onnx" onnx onnx_fp32_cpu
        benchmark_apple_variant "$RUN_DIR/deployment/model.uniform.int8.onnx" onnx onnx_uniform_int8_cpu
        benchmark_apple_variant "$RUN_DIR/deployment/model.risk_aware.int8.onnx" onnx onnx_risk_aware_int8_cpu
        benchmark_apple_variant "$RUN_DIR/deployment/model.mlpackage" coreml coreml
        ;;
      *) echo "Unsupported RACMED_APPLE_MODE=$APPLE_MODE" >&2; exit 2 ;;
    esac
    ;;
  jetson-cuda)
    JETSON_MODE="${RACMED_JETSON_MODE:-all}"
    benchmark_jetson_variant() {
      local artifact="$1" provider="$2" label="$3"
      if [[ -n "${RACMED_DEVICE_REPETITION:-}" ]]; then
        label="${label}_rep${RACMED_DEVICE_REPETITION}"
      fi
      if [[ ! -f "$artifact" ]]; then
        echo "Missing $artifact. Export and calibrate deployment artifacts on the workstation first." >&2
        exit 1
      fi
      echo "[$(racmed_timestamp)] Jetson variant START label=$label provider=$provider artifact=$(basename "$artifact")"
      python -u scripts/06_benchmark.py "${COMMON[@]}" \
        --target jetson-cuda --backend onnx --onnx-provider "$provider" \
        --artifact "$artifact" --output-label "$label" "${BENCHMARK_FLAGS[@]}"
      echo "[$(racmed_timestamp)] Jetson variant DONE label=$label"
    }
    benchmark_jetson_native_variant() {
      local artifact="$1" label="$2"
      if [[ -n "${RACMED_DEVICE_REPETITION:-}" ]]; then
        label="${label}_rep${RACMED_DEVICE_REPETITION}"
      fi
      if [[ ! -f "$artifact" ]]; then
        echo "Missing $artifact. Export and calibrate deployment artifacts on the workstation first." >&2
        exit 1
      fi
      echo "[$(racmed_timestamp)] Jetson native TensorRT START label=$label artifact=$(basename "$artifact")"
      python -u scripts/06_benchmark.py "${COMMON[@]}" \
        --target jetson-cuda --backend tensorrt \
        --artifact "$artifact" --output-label "$label" "${BENCHMARK_FLAGS[@]}"
      echo "[$(racmed_timestamp)] Jetson native TensorRT DONE label=$label"
    }
    case "$JETSON_MODE" in
      cuda-fp32)
        benchmark_jetson_variant "$RUN_DIR/deployment/model.onnx" cuda onnx_fp32_cuda
        ;;
      tensorrt-fp16)
        benchmark_jetson_variant "$RUN_DIR/deployment/model.onnx" tensorrt onnx_fp16_tensorrt
        ;;
      tensorrt-uniform-int8)
        benchmark_jetson_native_variant "$RUN_DIR/deployment/model.uniform.qdq.int8.onnx" tensorrt_native_uniform_qdq_int8
        ;;
      tensorrt-risk-aware-int8|tensorrt-int8)
        benchmark_jetson_native_variant "$RUN_DIR/deployment/model.risk_aware.qdq.int8.onnx" tensorrt_native_risk_aware_qdq_int8
        ;;
      all)
        benchmark_jetson_variant "$RUN_DIR/deployment/model.onnx" cuda onnx_fp32_cuda
        benchmark_jetson_variant "$RUN_DIR/deployment/model.onnx" tensorrt onnx_fp16_tensorrt
        benchmark_jetson_native_variant "$RUN_DIR/deployment/model.uniform.qdq.int8.onnx" tensorrt_native_uniform_qdq_int8
        benchmark_jetson_native_variant "$RUN_DIR/deployment/model.risk_aware.qdq.int8.onnx" tensorrt_native_risk_aware_qdq_int8
        ;;
      *)
        echo "Unsupported RACMED_JETSON_MODE=$JETSON_MODE" >&2
        echo "Use all, cuda-fp32, tensorrt-fp16, tensorrt-uniform-int8, or tensorrt-risk-aware-int8." >&2
        exit 2
        ;;
    esac
    ;;
  jetson-cpu|raspi5)
    CPU_MODE="${RACMED_CPU_MODE:-all}"
    benchmark_cpu_variant() {
      local artifact="$1" label="$2"
      if [[ -n "${RACMED_DEVICE_REPETITION:-}" ]]; then label="${label}_rep${RACMED_DEVICE_REPETITION}"; fi
      [[ -f "$artifact" ]] || { echo "Missing canonical ONNX artifact: $artifact" >&2; exit 1; }
      python -u scripts/06_benchmark.py "${COMMON[@]}" \
        --target "$TARGET" --backend onnx --onnx-provider cpu \
        --artifact "$artifact" --output-label "$label" "${BENCHMARK_FLAGS[@]}"
    }
    case "$CPU_MODE" in
      fp32) benchmark_cpu_variant "$RUN_DIR/deployment/model.onnx" onnx_fp32_cpu ;;
      uniform-int8) benchmark_cpu_variant "$RUN_DIR/deployment/model.uniform.int8.onnx" onnx_uniform_int8_cpu ;;
      risk-aware-int8) benchmark_cpu_variant "$RUN_DIR/deployment/model.risk_aware.int8.onnx" onnx_risk_aware_int8_cpu ;;
      all)
        benchmark_cpu_variant "$RUN_DIR/deployment/model.onnx" onnx_fp32_cpu
        benchmark_cpu_variant "$RUN_DIR/deployment/model.uniform.int8.onnx" onnx_uniform_int8_cpu
        benchmark_cpu_variant "$RUN_DIR/deployment/model.risk_aware.int8.onnx" onnx_risk_aware_int8_cpu
        ;;
      *) echo "Unsupported RACMED_CPU_MODE=$CPU_MODE" >&2; exit 2 ;;
    esac
    ;;
esac
echo "[$(racmed_timestamp)] RACMed real-device pipeline completed target=$TARGET"
