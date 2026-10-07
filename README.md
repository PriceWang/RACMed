# RACMed

RACMed (**Risk- and Contradiction-Aware Compression for Reliable Medical Image Analysis**) is a PyTorch/MONAI codebase for reliability-preserving compression of medical image models.

The pipeline combines:

1. region/sample risk maps from boundary proximity, small components, teacher entropy, teacher–compressed-model disagreement, and class rarity;
2. risk-gated distillation that protects diagnostic evidence while discounting uncertain teacher predictions;
3. layer-risk elastic pruning and risk-aware mixed precision based on feature drift, output degradation, gradients, quantization error, and high-risk activation correlation;
4. cross-task contradiction analysis using average performance, worst-task drop, calibration, false negatives, and measured deployment cost.

The retained experiment methods are `standalone`, `vanilla_kd`, `attention_kd`, `relation_kd`, and `racmed`. `compressed_model` is the deployable architecture in YAML configs.

## Environment

Conda installs only Python and pip. Every package is installed by pip through `requirements.txt`:

```bash
conda env create -f environment.yml
conda activate racmed
```

For a CUDA workstation, install the PyTorch wheel matched to the local driver before the remaining requirements:

```bash
conda create -n racmed python=3.11 pip -y
conda activate racmed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --upgrade onnxruntime-gpu
python -c 'import torch; import onnxruntime as ort; print(ort.get_available_providers())'
```

For CPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` installs the portable CPU `onnxruntime` package. The CUDA-workstation command replaces its runtime module with the GPU build after the portable requirements are installed; importing PyTorch first makes its bundled CUDA libraries available. Apple Silicon additionally receives `coremltools` through an environment marker. Jetson CUDA uses the PyTorch and ONNX Runtime/TensorRT builds supplied for its JetPack release. Optional structural pruning support is installed with:

```bash
pip install -r requirements-optional.txt
```

## Data and external teachers

Machine-local paths are supplied at runtime and are never stored in configs:

```bash
export RACMED_DATA_ROOT=/path/to/datasets
export RACMED_MODEL_ROOT=/path/to/external-models
```

Validated experiment families:

| Config | Dataset | External teacher |
|---|---|---|
| `tasks/classification/rsna.yaml` | RSNA Pneumonia Detection | TorchXRayVision DenseNet121 `densenet121-res224-all` |
| `tasks/segmentation2d/kvasir.yaml` | Kvasir-SEG | official MedSAM ViT-B |
| `tasks/segmentation2d/acdc.yaml` | ACDC ED/ES slices | official MedSAM ViT-B |
| `tasks/segmentation3d/brats2018.yaml` | BraTS 2018 | MONAI BraTS SegResNet bundle |

Expected model-store entries are:

```text
${RACMED_MODEL_ROOT}/torchxrayvision/
${RACMED_MODEL_ROOT}/medsam-vit-base/medsam_vit_b.pth
${RACMED_MODEL_ROOT}/brats_mri_segmentation/
```

The experiment adapters read the native RSNA, ACDC, and BraTS layouts. Kvasir-SEG uses the common 2D contract:

```text
root/{train,val,test}/images/*
root/{train,val,test}/masks/*
```

## One-command experiment pipeline

After setting the two path variables:

```bash
./full_pipeline.sh
```

The script validates all external teachers, runs each config’s three-seed method matrix, applies the compression policies, evaluates three-seed QDQ artifacts, profiles FLOPs, computes threshold-independent lesion FROC curves for Kvasir/BraTS, benchmarks local hardware, and writes cross-task analysis CSVs. Existing completed stages are reused by default. Set `RACMED_RESUME=0` for a clean rerun.

Progress is streamed without buffering. Every task, method, seed, pruning, export, benchmark, skip, failure, and completion event includes elapsed time and is also appended to:

```text
outputs/experiments/pipeline.log
```

Long-running inner loops use live `tqdm` bars: RSNA DICOM cache construction, training batches, validation/test inference, calibration, layer-risk profiling, ONNX reliability evaluation, and hardware benchmark repetitions. Structured progress remains in log files, so redirected/non-interactive runs are still observable. Set `RACMED_DISABLE_TQDM=1` only when a scheduler already renders its own progress UI.

CUDA is used automatically whenever `device: auto` and a compatible GPU are available. The RSNA first-run cache uses parallel CPU workers only for DICOM decoding, then performs XRV-compatible normalization and resize in GPU batches; later runs memory-map the FP16 cache. DataLoaders enable pinned memory, persistent workers and configurable prefetching, while training/evaluation use non-blocking transfers and AMP inference. Relevant overrides are:

```bash
python scripts/01_validate_teacher.py \
  --config configs/tasks/classification/rsna.yaml \
  --override dataset.preload_device=cuda \
  --override dataset.preload_workers=8 \
  --override dataset.preload_batch_size=64 \
  --override training.prefetch_factor=4
```

DICOM/NIfTI decoding and filesystem writes remain CPU/I/O operations because CUDA does not provide medical-file decoders; their output is parallelized and prefetched so GPU compute is not replaced with a numerically different preprocessing protocol.

Typical live output is:

```text
... | PIPELINE | START | task | task=kvasir_segmentation2d position=2/4
... | PIPELINE | START | train | method=racmed seed=0 run=5/15
... | INFO | train_progress epoch=1/50 batch=10/100 loss=0.8421
... | PIPELINE | LAYER_RISK_PROGRESS | layer=12/48 name=encoders.2.3
... | PIPELINE | ONNX_EVAL_PROGRESS | backend=onnx_risk_aware_int8 batch=10/25
... | PIPELINE | DONE  | quantize_export | variant=racmed_elastic_pruned seed=0 | elapsed=42.7s
```

When numbered scripts are run individually, stage events are written to `EXPERIMENT_DIR/logs/pipeline.log`; training batch and epoch logs remain in `EXPERIMENT_DIR/logs/train.log`.

The equivalent explicit command is:

```bash
python scripts/08_run_pipeline.py \
  --config configs/tasks/classification/rsna.yaml \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --config configs/tasks/segmentation2d/acdc.yaml \
  --config configs/tasks/segmentation3d/brats2018.yaml \
  --results-dir outputs/experiments \
  --resume
```

## Numbered pipeline commands

All entry points accept dotted overrides such as `--override training.batch_size=4`.

### 1. Validate an external teacher

```bash
python scripts/01_validate_teacher.py \
  --config configs/tasks/segmentation2d/kvasir.yaml
```

Produces teacher `config.yaml`, `metrics.json`, `summary.csv`, per-case metrics, provenance, and `reproduction_gate.json` under the configured teacher output directory.

### 2. Train compressed models

```bash
python scripts/02_train.py \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --method standalone \
  --override output_dir=outputs/experiments/kvasir/training/standalone/seed_0

python scripts/02_train.py \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --method vanilla_kd \
  --override output_dir=outputs/experiments/kvasir/training/vanilla_kd/seed_0

python scripts/02_train.py \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --method racmed \
  --override output_dir=outputs/experiments/kvasir/training/racmed/seed_0
```

Each run produces best/last checkpoints, train logs, TensorBoard events, validation history, test metrics, per-case metrics, risk visualizations, and a canonical summary.

### 3. Evaluate a checkpoint

```bash
python scripts/03_evaluate.py \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --checkpoint outputs/experiments/kvasir/training/racmed/seed_0/checkpoints/best.pt \
  --override output_dir=outputs/experiments/kvasir/training/racmed/seed_0
```

Produces final-test `metrics.json`, `per_case_metrics.csv`, and `summary.csv` without changing training data.

Add `--lesion-operating-curve` for segmentation checkpoints to write an IoU-matched lesion FROC curve, normalized FROC AUC over 0–4 false positives per case, and lesion/small-lesion recall at fixed FP budgets. This is threshold-independent and never selects a training operating point from the test split.

### 4. Profile layer risk and prune

```bash
python scripts/04_compress.py \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --checkpoint outputs/experiments/kvasir/training/racmed/seed_0/checkpoints/best.pt \
  --override output_dir=outputs/experiments/kvasir/compression/manual_racmed/seed_0
```

Produces `compression/layer_risk_scores.csv`, pruning and mixed-precision policies, `pruned.pt`, and `compression_metrics.json`.

Use `--backfill-flops` without a checkpoint to populate dense fvcore FLOPs for all persisted compression variants below the configured `output_dir`. Mask-based pruning correctly reports unchanged dense FLOPs; its effective parameter sparsity remains a separate field.

### 5. Quantize and export

```bash
python scripts/05_export.py \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --checkpoint outputs/experiments/kvasir/compression/manual_racmed/seed_0/compression/pruned.pt \
  --tensorrt-qdq \
  --evaluate-onnx \
  --override output_dir=outputs/experiments/kvasir/compression/manual_racmed/seed_0
```

Produces FP32 ONNX, uniform INT8 ONNX, selective risk-aware INT8 ONNX, quantization metadata, reliability metrics, and an artifact manifest. Add `--coreml` on Apple Silicon or `--tensorrt` on a host with `trtexec`.

### 6. Benchmark deployment

```bash
python scripts/06_benchmark.py \
  --config configs/tasks/segmentation2d/kvasir.yaml \
  --artifact outputs/experiments/kvasir/compression/manual_racmed/seed_0/deployment/model.risk_aware.int8.onnx \
  --backend onnx \
  --energy \
  --override output_dir=outputs/experiments/kvasir/compression/manual_racmed/seed_0
```

Produces latency mean/std/p50/p95, throughput, peak memory, backend/provider identity, hardware metadata, and best-effort energy telemetry. On Jetson, `--energy` runs an additional sustained inference workload while sampling board-input power from `tegrastats`; `deployment.energy_duration_seconds` controls the sampling window (default: 2 seconds).

Use `--energy-only` together with `--energy` and the same output label to add power telemetry to an existing formal latency record without repeating its full timing protocol. This is useful for slow 3D CPU artifacts; it never overwrites the latency JSON.

Use `--memory-only` with the same output label to run one inference and attach absolute process high-water RSS plus incremental RSS to an existing timing record. CUDA/TensorRT retains accelerator-allocated peak memory as the primary `peak_memory_mb` and also records process RSS.

### 7. Analyze cross-task contradictions

```bash
python scripts/07_analyze.py \
  --results-dir outputs/experiments
```

Produces contradiction matrix, worst-task drop, coverage, risk-adjusted score, bootstrap confidence intervals, threshold/weight sensitivity, paired statistical tests, paired per-case effects, lesion-FROC statistics, measured device latency/RSS summaries, and FLOP/parameter summaries under `outputs/experiments/analysis/`.

Analysis also writes `result_completeness.csv`, `result_artifact_manifest.csv`, and `result_manifest.json`. The manifest records protocol coverage, SHA-256 identities, package versions, pytest status, and a private-path audit.

## Real-device benchmark

Copy an experiment directory containing `checkpoints/` or `compression/` to the target and run:

```bash
./real_device.sh raspi5 configs/tasks/segmentation2d/kvasir.yaml EXPERIMENT_DIR
./real_device.sh jetson-cpu configs/tasks/segmentation2d/kvasir.yaml EXPERIMENT_DIR
./real_device.sh jetson-cuda configs/tasks/segmentation2d/kvasir.yaml EXPERIMENT_DIR
./real_device.sh apple-m4 configs/tasks/segmentation2d/kvasir.yaml EXPERIMENT_DIR
```

Raspberry Pi 5B and Jetson CPU use selective dynamic INT8 ONNX Runtime. A Jetson CUDA run executes four labeled variants by default: ONNX Runtime CUDA FP32, ONNX Runtime TensorRT FP16, native TensorRT uniform Q/DQ INT8, and native TensorRT risk-aware Q/DQ INT8. Select one with `RACMED_JETSON_MODE=cuda-fp32`, `tensorrt-fp16`, `tensorrt-uniform-int8`, or `tensorrt-risk-aware-int8`. Native TensorRT engines (`*.engine`), source-hash metadata, and `tensorrt_native.timing.cache` are stored beside the ONNX artifacts. The first engine build can take several minutes; large 3D networks can take substantially longer. Later runs reuse an engine only when its source hash, input shape, and precision all match.

Set `RACMED_DEVICE_REPETITION=1`, `2`, or `3` for repeated measurements; the value is appended to every output label so records are not overwritten.

Set `RACMED_MEMORY_ONLY=1` to supplement existing labeled latency records with one-inference peak-memory probes without repeating the formal warm-up and timing repetitions.

By default, Raspberry Pi and Jetson CPU runs benchmark canonical FP32, uniform INT8, and risk-aware INT8 ONNX artifacts. Select one with `RACMED_CPU_MODE=fp32`, `uniform-int8`, or `risk-aware-int8`. Apple runs those three CPU artifacts plus CoreML by default; select one with `RACMED_APPLE_MODE=coreml`, `cpu-fp32`, `cpu-uniform-int8`, or `cpu-risk-aware-int8`. If CoreML is absent, the device script uses `--coreml-only`, which cannot overwrite the canonical ONNX bundle.

Calibrated TensorRT Q/DQ artifacts must be exported on the data workstation with pipeline step 5 plus `--tensorrt-qdq`, then copied to the device. TensorRT INT8 is built and executed through JetPack's native `trtexec` and Python bindings; it is not routed through ONNX Runtime's TensorRT execution provider. This prevents silent CPU partition fallback and validates that the resulting engine contains INT8 layers. FP32 and FP16 still use ONNX Runtime, so verify that its JetPack-compatible build exposes both accelerated providers:

```bash
python -c 'import torch; import onnxruntime as ort; print(ort.get_available_providers())'
```

On Jetson, install TensorRT's system development and Python packages (`tensorrt`, `tensorrt-dev`, and `python3-libnvinfer`) from JetPack, then install the small artifact-runtime dependency set with `pip install -r requirements-jetson.txt`. The requirements file intentionally leaves the JetPack-matched PyTorch, TensorRT, and ONNX Runtime packages untouched. Power sampling starts continuous `tegrastats`, reads `VDD_IN` (or `POM_5V_IN` on older boards), and terminates the sampler explicitly, so it also supports JetPack builds without a `tegrastats --count` option.

Operator-quantized `ConvInteger` models are rejected on Jetson GPU because they are CPU deployment artifacts. Apple M4 Pro exports and benchmarks a CoreML ML Program. Labeled device records are written as `deployment/latency_<variant>_<device>.json` and `deployment/energy_<variant>_<device>.json`.

## Repository layout

```text
configs/tasks/    validated configs grouped by task family
scripts/          zero-padded pipeline entry points only
src/risk/         risk-map components
src/losses/       supervised, KD, RACMed, calibration, and sparsity losses
src/compression/  layer-risk profiling, pruning, quantization, export
src/deployment/   platform-aware runtime and telemetry probes
src/datasets/     validated dataset adapters
src/models/       external teacher wrappers and compressed architectures
src/engine/       training, evaluation, checkpointing, reproducibility
src/contradiction cross-task result analysis
```

## Output contract

```text
outputs/
  cache/teacher/                         reusable teacher logits and data caches
  experiments/
    <task>/
      teacher/{native,matched_test}/     teacher validation and provenance
      training/<method>/seed_<n>/        checkpoints, logs, metrics, visualizations
      ablations/<variant>/seed_<n>/      mechanism ablations
      evaluation/operating_points/...    auxiliary frozen-point evaluations
      compression/<variant>/seed_<n>/
        compression/                     risk scores, policies, pruned checkpoint
        deployment/                      ONNX/CoreML/TensorRT and device metrics
    analysis/                            cross-task statistics and result manifest
```

Every run directory contains its resolved `config.yaml`, canonical `summary.csv`,
`metrics.json`, and optional `per_case_metrics.csv`. This hierarchy is shared by
the full pipeline and the numbered commands, so individual stages remain
resumable and directly comparable.
