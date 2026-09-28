# E003 no-adaptation baselines

Run: **2026-08-15 (IST)**

E003 establishes point, physics, uncertainty, and runtime anchors on the
[`real_regime_v1`](../configs/splits/real_regime_v1.json) validation stream. The
ignored full result is `artifacts/e003/no_adaptation_results.json`; this file
preserves the durable evidence and interpretation. The 21,957-byte local JSON
has SHA-256 `cbf1178fc09d5e512e8dbfc0b179333c69bfe9b58f9ac958f4d9fb8fd8b2e98c`.

## Setup

- Real partitions: `val_re` (317 windows), `val_aoa` (630), and `val_joint`
  (84), totaling 1,031 ordered batch-one windows from 25 whole trajectories.
- Window protocol: 20 input frames, 20 target frames, stride 20, native
  64x128 fields subsampled by 2 to 32x64, channels `[u,v,p]`, real pressure
  zero-filled.
- Normalization: official `mean_std_real.pt`; previous target delivered with
  exact one-window alignment but ignored by every no-adaptation predictor.
- Models: persistence plus released CNO, packed-fp16 FNO, and Transolver from
  both `sim_pretrain` and `sim_real_ft`; inference itself is float32.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, NumPy 1.26.4, RTX 5050 Laptop
  GPU (`sm_120`), seed 0, cuDNN benchmarking disabled.
- Timing: wall clock around reset and prediction with CUDA synchronized before
  and after; checkpoint preparation (SHA-256 plus loading) is reported
  separately. No model has trainable adaptation parameters and reset is a no-op.
- Bounds: none. The official scorer therefore applies its default +/-5% band.
- Metrics: bundled v6 `scoring.py`, accumulated in 16-window metric batches.
  Unit tests reproduce its batch results, while model inference remains strict
  batch size 1. The unpublished `final_score` is not estimated.

The complete seven-model benchmark covered 7,217 model-window combinations.
All outputs were finite and exact-shape. It took 1,102.64 seconds of local
harness wall time plus 5.51 seconds of checkpoint hashing/loading; HDF5 reads
and local NumPy scoring dominate those wall totals and are not participant
inference time.

## Overall results

Higher is better for every score below. Time scores are local-hardware results;
the other four are prediction-derived.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Persistence | 92.164 | 72.697 | 94.427 | 99.510 | 13.772 |
| Sim-pretrain CNO | 77.748 | 66.383 | 73.239 | 77.045 | 0.895 |
| Sim-pretrain FNO fp16 | 77.876 | 66.969 | 71.700 | 87.633 | 0.790 |
| Sim-pretrain Transolver | 82.457 | 68.818 | 77.996 | 67.757 | 3.455 |
| Sim-real-ft CNO | 95.532 | 75.454 | 95.958 | 77.116 | 18.546 |
| **Sim-real-ft FNO fp16** | **96.348** | **78.629** | **96.956** | **87.833** | **20.614** |
| Sim-real-ft Transolver | 93.452 | 70.644 | 94.262 | 67.835 | 13.350 |

Underlying aggregate quantities:

| Model | Rel-L2 error | TKE error | MVPE error | SPS value | Coverage |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.170052 | 0.751133 | 0.118047 | 0.137717 | 29.107% |
| Sim-pretrain CNO | 0.572405 | 1.012797 | 0.730781 | 0.008946 | 2.570% |
| Sim-pretrain FNO fp16 | 0.568197 | 0.986452 | 0.789400 | 0.007896 | 2.279% |
| Sim-pretrain Transolver | 0.425496 | 0.906216 | 0.564247 | 0.034546 | 8.988% |
| Sim-real-ft CNO | 0.093530 | 0.650606 | 0.084242 | 0.185460 | 35.002% |
| **Sim-real-ft FNO fp16** | **0.075814** | **0.543582** | **0.062800** | **0.206140** | **36.904%** |
| Sim-real-ft Transolver | 0.140127 | 0.831092 | 0.121738 | 0.133497 | 27.947% |

## Runtime and footprint

`Timed total` is the sum of synchronized reset/prediction intervals over all
1,031 steps. Peak memory is allocated CUDA tensor memory, not total process or
driver memory.

| Model | Mean step | Timed total | Load + hash | Checkpoint | Peak GPU |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.018 ms | 0.018 s | <0.001 s | none | 0 MiB |
| Sim-pretrain CNO | 64.712 ms | 66.718 s | 1.798 s | 30.7 MiB | 152.9 MiB |
| Sim-pretrain FNO fp16 | 14.517 ms | 14.967 s | 1.399 s | 192.1 MiB | 532.1 MiB |
| Sim-pretrain Transolver | 165.071 ms | 170.188 s | 0.216 s | 48.1 MiB | 496.9 MiB |
| Sim-real-ft CNO | 64.195 ms | 66.185 s | 0.161 s | 30.7 MiB | 161.1 MiB |
| **Sim-real-ft FNO fp16** | **13.987 ms** | **14.421 s** | **1.758 s** | **192.1 MiB** | **532.1 MiB** |
| Sim-real-ft Transolver | 163.891 ms | 168.971 s | 0.180 s | 48.1 MiB | 496.9 MiB |

The selected FNO checkpoint is 201,396,349 bytes, leaving roughly 54 MB under
the 256 MB extracted-submission limit for code and other state. Its SHA-256 is
`9dc4e54866acae1f4a44cbb7a0c146baab1ff7cce3e5160e12ad2c6369edda2a`.

## Regime breakdown

The selected FNO and the leakage-free persistence anchor are shown below.

| Model / partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Persistence / Re | 93.988 | 72.753 | 95.853 | 99.543 | 17.758 |
| Persistence / AoA | 91.278 | 72.560 | 93.699 | 99.554 | 11.871 |
| Persistence / joint | 92.117 | 73.527 | 94.622 | 99.182 | 12.966 |
| FNO fp16 / Re | 97.064 | 76.929 | 97.582 | 87.774 | 23.554 |
| FNO fp16 / AoA | 95.988 | 79.328 | 96.643 | 87.849 | 19.169 |
| FNO fp16 / joint | 96.372 | 80.020 | 96.959 | 87.940 | 20.344 |

AoA-only validation is the hardest partition for Rel-L2, MVPE, and SPS for both
anchors. The selected FNO's weakest physics component is TKE on Re-only cases.

## Interpretation and decision

1. Every simulation-only neural checkpoint is dominated by persistence on all
   five local components. The sim-to-real domain gap is too large to treat
   sim-pretraining alone as a useful point-prediction baseline.
2. Real finetuning improves Rel-L2 by 10.995-18.472 points and MVPE by
   16.267-25.256 points, depending on architecture. This supports E003's first
   hypothesis.
3. Packed real-finetuned FNO beats the other neural models on every component,
   is about 4.6x faster than CNO and 11.7x faster than Transolver, and remains
   within the submission size limit. It is the base for subsequent training and
   adaptation experiments.
4. Default SPS intervals remain poorly calibrated: even the best model covers
   only 36.9% of scored elements. Explicit residual-based calibration remains a
   high-value experiment.

Prediction-derived FNO metrics were bitwise identical in a repeated
three-window CUDA replay. A real-finetuned CNO stream also passed under the
evaluator-compatible Python 3.10/Torch 2.2.2 CPU environment.

## Leakage caveat

- Persistence is genuinely leakage-free.
- `sim_pretrain` checkpoints did not see validation measurements, but did see
  simulations at the selected parameter values; they are not strict-regime
  models under E002's 72-case simulation policy.
- Released `sim_real_ft` checkpoints saw the complete real release, including
  these 25 validation trajectories. Their scores are diagnostic reproduction
  anchors, not leakage-safe validation estimates.

The next learned baseline was completed as E004: see
[`E004_RESULTS.md`](E004_RESULTS.md). A later strict-regime result must
additionally retrain simulation pretraining from the 72-case strict partition.
