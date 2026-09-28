# E017 online residual-scale SPS intervals

Run: **2026-08-18 (IST)**

E017 keeps the frozen E004 packed FNO and replaces the E006 global additive
floor `βσ_global` with a trajectory-resetting EMA of the previous-window RMS
residual. Ignored artifacts live under `artifacts/e017/`. Calibration JSON
SHA-256 `313ff2de5f46e102456068b7ec92869080681d904466b25f7939968cd12fdf09`.
Evaluation JSON SHA-256
`b11fc5fa8fd7416fe87f0b307d8028cfc256f486009afadbbe26d73647983906`.

## Hypothesis and decision

Selecting `(α, λ, k)` by maximum official SPS on train Re 3750/8850/16500/24150
will raise v1_val SPS by at least 1.0 over E007's 35.067, keep Rel-L2/TKE/MVPE
within 0.01 of E004, keep Time ≥86, and raise coverage above E006's 66.599%.

| Gate | Threshold | Observed | Result |
|---|---:|---:|---|
| Calibration SPS gain vs default | ≥ 2.000 | +21.065 | pass |
| Audit SPS gain vs default | ≥ 1.000 | +20.024 | pass |
| Rel-L2 / TKE / MVPE vs E004 | within 0.01 | 0.000 / 0.000 / 0.000 | pass |
| Time | ≥ 86 | 87.935 | pass |
| Validation SPS vs E007 | ≥ 36.067 | 34.205 | **fail** |
| Validation coverage vs E006 | > 66.599% | 69.698% | pass |

Hypothesis **rejected**. Selected interval: **α = 0.025, λ = 0.9, k = 0.3**.
SPS fell 0.862 below E007. Coverage rose, but official SPS penalizes the extra
width. Do not retune this grid on validation.

## Setup

- Base: frozen `artifacts/e004/fno_fp16.pth` SHA-256
  `75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`.
- Formula: `half = α|pred| + k σ_t` in physical space.
  `σ_t = λ σ_{t-1} + (1-λ) rms(ŷ_{t-1}-y_{t-1})` on measured `u,v`.
  `σ_0 = σ_global = 0.0563870259`. First window of each trajectory matches a
  static floor `k σ_global`. Pressure bounds are 0. Weights never update.
- Grid: 3 α × 4 λ × 6 k = 72 candidates. Default official band is `(0.05, 0)`.
- Calibration: 15 trajectories / 601 windows at train Re 3750, 8850, 16500,
  24150. Audit: remaining 41 train trajectories / 1,709 windows.
- Selection: maximum calibration SPS; ties break by smaller mean half-width,
  then smaller α, then smaller k, then larger λ. Validation was not used.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, RTX 5050, seed 0.
  Residual inference 59.38 s; validation mean step 13.72 ms.

Config SHA-256
`19dcc44538a19b370b1ab44c298ccc781c7f8d3c4324e5942ffd3fafec406d8b`.

## Training-side selection

| Split | Default SPS | Selected SPS | Coverage | Gain |
|---|---:|---:|---:|---:|
| Calibration | 19.569 | **40.634** | 75.721% | +21.065 |
| Audit | 18.864 | **38.888** | 74.270% | +20.024 |

E006's static `(0.025, 0.1)` had cal/audit SPS 44.234 / 41.345. The online
max-SPS pick is already worse than E006 on the same train residuals because
`k=0.3` triples the first-window additive floor
(`0.3 σ_global` vs `0.1 σ_global`).

## Validation results

Higher is better. Point errors match E004.

| Model | Rel-L2 | TKE | MVPE | Time | SPS | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| E007 `(0.025, 0.1)` | 94.720 | 70.497 | 95.143 | 88.018 | **35.067** | 66.599% |
| E012 `(0.025, 0.15)` | 94.720 | 70.497 | 95.143 | 88.096 | 35.938 | **75.463%** |
| **E017 `(0.025, 0.9, 0.3)`** | 94.720 | 70.497 | 95.143 | 87.935 | 34.205 | 69.698% |

E012 is a diagnostic, not a gate. E017 sits between E006 and E012 on coverage
and below both on SPS.

| Partition | Rel-L2 | TKE | MVPE | Time | SPS | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Re-only | 95.841 | 70.057 | 96.349 | 87.872 | 37.629 | 74.026% |
| AoA-only | 94.197 | 70.751 | 94.550 | 87.954 | 32.739 | 67.791% |
| Joint | 94.488 | 70.277 | 95.124 | 88.037 | 32.264 | 67.639% |

## Interpretation and decision

1. Revealed previous-window residual scale can be tracked cheaply and reset
   cleanly. That mechanism works. The train-only max-SPS rule then selected a
   wide `k` that official SPS punishes on `v1_val`.
2. Local val SPS 34.205 is below E007 35.067 and E012 35.938. The official SPS
   hole is not fixed by this isolated change.
3. Keep E006 `(0.025, 0.1)` on the leakage-safe submit path. Do not retune
   `(α, λ, k)` on validation or Codabench. Do not package E017.
4. E012 remains the frozen wider-band candidate if a later package is
   requested. I011 residual *point* feedback stays P2.
