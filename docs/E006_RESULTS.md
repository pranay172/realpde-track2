# E006 train-only SPS calibration

Run: **2026-08-15 (IST)**

E006 keeps the frozen E004 packed FNO and replaces the official default ±5%
band with a global interval `half = α|pred| + βσ`. Selection used only
training residuals. Ignored artifacts live under `artifacts/e006/`. The
evaluation JSON has SHA-256
`327aa40cd537231c203fec700b2fc77a5c79fb3624209d83622f5003b06724d3`.

## Hypothesis and decision

Selecting `(α, β)` by maximum official SPS on train Re 3750/8850/16500/24150
will raise validation SPS by at least 5.0 over E004's 15.024, keep Rel-L2/TKE/MVPE
within 0.01 of E004, and regress Time by at most 1.0. Training-side gates
required +2.0 calibration and +1.0 audit SPS versus the default `(0.05, 0)`.

| Gate | Threshold | Observed | Result |
|---|---:|---:|---|
| Calibration SPS gain | ≥ 2.000 | +24.665 | pass |
| Audit SPS gain | ≥ 1.000 | +22.480 | pass |
| Validation SPS | ≥ 20.024 | 35.067 | pass |
| Rel-L2 / TKE / MVPE vs E004 | within 0.01 | 0.000 / 0.000 / 0.000 | pass |
| Time | ≥ 87.167 | 88.130 | pass |

Hypothesis accepted. Selected interval: **α = 0.025, β = 0.1**.

## Setup

- Base: frozen `artifacts/e004/fno_fp16.pth` SHA-256
  `75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`.
- Formula: `half_width = α|pred| + βσ` in physical space after official
  `mean_std_real` denormalization. `σ = 0.0563870259`. Pressure bounds are 0.
- Grid: 5 α × 9 β = 45 candidates. Default official band is `(0.05, 0)`.
- Calibration: 15 trajectories / 601 windows at train Re 3750, 8850, 16500,
  24150. Audit: remaining 41 train trajectories / 1,709 windows.
- Selection: maximum calibration SPS; ties break by smaller mean half-width,
  then smaller α, then smaller β. Validation was not used for selection.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, RTX 5050, seed 0.
  Residual inference 62.69 s; validation wall 69.62 s; mean step 13.22 ms.

Config SHA-256:
`7569126db57c459a4c844f581fa71a4c71731d765f2075b59389934e4c8a0a79`.

## Training-side selection

| Split | Default SPS | Selected SPS | Coverage | Gain |
|---|---:|---:|---:|---:|
| Calibration | 19.569 | **44.234** | 80.431% | +24.665 |
| Audit | 18.864 | **41.345** | 75.953% | +22.480 |

## Validation results

Higher is better. Point errors are bitwise identical to E004.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | **72.697** | 94.427 | **99.510** | 13.772 |
| E004 default ±5% | 94.720 | 70.497 | 95.143 | 88.167 | 15.024 |
| **E006 α=0.025, β=0.1** | 94.720 | 70.497 | 95.143 | 88.130 | **35.067** |

E006 coverage 66.599% versus E004 29.982%. Time regressed 0.037.

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 95.841 | 70.057 | 96.349 | 88.137 | 39.096 |
| AoA-only | 94.197 | 70.751 | 94.550 | 88.139 | 33.405 |
| Joint | 94.488 | 70.277 | 95.124 | 88.039 | 32.300 |

## Interpretation and decision

1. A cheap additive floor plus a slightly tighter proportional term more than
   doubles SPS without changing the point predictor.
2. Training-side residuals were optimistic (cal 44.2, audit 41.3, val 35.1),
   but the validation gain remains +20.043.
3. Adopt `(0.025, 0.1)` as the frozen E004 uncertainty recipe. Do not retune
   this grid on the validation stream.
4. TKE is still 2.200 below persistence. The next isolated change should not
   reopen this interval.
