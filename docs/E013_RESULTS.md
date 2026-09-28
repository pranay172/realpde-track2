# E013 official score-aligned loss on the E004 recipe

Run: **2026-08-16 (IST)**

E013 repeats the E004 fine-tune with one isolated change: joint physical
`u/v` relative MSE is replaced by
`0.5 * official Rel-L2 + 0.3 * official TKE-rel-L2 + 0.2 * official MVPE-L2`,
the published SPS branch weights. Every other control is held fixed.
Evaluation uses the frozen E006 interval. Ignored artifacts live under
`artifacts/e013/`. The packed checkpoint SHA-256 is
`a2001324e4b9ee1cd49b90feb0e65a27d9c737c20429528cabcf315f98b99334`.
The evaluation JSON SHA-256 is
`45f8b1caf9281307d4369fb28431cde23185e92ed298138d4133b4eb03eee054`.

## Hypothesis and decision

The isolated loss change will raise Rel-L2 **or** MVPE by ≥0.30 versus E004,
keep the other within 0.25 of E004, keep TKE ≥70.247, and keep Time ≥86.

| Gate | Threshold | E013 | Result |
|---|---:|---:|---|
| Rel-L2 vs E004 | ≥ 95.020 (+0.30) **or** drop ≤0.25 | 93.364 (−1.356) | **fail** |
| MVPE vs E004 | ≥ 95.443 (+0.30) **or** drop ≤0.25 | 94.767 (−0.376) | **fail** |
| TKE floor | ≥ 70.247 | 78.555 | pass |
| Time floor | ≥ 86.000 | 88.120 | pass |

Hypothesis rejected. Rel-L2 and MVPE moved the wrong way. TKE rose 8.058 and
now matches the leaky released `sim_real_ft` FNO (78.629) on this stream.

## Setup

- Controls held equal to E004: sim-pretrain FNO init, official `mean_std_real`,
  56 `real_regime_v1/train` trajectories / 2,310 windows, Adam `3e-4`, cosine,
  batch 4, 600 updates, seed 0, no early stopping.
- Isolated change: joint physical relative MSE → official Rel-L2 / TKE /
  MVPE-L2 with SPS weights 0.5 / 0.3 / 0.2. Not a rerun of E011
  (energy-weighted Rel-L2 is kept; TKE is explicit; MVPE is official L2).
- Eval interval: frozen E006 `(0.025, 0.1)`.
- Training 107.58 s; first/final mean-25 loss `0.54356/0.22744`. Components
  Rel-L2 `0.365 → 0.124`, TKE `1.004 → 0.494`, MVPE `0.300 → 0.086`. Peak
  allocated 2.463 GiB.

Config SHA-256:
`6f4296e50f890fd51577b2fa686e7336321ff99e1f9826eead7a13315966f174`.

## Overall results

Primary column is `local_bench_v1` / `v1_val`.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | 72.697 | 94.427 | **99.630** | 13.772 |
| E004 / E007 | **94.720** | 70.497 | **95.143** | 88.068 | **35.067** |
| **E013 score-aligned** | 93.364 | **78.555** | 94.767 | 88.127 | 31.686 |

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 94.608 | 78.864 | 96.290 | 88.133 | 34.942 |
| AoA-only | 92.796 | 78.333 | 94.021 | 88.125 | 30.401 |
| Joint | 93.013 | 79.064 | 94.755 | 88.034 | 29.011 |

`complement_val` stress: E007 96.196/70.939/96.442/88.071/45.979 versus E013
95.111/79.406/96.231/88.107/43.089.

## Interpretation and decision

1. Explicit official TKE-rel-L2 is a strong TKE trainer. Raw TKE error starts
   near 1.0 versus Rel-L2 0.37, so a 0.3 TKE weight still dominates early
   gradients and buys leaky-model TKE at Rel-L2/MVPE's expense.
2. Do not promote E013 over E007. Do not retune 0.5/0.3/0.2, and do not
   train this mix longer, on `real_regime_v1` validation.
3. Keep E007 as the submission baseline. A later TKE-aware recipe would need
   a **pre-registered train-scale** mix, not a val search over these weights.
