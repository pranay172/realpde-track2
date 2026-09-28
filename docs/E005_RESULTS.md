# E005 one-step previous-window TTT

Run: **2026-08-15 (IST)**

E005 wraps the frozen E004 packed FNO in the official previous-window
adaptation loop: one SGD step on the cached previous input and the revealed
previous target, then a no-grad current prediction. Ignored artifacts live
under `artifacts/e005/`. The evaluation JSON has SHA-256
`c9e4367f66e91e632f34c7e4ce1525222472f455153b2aa13efc894de2f5469d`.

## Hypothesis and decision

One SGD step at `1e-4` with E004's physical `u/v` relative MSE will raise TKE
above 70.497 without dropping Rel-L2 below 94.470 or MVPE below 94.893, and
will keep Time at least 80.

| Gate | Threshold | E005 | Result |
|---|---:|---:|---|
| TKE vs E004 | > 70.497 | 70.498 | pass (0.001; not a material gain) |
| Rel-L2 floor | ≥ 94.470 | 94.724 | pass |
| MVPE floor | ≥ 94.893 | 95.149 | pass |
| Time floor | ≥ 80.000 | 78.255 | **fail** |

Hypothesis rejected. Point scores are unchanged from E004 within noise; Time
regressed 9.912 points.

## Setup

- Base: frozen `artifacts/e004/fno_fp16.pth` SHA-256
  `75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`.
- Stream: `real_regime_v1` validation, 1,031 windows / 25 trajectories.
- Protocol: adapt only when a previous target and cached previous input exist
  (1,006 updates). First step of each trajectory is no-adaptation.
- Optimizer: SGD `1e-4`, momentum 0, one step, all parameters.
- Loss: per-window physical relative MSE on scored `u/v`.
- BatchNorm: left in eval / running-stats mode so batch-one updates cannot
  overwrite those buffers. Reset restores E004 weights, a fresh optimizer, and
  a cleared input cache.
- Bounds: none; official default ±5% band.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, RTX 5050, seed 0.
  Evaluation wall 113.94 s; timed total 58.03 s; peak allocated 1.295 GiB.

Config SHA-256:
`9985a342427c866e844cff19eb4b9112945b0d893fa4c6d2056fe26e3df6bfb5`.

## Overall results

Higher is better. Time is local-hardware.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | **72.697** | 94.427 | **99.510** | 13.772 |
| E004 no-adapt FNO | 94.720 | 70.497 | 95.143 | 88.167 | 15.024 |
| **E005 one-step TTT** | 94.724 | 70.498 | 95.149 | 78.255 | 15.047 |

E005 mean step 56.28 ms versus E004 13.13 ms. Mean adaptation loss 0.01337.
SPS coverage 30.030% versus E004 29.982%. CPU load of the same wrapper
reproduced a finite two-step stream and a bitwise reset replay.

## Regime breakdown

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 95.845 | 70.058 | 96.356 | 78.174 | 18.386 |
| AoA-only | 94.201 | 70.752 | 94.556 | 78.292 | 13.527 |
| Joint | 94.491 | 70.277 | 95.127 | 78.289 | 13.833 |

Deltas versus E004 are at most a few thousandths on the quality scores.

## Interpretation and decision

1. This lightweight full-parameter step is a no-op on Rel-L2, TKE, and MVPE.
   TKE remains 2.199 below persistence.
2. The extra adapt-forward and backward roughly 4.3x the per-step time and
   failed the pre-registered Time gate.
3. Do not promote E005 and do not retune this learning rate on the same
   validation stream.
4. Keep E004 as the leakage-safe point-prediction reference. The next isolated
   change should not repeat a 1e-4 full-model update; SPS calibration on frozen
   E004 residuals is the higher-value follow-up because default coverage is
   still 30%.
