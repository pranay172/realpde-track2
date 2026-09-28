# E014 train-scaled official score-aligned loss

Run: **2026-08-16 (IST)**

E014 repeats the E004 fine-tune with one isolated change: the official
`0.5 Rel-L2 + 0.3 TKE + 0.2 MVPE-L2` mix from E013, after dividing each term
by its frozen E004 mean on `real_regime_v1/train`. Scales were measured on
the 2,310 train windows only and written into the config before training.
Evaluation uses the frozen E006 interval. Ignored artifacts live under
`artifacts/e014/`. Packed checkpoint SHA-256
`aadf61e7d049ae77f86a71878cdf63acc3c8cca8b9db8d6edc7e4df4630c1979`.
Evaluation JSON SHA-256
`a402db9e54225f5e709a7253552abff8d6e053a4e50005c9dcbc7e7c72541a8a`.
Train-scale JSON SHA-256
`c3624ab49ff44d6d6fefb604a17c6f050bcd748c0f2c855eab5fb4fda36e5834`.

## Hypothesis and decision

The isolated scaled mix will raise Rel-L2 **or** MVPE by ≥0.30 versus E004,
keep the other within 0.25 of E004, keep TKE ≥70.247, and keep Time ≥86.

| Gate | Threshold | E014 | Result |
|---|---:|---:|---|
| Rel-L2 vs E004 | ≥ 95.020 (+0.30) **or** drop ≤0.25 | 94.519 (−0.201) | **fail** gain / pass drop |
| MVPE vs E004 | ≥ 95.443 (+0.30) **or** drop ≤0.25 | 95.097 (−0.046) | **fail** gain / pass drop |
| TKE floor | ≥ 70.247 | 75.612 | pass |
| Time floor | ≥ 86.000 | 88.131 | pass |

Hypothesis rejected. Neither Rel-L2 nor MVPE gained 0.30. The drop constraint
held. TKE rose 5.115 and now beats persistence (72.697).

## Setup

- Controls equal to E004: sim-pretrain FNO init, official `mean_std_real`,
  56 train trajectories / 2,310 windows, Adam `3e-4`, cosine, batch 4, 600
  updates, seed 0.
- Isolated change versus E013: same official functionals and 0.5/0.3/0.2
  weights, each divided by frozen E004 train means
  `s_rel=0.088526`, `s_tke=0.812127`, `s_mvpe=0.082204`.
- Eval interval: frozen E006 `(0.025, 0.1)`.
- Training 107.66 s. First/final mean-25 loss `2.764/0.935`. Raw components
  Rel-L2 `0.299→0.097`, TKE `0.919→0.560`, MVPE `0.302→0.074`. Scaled
  first25 Rel/TKE/MVPE `3.380/1.132/3.671` (Rel and MVPE dominate at init).
  Peak allocated 2.463 GiB.

Config SHA-256
`89cd179d484013f309d127e5aa4366c5d35b6c9aaf855254395093d05f4fd7cb`.

## Overall results

Primary column is `local_bench_v1` / `v1_val`.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | 72.697 | 94.427 | **99.628** | 13.772 |
| E004 / E007 | **94.720** | 70.497 | **95.143** | 88.033 | 35.067 |
| E013 unscaled mix | 93.364 | **78.555** | 94.767 | 88.120 | 31.686 |
| **E014 train-scaled** | 94.519 | 75.612 | 95.097 | 88.070 | **35.339** |

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 95.629 | 76.369 | 96.509 | 88.169 | 39.580 |
| AoA-only | 93.997 | 75.278 | 94.416 | 88.132 | 33.575 |
| Joint | 94.314 | 75.298 | 94.995 | 87.978 | 32.541 |

`complement_val` stress: E007 96.196/70.939/96.442/88.086/45.979 versus E014
95.945/77.786/96.691/88.084/47.023.

## Interpretation and decision

1. Train-scaling did what it was supposed to: Rel-L2 no longer collapses
   (E013 −1.356 vs E014 −0.201) and TKE still moves (+5.115, above
   persistence). It did not buy the Rel-L2/MVPE gains the gates required.
2. Do not promote E014 over E007. Do not retune the frozen train scales or
   the 0.5/0.3/0.2 weights on `real_regime_v1` validation.
3. Keep E007 as the submission baseline. Stop this official-loss family on
   this split. I011 remains unused until a better point model exists.
