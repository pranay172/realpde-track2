# E009 longer split-safe FNO fine-tune

Run: **2026-08-15 (IST)**

E009 repeats the E004 recipe with a pre-registered 3× update budget (1800
Adam steps, 7200 examples, 3.12 epochs). Evaluation uses the frozen E006
interval. Ignored artifacts live under `artifacts/e009/`. The packed
checkpoint SHA-256 is
`b821e85d28626ee43da6e5ba981278edb2d289267900072f5a8c754344cf4a4b`.
The evaluation JSON SHA-256 is
`7143c15f59c17faa6da5ca3b64ea712cf472fd008cfc664b72af36749b7cf87e`.

## Hypothesis and decision

Tripling the E004 budget will raise Rel-L2 by ≥0.30 and MVPE by ≥0.20 versus
E004, keep TKE ≥70.247, and keep Time ≥86.

| Gate | Threshold | E009 | Result |
|---|---:|---:|---|
| Rel-L2 vs E004 | ≥ 95.020 (+0.30) | 94.994 (+0.274) | **fail** |
| MVPE vs E004 | ≥ 95.343 (+0.20) | 95.234 (+0.091) | **fail** |
| TKE floor | ≥ 70.247 | 72.056 | pass |
| Time floor | ≥ 86.000 | 88.067 | pass |

Hypothesis rejected. Rel-L2 almost cleared the gate; MVPE barely moved. TKE
improved 1.559 and is now within 0.641 of persistence.

## Setup

- Controls held equal to E004: sim-pretrain FNO init, official `mean_std_real`,
  56 `real_regime_v1/train` trajectories, Adam `3e-4`, cosine, batch 4, seed 0,
  physical `u/v` relative MSE, no early stopping.
- Isolated change: 600 → 1800 updates.
- Eval interval: frozen E006 `(0.025, 0.1)`.
- Training 318.35 s; first/final mean-25 loss `0.10436/0.00610` (E004 final
  mean-25 was 0.00945). Peak allocated 2.645 GiB.

Config SHA-256:
`f40add4f640af78e74fe5f6223d1c70280641d839e1b5abb0f43284401bb7952`.

## Overall results

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | **72.697** | 94.427 | **99.510** | 13.772 |
| E004 / E007 | 94.720 | 70.497 | 95.143 | 87.286 | 35.067 |
| **E009 1800 updates** | 94.994 | 72.056 | 95.234 | 88.067 | 36.637 |

E007 Time/SPS include the submission wrapper; E009 Time/SPS are the eval
harness with the same interval.

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 96.080 | 71.302 | 96.479 | 88.061 | 40.948 |
| AoA-only | 94.484 | 72.457 | 94.636 | 88.062 | 34.857 |
| Joint | 94.788 | 71.938 | 95.106 | 88.132 | 33.697 |

## Interpretation and decision

1. Extra epochs mainly bought TKE, not the Rel-L2/MVPE gains the gates
   required. MVPE, the official hidden-set hole, barely moved.
2. Do not promote E009 over E007 and do not train this recipe longer on the
   same validation split.
3. Keep E007 as the submission baseline. The unused complementary-fold check
   remains the unused half of I007 and is now a separate later idea.
