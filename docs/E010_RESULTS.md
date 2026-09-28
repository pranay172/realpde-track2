# E010 complementary-fold E004 recipe

Run: **2026-08-15 (IST)**

E010 retrains the E004 recipe on a pre-registered complementary regime split
and evaluates it once on that fold only. Holdouts are the lowest and highest
audited nominal Re (3750, 26700) and AoA 5, chosen from the release grid, not
from `real_regime_v1` scores. Ignored artifacts live under `artifacts/e010/`.

This model saw original v1 validation trajectories during training. It is
**not** a leakage-safe replacement for E007.

## Hypothesis and decision

The E004 recipe will beat persistence on Rel-L2 and MVPE on the complementary
validation stream and will not drop TKE below 66.969.

| Gate | Threshold | E010 FNO | Result |
|---|---:|---:|---|
| Rel-L2 vs fold persistence | > 94.213 | 96.053 | pass |
| MVPE vs fold persistence | > 95.602 | 96.230 | pass |
| TKE floor | ≥ 66.969 | 71.686 | pass |

Hypothesis accepted. TKE is 0.088 below this fold's persistence (71.774).

## Setup

- Split: [`real_regime_complement_v1`](../configs/splits/real_regime_complement_v1.json).
  Train 61 trajectories / 2,520 windows. Val: Re-only 6/252, AoA-only 13/527,
  joint 1/42 (821 windows).
- Recipe: identical to E004 (sim-pretrain FNO init, official stats, Adam
  `3e-4`, cosine, batch 4, 600 updates, physical `u/v` relative MSE).
- Bounds: official default ±5%. E006 was not reused because its calibration
  Re included 3750.
- Persistence measured on the same 821-window stream before FNO evaluation.
- Training 106.56 s; first/final mean-25 loss `0.08641/0.00989`.

Config SHA-256
`e500abeafecdaa97956ae292b09a4d3aa5e0cace12d603cfe1457f1d47ac359d`.
FNO eval JSON SHA-256
`79a0784fe7b0540e15e2148dea4ad59d50b9f5c7787f181a13eab72405567c68`.

## Overall results

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Complement persistence | 94.213 | **71.774** | 95.602 | **99.629** | 18.581 |
| **E010 FNO** | **96.053** | 71.686 | **96.230** | 88.093 | 19.141 |

| Partition | Rel-L2 | TKE | MVPE |
|---|---:|---:|---:|
| Re-only (3750/26700) | 94.318 | 72.222 | 94.495 |
| AoA-only (5°) | 96.953 | 71.030 | 97.157 |
| Joint (3750/5) | 95.474 | 77.194 | 95.327 |

## Interpretation and decision

1. The E004 recipe still beats persistence on Rel-L2 (+1.840) and MVPE
   (+0.628) under a second, grid-defined regime cut. The original local win
   is not an artifact of holding out Re 12675/21600 and AoA 15.
2. TKE again trails persistence slightly, matching the v1 pattern.
3. Do not submit or promote this checkpoint: it trained on v1 validation
   files. Keep E007 as the leaderboard baseline.
4. I008 is closed. A later candidate still needs a v1-safe point change that
   does not retune to Codabench.
