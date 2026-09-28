# E008 last-layer previous-window TTT

Run: **2026-08-15 (IST)**

E008 adapts only FNO `fc1`/`fc2` with one SGD step at `1e-3` on the cached
previous window, then predicts with frozen E006 `(0.025, 0.1)` bounds. The
ignored evaluation JSON has SHA-256
`27e41f2a94ee889f6c871c98f317d69cb25a09bc2e1dc269cd0c93de4195e2d0`.

## Hypothesis and decision

Last-layer TTT will raise Rel-L2 or MVPE by at least 0.15 versus no-adaptation
E007, keep Rel-L2 ≥ 94.470, MVPE ≥ 94.893, TKE ≥ 70.247, and Time ≥ 82.

| Gate | Threshold | E008 | Result |
|---|---:|---:|---|
| Rel-L2 or MVPE gain vs E007 | ≥ +0.15 | +0.016 / +0.027 | **fail** |
| Rel-L2 floor | ≥ 94.470 | 94.736 | pass |
| MVPE floor | ≥ 94.893 | 95.170 | pass |
| TKE floor | ≥ 70.247 | 70.502 | pass |
| Time floor | ≥ 82.000 | 84.014 | pass |

Hypothesis rejected. Quality change is noise; Time is cheaper than E005 but still
worse than E007.

## Setup

- Base: frozen E004 packed FNO SHA-256 `75cc07b4…`.
- Trainable: `fc1.*` and `fc2.*` only; all other weights frozen.
- Optimizer: SGD `1e-3`, momentum 0, one step, 1,006 updates / 25 resets.
- Loss: per-window physical relative MSE on scored `u/v`.
- Bounds: frozen E006 `(0.025, 0.1)`.
- BN remains in eval / running-stats mode.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, RTX 5050, seed 0.
  Mean step 26.39 ms; peak allocated 0.903 GiB; mean adapt loss 0.01332.

Config SHA-256:
`09182e093f8cd2b49cfd74758fbc06989c232ca93464f361210b95afde205bd7`.

## Overall results

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E007 no-adapt | 94.720 | 70.497 | 95.143 | **87.286** | 35.067 |
| E005 full-model `1e-4` | 94.724 | 70.498 | 95.149 | 78.255 | 15.047 |
| **E008 last-layer `1e-3`** | 94.736 | 70.502 | 95.170 | 84.014 | 35.210 |

E005 SPS used the default ±5% band. E008 uses E006 bounds.

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 95.856 | 70.062 | 96.382 | 83.773 | 39.269 |
| AoA-only | 94.214 | 70.756 | 94.576 | 84.143 | 33.537 |
| Joint | 94.499 | 70.279 | 95.141 | 83.975 | 32.419 |

## Interpretation and decision

1. Restricting TTT to the projection head and raising the learning rate still
   does not move Rel-L2 or MVPE by a decision-relevant amount.
2. Last-layer backward is cheaper than E005 (26.4 ms vs 56.3 ms) but still
   costs about 3.3 Time points versus no-adaptation E007.
3. Do not retune this learning rate or grow the trainable set on
   `real_regime_v1` validation. Stop this one-step SGD TTT family.
4. Keep E007 as the submission baseline. The next isolated change is a
   stronger split-safe point fine-tune (I007).
