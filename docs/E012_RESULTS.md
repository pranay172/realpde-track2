# E012 coverage-targeted interval on frozen E004

Run: **2026-08-16 (IST)**

E012 does not retrain. It replaces E006's max-SPS pick `(0.025, 0.1)` with the
already-computed train-only coverage-targeted pick `(0.025, 0.15)` on the same
frozen E004 packed FNO. Ignored evaluation lives under `artifacts/e012/`. The
E004 checkpoint SHA-256 is unchanged:
`75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`.
The evaluation JSON SHA-256 is
`df795df02eec0cf78f772fb00852230e88a2a6489a58e69c721deab2e57fdf05`.

## Hypothesis and decision

Applying this frozen interval will keep Rel-L2/TKE/MVPE within 0.01 of E004,
keep Time ≥86, raise validation coverage above E006's 66.599%, and not drop
SPS below 34.067.

| Gate | Threshold | E012 | Result |
|---|---:|---:|---|
| Rel-L2 vs E004 | 94.720 ± 0.01 | 94.720 (0.000) | pass |
| TKE vs E004 | 70.497 ± 0.01 | 70.497 (0.000) | pass |
| MVPE vs E004 | 95.143 ± 0.01 | 95.143 (0.000) | pass |
| Time | ≥ 86.000 | 88.096 | pass |
| Coverage vs E006 | > 66.599% | 75.463% | pass |
| SPS vs E007 | ≥ 34.067 | 35.938 (+0.871) | pass |

Hypothesis accepted. Point errors are bitwise identical to E004/E006.

## Setup

- Base: frozen `artifacts/e004/fno_fp16.pth`. No new weights.
- Formula: `half = α|pred| + βσ` with `σ = 0.0563870259`.
- Selection (train-only, from `artifacts/e006/calibration.json`): among
  calibration candidates with `sps_score ≥ max − 1.0`, maximum coverage;
  ties break by smaller width, then smaller α, then smaller β. That unique
  pick is `(0.025, 0.15)` (cal SPS 43.277, coverage 87.004% versus E006
  `(0.025, 0.1)` cal 44.234 / 80.431%). Validation was not used to choose
  the pair.
- Eval: one official-protocol `real_regime_v1` stream, then `local_bench_v1`.

Config SHA-256:
`14168880ca23dc730bd82f2ce65c3708d281b3554882b86ddea62cecee38f2e1`.

## Overall results

| Model | Rel-L2 | TKE | MVPE | Time | SPS | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | **72.697** | 94.427 | **99.608** | 13.772 | — |
| E006 / E007 `(0.025, 0.1)` | 94.720 | 70.497 | 95.143 | 88.018 | 35.067 | 66.599% |
| **E012 `(0.025, 0.15)`** | 94.720 | 70.497 | 95.143 | 88.096 | **35.938** | **75.463%** |

`local_bench_v1` `complement_val` (stress; E004 trained on most of those
files): E007 96.196/70.939/96.442/88.139/45.979 versus E012
96.196/70.939/96.442/88.088/44.718. Wider band costs 1.261 SPS there
because coverage is already high.

| Partition | Rel-L2 | TKE | MVPE | Time | SPS | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Re-only | 95.841 | 70.057 | 96.349 | 87.979 | 39.131 | 80.500% |
| AoA-only | 94.197 | 70.751 | 94.550 | 88.145 | 34.624 | 73.387% |
| Joint | 94.488 | 70.277 | 95.124 | 88.175 | 33.724 | 71.992% |

## Interpretation and decision

1. The isolated interval change does what the train-only rule promised:
   more coverage, unchanged points, Time intact, and a small local SPS
   gain on the harder `v1_val` stream.
2. Adopt `(0.025, 0.15)` as the frozen uncertainty candidate for a later
   E004 package. Do not retune after this look. Do not reopen the E006 grid.
3. Keep the submitted E007 zip as the leaderboard baseline until a new
   archive is explicitly requested. Official SPS was the hole (26.778 vs
   local 35.067); this wider train-only band is the intended cheap attempt
   to recover hidden coverage, not a fit to Codabench.
