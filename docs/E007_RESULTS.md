# E007 no-adaptation submission candidate

Run: **2026-08-15 (IST)**

E007 packages the frozen E004 packed FNO and the E006 `(α=0.025, β=0.1)`
interval as a Track 2 zip. The ignored archive is
`artifacts/e007/e007_candidate.zip` (186,181,793 bytes, SHA-256
`2f2f3dcad4a43618ea5c72836cee71007f8602846943a2e24d1b6ca5c12b4ba5`).
The extracted tree is 201,422,923 bytes. The local report SHA-256 is
`41fab9a85e6ee6f9c86d11d66ffaa0d069af913e79ae3be1e29326e13c133da0`.

## Hypothesis and decision

A no-adaptation `submission.py` wrapping that checkpoint and interval will pass
the archive/container contract and reproduce E006 validation Rel-L2/TKE/MVPE
within 0.01 and SPS within 0.05 of 35.067.

| Gate | Result |
|---|---|
| `submission.py` at zip root | pass |
| Extracted size < 256 MB | pass (201,422,923) |
| Host `local_eval` on kit example data | pass |
| Official Track 2 image, `--network none` | pass (5.64 s) |
| Reset replay bitwise | pass |
| Val Rel-L2/TKE/MVPE vs E006 | pass (bitwise) |
| Val SPS vs E006 35.066879 | pass (`35.066880`) |

Hypothesis accepted. This archive is the first Track 2 submission candidate.

## Package

Root contents: `submission.py`, `model.pth`, slim `load_baseline.py` + FNO-only
`rpde_baselines`. No CNO/Transolver sources. No adaptation state: `adapt_loss`
is always `None`, and `reset_ttt_state` only keeps the model in eval mode.
Bounds are computed every step in official normalized space from the same
physical formula as E006.

Checkpoint SHA-256
`75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`.
Config SHA-256
`66d581d3e1dee1629effa89ba70b8506a82ca20d1b6f404a460db333814ad8cf`.

## Local validation proxy

Same 1,031-window `real_regime_v1` stream as E006. Point errors are bitwise
identical. SPS differs by `<1e-9`. Time is 87.286 versus E006 88.130 because
bound construction is now inside the timed `ttt_step`.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E006 reference | 94.720 | 70.497 | 95.143 | 88.130 | 35.067 |
| **E007 wrapper** | 94.720 | 70.497 | 95.143 | 87.286 | 35.067 |

Example-data `local_eval` scores are not leaderboard-comparable. They only
prove the official scorer accepted participant bounds.

## Codabench result

User-reported Development scores for this exact archive:

| Subscore | Local E007 | Codabench | Δ |
|---|---:|---:|---:|
| Rel-L2 | 94.720 | 93.779908 | −0.941 |
| TKE | 70.497 | 69.824232 | −0.673 |
| MVPE | 95.143 | 91.189871 | −3.953 |
| Time | 87.286 | 89.437108 | +2.151 |
| SPS | 35.067 | 26.777889 | −8.289 |
| `final_score` | unpublished locally | **75.423409** | — |

User-reported rank **#55**. Leader ≈81; top-10 threshold ≳80.

The wrapper is not broken: Rel-L2/TKE only slipped about one point, Time improved
on the official GPU, and SPS stayed well above the default-band local 15.024.
The hidden stream is harder, especially wake MVPE, and that accuracy drop also
punishes SPS.

## Interpretation and decision

1. Freeze this exact zip SHA as the first official baseline. E016 later
   replaced it as the current Codabench row (final 75.496987); this zip
   remains the leakage-safe local reference.
2. Do not retune E004 weights or the E006 interval from this hidden-set row.
3. Official Time headroom makes a later last-layer TTT cheaper than E005
   looked locally. A later point-model change needs unused train-side evidence.
