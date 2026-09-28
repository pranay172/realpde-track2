# E016 full-release submission archive

Run: **2026-08-16 (IST)**

E016 packages the E015 packed FNO and the E006 `(α=0.025, β=0.1)` interval as
a Track 2 zip. The ignored archive is
`artifacts/e016/e016_candidate.zip` (186,177,568 bytes, SHA-256
`6c55286aa7d2b08d6238b8cb6040efc3adf94a6fb93c9ac944b2729fbc258f86`).
The extracted tree is 201,422,929 bytes. The local report SHA-256 is
`ffc7f285b54dd0fa334a611fda854cd8a87e4a2d58365b98111ed862a78aca7b`.

## Hypothesis and decision

A no-adaptation `submission.py` wrapping that checkpoint and interval will pass
the archive/container contract. `v1_val` is not a quality gate.

| Gate | Result |
|---|---|
| `submission.py` at zip root | pass |
| Extracted size < 256 MB | pass (201,422,929) |
| Host `local_eval` on kit example data | pass |
| Reset replay bitwise | pass |
| Official Track 2 image, `--network none` | pass (4.23 s) |
| `v1_val` quality proxy | skipped (contaminated) |

Hypothesis accepted as a contract-valid archive. Official Rel-L2/MVPE/`final_score`
were later recorded on Codabench (see below).

## Package

Root contents: `submission.py`, `model.pth`, slim `load_baseline.py` + FNO-only
`rpde_baselines`. No CNO/Transolver sources. No adaptation: `adapt_loss` is
always `None`. Bounds are the E006 physical formula returned in official
normalized space on every step.

Checkpoint SHA-256
`1e61d570bf941985d5caa0a033ada7c59a1d609c38eef918ae0058583ba73343`.
Config SHA-256
`db576637659e9e6a17fdcd0fdb5ff9567efcd5230cef7218c90dd179e0a663aa`.

Example-data `local_eval` scores are not leaderboard-comparable.

## Codabench result

User-reported Development scores for this exact archive, versus the submitted
E007 zip:

| Subscore | E007 official | E016 official | Δ |
|---|---:|---:|---:|
| Rel-L2 | 93.779908 | 93.799842 | +0.020 |
| TKE | 69.824232 | 69.752667 | −0.072 |
| MVPE | 91.189871 | 91.710703 | +0.521 |
| Time | 89.437108 | 89.478629 | +0.042 |
| SPS | 26.777889 | 26.845200 | +0.067 |
| `final_score` | 75.423409 | **75.496987** | +0.074 |

Pre-registered E015 official gates (Rel-L2 or MVPE +≥0.30 vs E007, or
`final_score` > 75.423409, and Time ≥86) pass: MVPE +0.521, final +0.074,
Time 89.479. Rel-L2 +0.020 misses +0.30. SPS +0.067 is noise-scale. TKE
slipped 0.072.

The remaining official-minus-E007-local gaps are Rel-L2 −0.920, TKE −0.744,
MVPE −3.432, Time +2.193, SPS −8.222. Extra released regimes recovered some
wake MVPE; they did not close the SPS hole.

## Interpretation and decision

1. Freeze this official row. E016 is the current Codabench baseline.
2. Do not retune E015 weights or the E006 `(0.025, 0.1)` interval to this
   hidden-set look.
3. E007 remains the leakage-safe local reference. Never score E015 on `v1_val`.
