# Results — Track 2

## Best recorded development-leaderboard result

User-supplied Codabench result for **E033**, not independently fetched server logs:

| Rel-L2 | TKE | MVPE | Time | SPS | final_score |
|---:|---:|---:|---:|---:|---:|
| 93.614317 | 77.141376 | 93.391126 | 88.185671 | 32.109079 | **77.615784** |

E029 scored 76.760173; E033 improved the reported final_score by 0.855611.
All four quality components improved; Time decreased by 0.557739. A single
server comparison cannot isolate initialization, sampling, budget or timing causes.
No score above 80 or final private-test placement is claimed.

## Selected controlled local evidence

E032 stride20/1800, seed 0, same inference wrapper:

| Fold | Rel-L2 | TKE | MVPE | SPS |
|---|---:|---:|---:|---:|
| Main | 94.447761 | 77.114139 | 95.798160 | 37.542918 |
| Complement | 96.120619 | 76.470977 | 97.079123 | 47.262038 |

Seed 1 confirmed the locked acceptance criteria. Dense stride-1 training did
not pass all preservation gates; higher local SPS alone was insufficient.
See [E032](docs/E032_RESULTS.md) for full factorial controls.

## Lessons and failures

- One-step gradient TTT incurred cost with negligible gain.
- Mean/fluctuation decomposition and causal revealed-target variance adaptation
  were more useful than several residual-feedback or variance-head alternatives.
- Simulation/real mixing without alignment degraded TKE.
- Initialization must be verified by loaded tensors, not config labels:
  historical E020/E021 evidence is qualified/corrected.
- Exact reset replay and package/research parity were tested separately from
  quality. E033 historically passed 108 evaluator-compatible tests; that is
  a recorded historical run, not a post-cleanup test result.

Historical E033 archive SHA:
`50a5b3dc2550b4129eabfe917085e98aedceb0224188c21e269856ae60b6ba93`.
The archive and trained checkpoint were deleted. The organizer's small examples
are external smoke-test dependencies, not included in this snapshot and not
the original training release or a clean audit.

See [E033](docs/E033_RESULTS.md) for official scores, runtime qualifications,
packed-model provenance and reproduction commands. The remaining final-score
gap was 2.384216; no linear extrapolation from local subscores is justified.
