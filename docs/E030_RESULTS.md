# E030 — Spatially pooled AR(2) fluctuation donor

Date: **2026-08-23 (IST)**. Status: **complete — rejected by locked quality gates**.

## Objective

E029 showed that emaV recovers hidden-stream fluctuation energy but gives up
some pointwise accuracy, suggesting that amplitude is better calibrated than
phase. E030 tested the smallest causal phase model compatible with the 20-frame
input: one stable ridge-AR(2) fit jointly across every pixel and both measured
channels, then rolled forward for 20 frames and blended into the FNO
fluctuation before emaV.

The experiment used a newly trained model on
`real_regime_complement_v1/train`. Its `val_re`, `val_aoa`, and `val_joint`
partitions were opened once for the fixed comparison. This removes the
training overlap that made E020's earlier complement result diagnostic only.

## Locked design

- Model: dual-head mean/fluctuation FNO, 600 Adam updates, batch size 4,
  learning rate `3e-4` with cosine decay, seed 0.
- Initialization: unpacked and strictly loaded official simulation-pretrained
  FNO, converted identically into the dual heads.
- Baseline: mean-bias EMA (`gamma=0.9`) + emaV (`gamma=0.9`, smoothing
  `sigma=0.5`, ratio clip `0.2–5.0`) + bounds `(0.02, 0.14)`.
- Candidate: baseline plus pooled AR(2), relative ridge `1e-4`, characteristic
  root cap `0.98`, fluctuation blend `0.25`.
- The AR forecast is centered over its output horizon, so it cannot alter the
  separately calibrated temporal-mean path.
- No coefficient, blend, or gate was changed after viewing the complement
  holdout.

Config SHA-256 values:

| Config | SHA-256 |
|---|---|
| Experiment | `3be439589bd987460b601c9b2d441d7b3feab30df2643a8899cc7ccd8afcffee` |
| Baseline evaluation | `7ef0c52fada56027bd7d13d49eb136c556ea382600ce3af1f58ba43d09cf15f1` |
| Candidate evaluation | `a5fcee64ecbd5e4ffc1d6a4129b9f1c58db2ec2092f9e1a01f833aaa9ba7eba2` |

## Initialization audit and correction

While smoke-testing E030, the historical dual-head trainer reported
`converted: False` for a packed `state_fp16` checkpoint. It passed the wrapper
dictionary directly to the initializer and did not reject missing keys, so the
configured weights were never installed. Consequently:

- The audit initially inferred that E020 trained from seeded random
  initialization. **Later checkpoint inspection (2026-09-21) contradicts that
  inference:** its verified original checkpoint has BatchNorm counters 5,600,
  matching simulation's 5,000 plus 600 updates. The exact historical execution
  remains unresolved; E020 must not be used as a random-init control.
- E021 trained from its seeded random initialization, not E004.
- Their learned checkpoints and all subsequent measured results remain valid;
  only the claimed initialization provenance and step-zero equivalence were
  wrong.

The trainer now recognizes packed `state_fp16` and common training-checkpoint
wrappers, unpacks complex-safe fp16 weights, and raises on any missing or
unexpected initialization key. E030 printed `converted: True` before training.
The corresponding corrections are recorded in the E020 and E021 result notes.

## Training record

- Environment: `.venv-gpu`, RTX 5050 Laptop GPU.
- Trainable parameters: 50,358,342 (all dual-head model parameters).
- Runtime: 129.11 seconds for 600 updates.
- Loss: step 1 `0.119408`; step-100 trailing mean `0.022552`; step-300 trailing
  mean `0.014337`; step 600 `0.011459`, trailing mean `0.010210`.
- Packed checkpoint: `artifacts/e030/fno_dual_head_fp16.pth`, 201.40 MB.
- Checkpoint SHA-256:
  `abe3c93ee504aaa467b1fd9969715df506ed139fc28b2f814b34e4029d81a333`.

The checkpoint is intentionally ignored as a generated artifact; the configs
and exact training recipe are tracked.

Peak GPU allocation was not instrumented for this rejected internal run. This
is a reporting limitation, not a submission compatibility claim.

## Locked complement result

All 821 windows were evaluated. Deltas are candidate minus baseline; higher is
better for every reported competition score.

| Metric | emaV baseline | pooled AR(2) + emaV | Delta | Gate | Result |
|---|---:|---:|---:|---:|---|
| Rel-L2 | 95.944491 | 96.026476 | +0.081985 | at least +0.15 | **fail** |
| TKE | 76.625625 | 76.416429 | −0.209196 | no drop | **fail** |
| MVPE | 96.971424 | 96.971424 | +0.000000 | no worse than −0.05 | pass |
| SPS | 46.636387 | 46.782937 | +0.146550 | no worse than −0.25 | pass |
| Time | 83.862058 | 86.464063 | +2.602005 | candidate at least 86 | nominal pass* |
| Five-component mean | 80.007997 | 80.532266 | +0.524269 | at least +0.30 | nominal pass* |

Coverage changed from 87.247% to 87.495%. The quality-only four-component
average changed by just `+0.004835`, effectively neutral.

Ignored evaluation records:

| Record | SHA-256 |
|---|---|
| `artifacts/e030/complement_emav.json` | `d5ba6fa27a9d58c83e31846d072b1c64ec525564aaafa90759d35ed74c048f26` |
| `artifacts/e030/complement_ar2_emav.json` | `352c47ace68bb73d6bd9d65309b4481aadce5d4626a222f65d8477d4b445e941` |

\* The candidate cannot physically be faster because it adds AR fitting and a
20-step rollout. The apparent Time gain is session/runtime noise in the current
environment and entirely explains the apparent five-component gain. It is not
evidence for promotion. The local mean also is not comparable to Codabench's
official final score.

## Verification

- Evaluator-compatible CPU suite: **96/96 passed**.
- Source/script/test bytecode compilation and `git diff --check`: passed.
- Packed-checkpoint regression test covers real and complex tensors.
- A real-checkpoint trainer smoke reported strict `converted: True`; the full
  CUDA run and both 821-window evaluations completed without failures.
- No official-container/package replay was required because the mechanism was
  rejected and no E029 submission file changed.

## Decision

Reject the pooled AR(2) donor. It misses both decisive quality gates: the
Rel-L2 improvement is too small and TKE regresses. Do not retune its blend or
ridge against the opened holdout. E029 remains the submission baseline at
76.760173; E030 does not establish progress toward an official score of 80.

The strict-init complement checkpoint and loader correction remain useful.
The strongest next diagnostic is a two-fold initialization crossover: train a
correctly simulation-initialized model on `real_regime_v1` and a seeded-random
model on the complement split, then compare them with the existing E020
(historical/v1, initialization uncertain) and E030 (simulation/complement) anchors. That separates the newly
discovered initialization effect from the split effect before risking a
full-data retrain and submission.
