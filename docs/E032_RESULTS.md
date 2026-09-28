# E032 — Sampling × training-budget factorial

Status: complete — stride20/1800 accepted after seed-1 confirmation, 2026-09-21.
Official reference remains E029 (76.760173).

## Locked design

E031 selected verified official simulation initialization. Cross stride 20 or 1
with 600 or 1,800 updates on each of the two regime folds, seed 0. All cells use
the same dual-head FNO, batch 4, Adam 3e-4 with cosine decay, physical relative
u/v MSE, normalization, packed evaluation, emaV, and fixed (0.02, 0.14) bounds.
Neither architecture nor deployed runtime graph changes.

Both sampling schemes select trajectories uniformly and available window starts
uniformly with replacement. This deliberately differs from E031's shuffled
window traversal; compare factorial effects against E032's own stride20/600
cell. Dense overlapping offsets are correlated data, not independent examples.
Cosine schedules cover the specified total updates, so the budget comparison
includes the corresponding change in learning-rate schedule.

Gate vs stride20/600: two-fold mean gains of at least 0.15 Rel-L2 and 0.5 SPS;
neither fold loses more than 0.10 Rel-L2, 0.25 TKE, 0.10 MVPE, or 0.25 SPS.
Among passing recipes, select highest two-fold mean SPS; ties within 0.1 prefer
fewer updates, then stride20. Evaluate only that selected recipe and comparator
with seed 1. **Confirmation requires passing the same gate on seed 1**, not
merely retaining a positive pooled effect. This clarification was recorded
before seed-0 selection or any seed-1 runs. Retain historical E020+emaV and
E030's independent complement model as additional quality anchors.

## Reproduction and provenance

Registration: `cb9bcbe`. Study config SHA-256:
`4f3d0d173831daf2c0e3e9466dbb0545f49291f78a2b1efa84e5a34c98c0d08c`.
Run with `.venv-gpu` from the parent workspace:

```bash
../.venv-gpu/bin/python scripts/run_sampling_study.py --initialization checkpoint
../.venv/bin/python scripts/summarize_sampling_study.py --initialization checkpoint
```

Each cell under `artifacts/e032/` retains train/eval configs, logs, checkpoints,
loss history, strict tensor-loading checks, train-file membership, configuration
and checkpoint hashes, memory/runtime, and per-trajectory evaluation metrics.
Generated assets are ignored. Existing retained outputs are resumed only when
their configurations and checkpoint hashes match; different controls require a
fresh output directory. Four provenance-guard regression tests pass. The guard
was added while the original seed-0 orchestrator was already running; it will
apply to subsequent invocations, including seed-1 confirmation.

No submission-facing code has changed. A confirmed recipe still needs a
full-data refit and exact-archive/container verification before submission.
Local component gains do not establish an official final score above 80.

Evaluator-compatible regression suite: 101 tests passed after adding four
synthetic gate/selection/confirmation tests. The suite includes stream reset,
target alignment, and per-trajectory metric aggregation checks.

Packaging validation was strengthened while training: reset replay now occurs
after a step with a revealed target, checks prediction/bound shapes, finiteness
and ordering on all three steps, and demands bitwise replay of both points and
bounds. The unchanged E029 extracted submission passes on evaluator-compatible
CPU; three new synthetic regression tests pass. This modifies only the local
packaging gate, not E029's wrapper or preserved archive.

## Seed-0 results

All cells evaluated 1,031 main-fold or 821 complement-fold windows, with no
training overlap within a fold. Both folds have been used during earlier
development; these are not untouched final tests.

| Fold | Stride | Updates | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 20 | 600 | 94.116256 | 77.249801 | 95.629370 | 86.677886 | 36.397565 |
| v1 | 20 | 1800 | 94.447761 | 77.114139 | 95.798160 | 86.983092 | 37.542918 |
| v1 | 1 | 600 | 94.134092 | 77.258355 | 95.647524 | 86.883400 | 36.469821 |
| v1 | 1 | 1800 | 94.510885 | 77.059214 | 95.831988 | 86.738606 | 37.716577 |
| complement | 20 | 600 | 95.904282 | 76.585232 | 96.912627 | 86.568072 | 46.511037 |
| complement | 20 | 1800 | 96.120619 | 76.470977 | 97.079123 | 86.792060 | 47.262038 |
| complement | 1 | 600 | 95.910614 | 76.583994 | 96.920245 | 86.966539 | 46.537807 |
| complement | 1 | 1800 | 96.157316 | 76.292152 | 97.097166 | 86.820362 | 47.313019 |

The locked rule selects **stride20/1800**: mean gains +0.273921 Rel-L2,
−0.124958 TKE, +0.167643 MVPE, +0.948177 SPS vs stride20/600; both folds
meet the preservation limits. Dense/1800 has slightly higher Rel-L2/SPS but
fails the complement TKE limit (−0.293079 versus allowed −0.25). Dense/600
adds only +0.012084 Rel-L2 and +0.049513 SPS on average and fails gain gates.
No thresholds were changed after these results. Time is descriptive, not a
selection criterion between identical deployed graphs.

Training used RTX 5050, Torch 2.7.1+cu128, float32, all 50,358,342 parameters.
Seed-0 training took 104.75–105.41 seconds for 600 updates and 313.22–323.86
seconds for 1,800; peak CUDA allocation was 2,647,842,304 bytes in every cell.
Packed size stays approximately 201.4 MB. The retained selected/control
checkpoint identities are:

| Fold / updates / seed | Packed checkpoint SHA-256 |
|---|---|
| v1 / 600 / 0 | `93c6931baaf7b67c9a779fe528969d84738ce42886b1d53311e641ae0592fd65` |
| v1 / 1800 / 0 | `b6c9a4cfef7960e1827f48c180b8e32bd55c9b1c1a4908ccc3decf64e0725318` |
| complement / 600 / 0 | `da3395dd96641e6b597fc6c5fd13320908a50296ad93bc4a97cecd9c57cd531d` |
| complement / 1800 / 0 | `d452b2db3f2d5bd0cdffc84c3709345f52f4ba72765837f6562cbf1289150909` |
| v1 / 600 / 1 | `28b5f7fdb3a3a7fb2d5e16a4984cc7e921a46a35387af03413a92d40a2823802` |
| v1 / 1800 / 1 | `c28d8f7b171fa40765a92daaac126f2b7c49739bc07bafbb687fa5a9721aff19` |
| complement / 600 / 1 | `c1d670c69a8a1ab2de46556f4cc0dfd0452b8ec6b8e4ad8c756966fc2d40749d` |
| complement / 1800 / 1 | `5b70187d2be0247fa4e2669ca4ddbf6a6e230922688003b5282cb7b847991137` |

Seed-1 confirmation command (each runner covers both folds):

```bash
../.venv-gpu/bin/python scripts/run_sampling_study.py --initialization checkpoint --seed 1 --stride 20 --updates 600
../.venv-gpu/bin/python scripts/run_sampling_study.py --initialization checkpoint --seed 1 --stride 20 --updates 1800
../.venv/bin/python scripts/summarize_sampling_study.py --initialization checkpoint --seed 1 --candidate stride20_u1800
```

## Seed-1 confirmation and decision

| Fold | Updates | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 600 | 94.090706 | 77.276436 | 95.622499 | 86.940436 | 36.375816 |
| v1 | 1800 | 94.430161 | 77.138303 | 95.788491 | 86.996259 | 37.481896 |
| complement | 600 | 95.910436 | 76.616308 | 96.954406 | 86.971774 | 46.564522 |
| complement | 1800 | 96.122652 | 76.485781 | 97.080561 | 86.632879 | 47.272164 |

Seed 1 passes the same locked gate: mean +0.275835 Rel-L2, −0.134330 TKE,
+0.146073 MVPE, +0.906861 SPS. Neither fold breaches a preservation limit.
Across both folds and seeds, gains are **+0.274878 Rel-L2, −0.129644 TKE,
+0.156858 MVPE, +0.927519 SPS**. This is a reproducible field/interval-score
improvement with a small measured TKE tradeoff, not an across-metric dominance.

The selected seed-0 main model also improves over historical E020+emaV by
approximately +0.354 Rel-L2, +0.134 MVPE, +1.093 SPS, with −0.204 TKE.
The selected complement model improves over the E030 anchor by +0.176 Rel-L2,
+0.108 MVPE, +0.626 SPS, with −0.155 TKE. Thus promotion is not solely an
artifact of a weaker freshly trained comparator.

Accept simulation initialization, trajectory-balanced stride20 sampling and
1,800 updates for E033's all-valid-real refit, preserving the E029 wrapper and
bounds. Do not promote dense offsets. E033's full-data model must not be scored
on released holdouts as if they remained unseen. Official >80 is still unproven.

Reports: `artifacts/e032/summary.json` and `confirmation_seed1.json`.
The full evaluator-compatible suite passes **108 tests** after all local
verification additions. Packaging and official scoring belong to E033.
