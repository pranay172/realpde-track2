# E031 — Controlled initialization comparison

Status: complete — simulation initialization accepted, 2026-09-21.
Baseline official submission remains E029.

## Registered design

Eight new training runs: random or official simulation checkpoint initialization,
crossed with `real_regime_v1` and `real_regime_complement_v1`, seeds 0 and 1.
Every run uses the same 600-update dual-head FNO recipe, batch size 4, Adam
3e-4 with cosine decay, physical relative u/v MSE, and stride-20 windows.
Both seeds were fixed before the first evaluation. All valid loaded source
weights are checked for exact tensor equality after conversion, including both
projection heads and BatchNorm buffers. The first random run began before this
extra equality assertion was added; no external weights were loaded for it.

Evaluation uses the packed checkpoint with E029 emaV and bounds (0.02, 0.14).
Only held-out real trajectories are scored; simulation initialization contains
released simulation regimes and is not a strictly unseen-simulation-regime test.
Both real folds have influenced previous development and are not untouched tests.

Simulation initialization passes only if the mean Rel-L2 gain over both folds
and seeds is at least 0.15, and neither fold's seed-average drops exceed 0.25
TKE, 0.10 MVPE, or 0.25 SPS. Otherwise keep random initialization. Timing is
reported but excluded from selection between identical inference graphs.

## Reproduction and evidence

Run `../.venv-gpu/bin/python scripts/run_initialization_study.py`, then
`../.venv/bin/python scripts/summarize_initialization_study.py`.
Controls: `configs/experiments/e031_initialization_study.json`.
Registered study SHA-256:
`634c575ca46b3e5ea6127136d1931c0499e74152edd35c98e171bff27b098f22`.
Each directory under `artifacts/e031/` retains resolved train/eval configs,
logs, fp32/fp16 checkpoints, full loss history, seed, data membership, loading
checks, memory/runtime, checkpoint hashes, and per-trajectory scores.
Implementation registration commit: `cb9bcbe`; initial runs started with its
working-tree changes before that commit, which their recorded HEAD reflects.

Trajectory bootstrap intervals use 5,000 paired resamples with fixed seed 0.
They summarize equal-weight trajectory score differences within a fold/seed;
they are not confidence intervals for the official hidden score or substitutes
for seed replication. Primary gates use the official aggregate component scores.

## Results

| Fold | Initialization | Rel-L2 | TKE | MVPE | SPS |
|---|---|---:|---:|---:|---:|
| v1 | random | 93.943576 | 75.138149 | 95.319939 | 35.398427 |
| v1 | simulation | 94.134797 | 77.252565 | 95.654592 | 36.491660 |
| complement | random | 95.693439 | 74.937046 | 96.622143 | 45.294639 |
| complement | simulation | 95.944491 | 76.625625 | 96.971424 | 46.636387 |

Seed 1:

| Fold | Initialization | Rel-L2 | TKE | MVPE | SPS |
|---|---|---:|---:|---:|---:|
| v1 | random | 93.918056 | 74.686810 | 95.317674 | 35.195366 |
| v1 | simulation | 94.085511 | 77.268297 | 95.622406 | 36.363481 |
| complement | random | 95.614675 | 74.645995 | 96.523502 | 44.954986 |
| complement | simulation | 95.872480 | 76.586559 | 96.902583 | 46.376065 |

Across both folds and seeds, simulation initialization gains +0.216883 Rel-L2,
+2.081262 TKE, +0.341937 MVPE, and +1.256044 SPS. All 16 paired quality deltas
are positive and all their equal-trajectory bootstrap 95% intervals exclude zero.
The locked gates pass. Select the official simulation initializer for E032;
this is a training-recipe decision, not a claim that the hidden score exceeds 80.

Training took 105–108 seconds per run on RTX 5050 with Torch 2.7.1+cu128;
all 50,358,342 parameters trained in float32, peak CUDA allocation 2,647,842,304
bytes; packed checkpoints are about 201.4 MB. Local Time is approximately 86.7–87.0
across the identical architectures and is not used to select initialization.
Full paired statistics are retained in `artifacts/e031/summary.json`.

Packed checkpoint SHA-256 values (rows ordered by fold, seed, initialization):

| Fold / seed / initialization | SHA-256 |
|---|---|
| v1 / 0 / random | `4c064ea2edea6c0b26875c087aaa0f31b97d251e84872211f5cef09a1dc3fa36` |
| v1 / 0 / simulation | `87a31c959d4eb080cd4458890fdfb6efff86d8f2ab29a35a2dc00d633a50174c` |
| v1 / 1 / random | `16d8e692101004fdfa5b3ec2845029161f3c8c6f0c95a39e80ee0051f0edc8dd` |
| v1 / 1 / simulation | `43216e6d6a6be369dc19445fc06e63fa8748413bf841ded5f3a1654b8d1e678d` |
| complement / 0 / random | `e76ba490b42aff5deaa30db3ab78dbe15f3793a3419e4c80ef8989593d583e71` |
| complement / 0 / simulation | `abe3c93ee504aaa467b1fd9969715df506ed139fc28b2f814b34e4029d81a333` |
| complement / 1 / random | `81c9e36472c5f39feae877647cbf715bb65ebd5ee86d08e84b29003a4d6735a0` |
| complement / 1 / simulation | `cca0f49b260dd9d9599a4d6b18c5b60f4a040793f15d950f9d4f1ba2b1e308ad` |

The simulation/complement quality result reproduces E030 exactly. The random/v1
run does not reproduce historical E020. A subsequent checkpoint audit found
that E020's original verified packed checkpoint has all four BatchNorm counters
at 5,600, matching simulation's 5,000 plus 600 updates, whereas E021 and E031's
fresh random run have counters 600. Thus E030's blanket random-init label was
not justified for E020. Counters support inherited training state but do not
fully reconstruct the historical execution. Initialization effects here are
estimated entirely from fresh matched runs, and historical E020/E029 remain
separate quality anchors for eventual promotion. See E020's corrected note.

Verification: 97 evaluator-compatible tests passed, including trajectory-boundary
aggregation; compilation and whitespace checks passed. No submission code changed.
