# Track 2 Agent Onboarding and Development Guide

This repository is the primary workspace for RealPDE Competition Track 2:
Long-Term Test-Time Adaptation (LTTTA). This document is the stable operating
contract for agents and contributors. Keep experiment-specific state, scores,
and candidate details in `LEDGER.md` and `docs/`, not here.

## Public snapshot boundary

Read `EXTERNAL_DEPENDENCIES.md` before running models. The organizer kit,
example data, copied pages, host-specific handoff, and private Git history are
not shipped. Do not re-add them or claim that restoring a kit grants rights to
redistribute it. Historical experiment notes retain old artifact paths and
commit hashes as provenance, not promises that those assets are present.

## Start here

Before changing code or proposing work, read:

1. `LEDGER.md` for the current baseline, active work, prior outcomes, and
   decisions.
2. `TRACK2_REFERENCE.md` for the task, protocol, rules, metrics, limits, and
   current competition information.
3. `README.md` for repository entry points and directory roles.
4. `environment-locks/README.md` for the shared environments, GPU/Docker behavior,
   and verified smoke tests.
5. The relevant file under `docs/` and matching configuration under `configs/`
   before revisiting an existing approach.
6. The organizer starting kit, especially
   `realpde_t2_starting_kit_v6/docs/interface.md`, `scoring.py`, `local_eval.py`,
   and `submission_template.py`, before changing submission code.

When local notes conflict with a live organizer page, treat the organizer page
as authoritative. Update `TRACK2_REFERENCE.md` and the ledger; link to official pages rather than
copying their text into the public repository.

## Core correctness constraints

- Process evaluation as an ordered, batch-one stream. Never shuffle windows or
  combine state across trajectories.
- A revealed previous target may update state only for later predictions. Never
  use the current target to predict the current window.
- Cache the input paired with each later-revealed target; test this alignment
  explicitly rather than relying on loop position.
- Reset every mutable object at a trajectory boundary: weights, optimizer and
  scheduler state, running statistics, controllers, calibration state, history,
  and cached tensors.
- Keep scored channel order, normalization, shapes, and output conventions
  identical to the organizer interface. Assert them at module boundaries.
- Treat optional uncertainty bounds as part of the contract: return both or
  neither for an entire run, with exact shape, finite values, and `lower <= upper`.
- Use only competition-permitted data and weights. Preserve data provenance and
  the mandatory invalid-file exclusion documented in `TRACK2_REFERENCE.md`.
- Keep evaluation offline except for organizer-authorized services. Never read
  hidden targets outside the values explicitly revealed by the streaming API.
- Respect the submission size and total-runtime limits. Loading, reset, and
  adaptation behavior must be tested under evaluator-compatible conditions.

## Repository layout

```text
RealPDE_T2/
├── AGENT.md                 stable development and onboarding contract
├── AGENTS.md                pointer automatically read by agents
├── LEDGER.md                compact live state and experiment index
├── README.md                workspace map and common entry points
├── TRACK2_REFERENCE.md      competition facts, rules, and protocol
├── configs/
│   ├── data/                release manifests and provenance
│   ├── splits/              versioned trajectory-level splits
│   ├── eval/                evaluation and comparison configurations
│   └── experiments/         one reproducible configuration per experiment
├── docs/                    durable audits and experiment result notes
├── scripts/                 thin train, evaluate, diagnose, and package CLIs
├── src/realpde_t2/          reusable tested implementation
├── tests/                   deterministic unit and stream-contract tests
├── submission/              minimal versioned evaluator-facing wrappers
├── artifacts/               ignored checkpoints, reports, caches, and archives
└── realpde_t2_starting_kit_v6/  external dependency, ignored and not shipped
```

Put reusable logic in `src/`, configuration in `configs/`, and orchestration in
`scripts/`. Store a concise durable result note in `docs/`; keep large generated
outputs under ignored `artifacts/`. Do not edit organizer files unless the task
specifically requires it and the reason is documented.

Large shared datasets and official checkpoints live outside this repository.
Never commit environments, full datasets, HDF5 outputs, weights, submission
archives, predictions, caches, raw logs, or organizer example assets.

## Artifact retention and cleanup

Treat ignored artifacts as disposable unless a tracked note identifies them as
an active dependency:

- Preserve an exact active submission archive, its extracted validation copy,
  and the hashes needed to identify it until its official result is settled.
- For an accepted model foundation, retain the smallest loadable checkpoint and
  decisive result report; remove redundant `latest`/full-precision copies once
  no active follow-up consumes them.
- For a rejected experiment, keep measured results, hashes, configuration, and
  logs needed for diagnosis. Large checkpoints may be removed after the result
  note is complete and no queued experiment depends on them.
- Delete scratch arrays, prediction caches, extracted archives, and transient
  logs when the approach family closes or the generating script can reproduce
  them cheaply.
- Before material deletion, list exact consumers and retained replacements.
  Never use broad globs or remove the entire artifact root.

Python bytecode caches and other tool-generated caches may be cleared during
routine cleanup. Record material artifact deletions and approximate reclaimed
space in the ledger.

## Environment selection

For reconstruction after cleanup, use the tracked `environment-locks/README.md`.
Install the environments in the parent project root, shared
between tracks; dependency snapshots do not restore data or model artifacts.

The parent workspace owns two shared Python environments:

- `../.venv`: evaluator-compatible Python/Torch stack for CPU validation.
- `../.venv-gpu`: local development and training stack for the RTX GPU.

Run commands from the repository or parent workspace with the environment's
Python explicitly, for example:

```bash
../.venv/bin/python -m unittest discover -s tests -v
../.venv-gpu/bin/python gpu_smoke_test.py
```

The local GPU requires the newer development environment; that does not prove
evaluator compatibility. Final submission-facing changes must also pass in the
evaluator-compatible environment and, when material, the pinned official
container. See `environment-locks/README.md` for exact versions and the current
container command.

Codex's normal sandbox hides the GPU and Docker socket. Request narrow host-level
approval for CUDA or Docker commands. Do not change device permissions, Docker
groups, drivers, or host security settings as a workaround.

## Development process

### 1. Establish a falsifiable question

- Read the current ledger, relevant result notes, and configuration first.
- Reserve the next monotonic `E###` before a meaningful experiment begins.
- State one hypothesis, one comparator, and the decision threshold.
- Fix data provenance, whole-trajectory split, seed, metrics, resource budget,
  and stopping rule before looking at held-out results.
- Explain which future decision the result will change. Avoid broad sweeps with
  no promotion or rejection boundary.

### 2. Implement the smallest reproducible change

- Prefer a configuration change or isolated component over a coupled rewrite.
- Use repository-relative or configured paths, never machine-specific paths.
- Seed Python, NumPy, and Torch; make device, dtype, and precision explicit.
- Keep loading, filtering, normalization, window construction, ordering, and
  split selection independently testable.
- Separate forecasting, adaptation, state management, uncertainty, and scoring
  so ablations share the same data and evaluator path.
- Bound adaptation steps, memory, controller history, and per-trajectory compute
  in configuration.
- Add assertions for target alignment, tensor layout, finite values, and reset
  isolation close to the failure boundary.

### 3. Validate in increasing cost order

1. Parse/import checks and focused unit tests.
2. Tiny CPU stream with shape, finiteness, alignment, and reset assertions.
3. Deterministic reset replay from fresh state.
4. Checkpoint load/round-trip and a representative GPU stream.
5. Leakage-safe held-out trajectories with every official component metric and
   synchronized timing.
6. Evaluator-compatible environment run.
7. Pinned official-container replay for submission-facing changes.
8. Exact archive validation: expected root files, offline imports, extracted
   size, output/bounds contract, multi-trajectory reset, and runtime margin.

Exercise multiple trajectories and boundary conditions. A passing single-window
smoke is necessary but insufficient for a stateful submission. Test failure and
fallback paths deliberately when they are part of the wrapper.

### 4. Record and decide

After every substantive conversation or attempted approach:

- update the ledger's date, current state, and next action;
- add or update one compact experiment-index row;
- put full setup, measurements, limitations, and artifact references in the
  corresponding `docs/E###_RESULTS.md` file;
- close or reprioritize ideas and record durable decisions;
- preserve negative results so later agents do not repeat them.

The ledger is an index, not a raw lab notebook. Do not copy full result notes or
Git history into it. If a table cell needs a paragraph, move that detail to
`docs/` and link it.

### 5. Commit coherently

- Inspect `git status` before and after work; preserve unrelated user changes.
- Keep commits purpose-specific and include the implementation, configuration,
  tests, and result note needed to reproduce the change.
- Use imperative commit messages.
- Run `git diff --check` and inspect the staged diff before committing.
- Never stage secrets, data, weights, archives, caches, or raw run outputs.

## Experiment reporting minimum

A completed `E###` must make these reconstructable in its config and result note:

- hypothesis, comparator, gate, and conclusion;
- commit, configuration, seed, and code/checkpoint hashes when material;
- exact data provenance, exclusion policy, and trajectory-level split;
- environment, device, precision, trainable parameters, runtime, and peak memory;
- all available official component scores and any Codabench final score;
- adaptation/reset behavior and uncertainty method;
- submission size, timing method, compatibility checks, and archive hash;
- limitations, failed gates, and next action.

Prefer compact tables and measured deltas against a frozen comparator. Reference
ignored artifact paths for local reproducibility, but keep the decisive numbers
and hashes in tracked documentation.

## Coding preferences

- Target Python 3.10 and evaluator-compatible APIs.
- Favor small composable functions, explicit configuration, `pathlib`, useful
  type hints, and concise docstrings for non-obvious behavior.
- Name layout transformations and assert shapes; silent channel permutation and
  previous/current-window mistakes are high-cost failures.
- Fail early on missing or duplicate trajectories, non-finite tensors, stale
  caches, unexpected schemas, and incomplete bounds.
- Keep submission imports minimal and offline-safe. Vendor only suitable code
  whose license and size are understood.
- Profile before optimizing. Report accuracy, uncertainty, stability, reset
  cost, mean step time, and total runtime together.
- Preserve a deterministic no-adaptation path and prefer reversible changes.

## Definition of done

A change is ready to hand off when:

- its question, comparator, gate, and result are clear;
- focused tests and a representative multi-trajectory stream pass;
- target alignment and complete reset isolation are verified;
- evaluator compatibility is checked when submission code changes;
- no forbidden or generated files are staged;
- the result note and compact ledger index agree;
- the next action is explicit and the worktree is clean.
