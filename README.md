# RealPDE Track 2 — Streaming Flow Adaptation

Research code and experiment records for our NeurIPS 2026 RealPDE Track 2
participation. Forecast flow in an ordered stream, adapting only from targets
revealed for previous predictions.

**Best reported development-leaderboard final_score: 77.615784 (E033).**
This is a user-reported official score, not a final private-test ranking.

## Approach

- Simulation-initialized Fourier neural operator with temporal-mean and
  zero-mean-fluctuation output heads.
- Trajectory-balanced stride-20 real-data training, 1800 updates.
- Online mean-bias correction and per-pixel variance EMA from revealed targets.
- Bounded, smoothed fluctuation rescaling, calibrated uncertainty intervals,
  and complete state reset at every trajectory boundary.
- Explicit verification of checkpoint initialization and deployment parity.

Read [METHOD.md](METHOD.md), [RESULTS.md](RESULTS.md), and
[REPRODUCING.md](REPRODUCING.md). Experiment IDs are track-local.

## What is available

Participant source, configurations, tests, historical result notes/hashes, and pinned
[environment snapshots](environment-locks/README.md). The full competition data,
trained checkpoints and generated submission archives were deleted.
Reproduction requires reacquiring the authorized release and retraining;
no fresh end-to-end reproduction was run after cleanup.

Organizer code, copied competition pages, example assets, and the original Git
history are intentionally omitted. Obtain required dependencies using
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).

## Start here

```bash
# No Python packages, GPU, or competition data required:
python3 scripts/check_archive.py
python3 scripts/publication_audit.py
```

Follow [REPRODUCING.md](REPRODUCING.md) for restored-environment tests and training.
The sibling project is `realpde-track1` (sim-to-real forecasting); clone this
repository locally as `RealPDE_T2` for the historical workspace layout.

## Repository map

- `src/realpde_t2/`, `scripts/`, `submission/`: models, streaming logic and CLIs.
- `configs/`, `tests/`: reproducible controls and contract tests.
- [LEDGER.md](LEDGER.md), `docs/E*_RESULTS.md`: experiment history and failures.
- [TRACK2_REFERENCE.md](TRACK2_REFERENCE.md): dated organizer protocol notes.
- [AGENT.md](AGENT.md): development conventions.
- [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md): release checks and open issues.

## Licensing and credit

MIT for original contributions only. Upstream/derived code and examples retain
their own terms, including non-commercial restrictions; the entire repository
is **not** MIT. Read [LICENSING.md](LICENSING.md) and
[ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md). This is a source-only public-release snapshot; see
[PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) for checks and limitations.
