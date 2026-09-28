# E024 — Submission candidate: full-release dual-head FNO + emaV streaming adaptation

Date: 2026-08-22. Status: **Complete — platform bundle-fetch failure; superseded by E029.**

## What shipped

E024 replaces E022's static `fluct_scale = 1.50` with **emaV** (the E023-accepted
mechanism): at every step after the first reveal, the predictor maintains a
per-pixel EMA (γ=0.9) of the *revealed previous target's* temporal-variance map
and rescales the model's centered fluctuation by the smoothed
(σ=0.5, 3×3 reflect kernel), clipped ([0.2, 5.0]) ratio
`sqrt(E / V_model)`. The first window of each trajectory uses identity
scaling. Mean-bias EMA (γ=0.9) and frozen SPS bounds `(0.02, 0.14)` are
unchanged from E022. Weights are the unchanged E021 full-release packed
dual-head FNO.

## Implementation and verification chain

- `src/realpde_t2/streaming_state.py`: `fluct_mode="emav"` added to
  `StreamingStatePredictor` (fixed mode bitwise-preserved); shared
  `smooth_variance_ratio` helper. Used by local evaluation.
- `submission/e024/submission.py`: self-contained wrapper
  (`DualHeadEmaVPredictor`), no imports outside torch + bundled kit files.
- `scripts/evaluate_emav_stream.py` + `configs/eval/e024_emav_e020.json`:
  official-protocol stream evaluation of the *deployed* class with
  leakage-safe E020 weights.
- `tests/test_emav.py`: 7 focused tests (kernel reference math, first-step
  identity, EMA math, reset clearing, amplification direction, clip
  finiteness, bitwise replay after reset). Full suite: 84 tests pass in
  `.venv` (evaluator environment).
- `configs/experiments/e024_submission.json` +
  `scripts/package_submission.py`: archive and contract gates.

## Deployed validation (leakage-safe E020 weights, `real_regime_v1` val, 1,031 windows)

| Run | Rel-L2 | TKE | MVPE | SPS | Coverage | Mean step |
|---|---:|---:|---:|---:|---:|---:|
| E023 offline replay | 94.094 | 77.318 | 95.665 | 36.450 | 71.9% | — |
| **E024 deployed code** | **94.094** | **77.318** | **95.665** | **36.450** | 71.9% | 38.70 ms |
| fixed-mode same env (sanity) | 94.658 | 74.111 | 95.665 | 36.900 | 73.1% | 38.01 ms |

The deployed class reproduces the E023 replay to <0.001 on every component,
and the fixed path reproduces stored E020 exactly (my `streaming_state.py`
edits are behavior-preserving). emaV adds **+0.68 ms/step** over fixed mode in
the same environment (Time −0.14 local).

Timing caveat: this GPU currently runs the FNO forward ~2.4× slower than
during the Aug-15/21 measurements (25.9 ms vs 13–14 ms/forward, sustained
load, clocks at max) — an environment-level change that affects absolute Time
only; the emaV-vs-fixed *delta* is the decision-relevant number. Official
E022 Time was 89.02 (≈11.1 ms/step on Codabench hardware); +0.68 ms projects
to ≈ −0.3 Time there.

## Archive gates (`artifacts/e024/report.json`)

- Archive `e024_candidate.zip`: 185,748,360 B, SHA-256 `708b9a74…`;
  extracted 201,432,727 B < 256 MB.
- Host `local_eval` on kit example data: passed.
- Bitwise reset replay (first step vs post-reset step, bounds included):
  passed.
- Pinned official container (`w3nhao/realpde-track2@sha256:f07d1d68…`),
  offline/read-only, `local_eval`: passed in 7.5 s.
- Validation proxy: intentionally skipped (E021 weights saw all 81 valid real
  trajectories; the leakage-safe proxy above used E020 weights instead).

## Official attempt and decision

Codabench submission `897439` failed before participant execution. Its live
status tooltip identifies a transient platform storage error while downloading
the organizer `ingestion_program` bundle, explicitly says the participant
submission is not at fault, and requests resubmission. The attempt returned its
daily and total quota.

D039 supersedes D035 operationally: submit E029, whose arithmetic smoothing
and guarded emaV branch preserve E024 outputs to sub-float32-roundoff scale.

## Next

After an additional server-side failure, the unchanged E029 archive eventually
scored successfully at 76.760173 and superseded E024 as the official baseline.
E025–E027 were completed as negative follow-ups; their conclusions are recorded
in their corresponding result notes.
