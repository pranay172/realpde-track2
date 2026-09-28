# RealPDE Track 2 reference

Last reviewed: **2026-09-21 (IST)** against the live competition overview and
the locally pinned Codabench/FAQ snapshots. Those snapshots are intentionally
omitted from this public release; see
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).

## Canonical links

- Competition overview: <https://realpdecompetition.github.io/>
- Track 2 Codabench: <https://www.codabench.org/competitions/17385/>
- Track 2 forum: <https://www.codabench.org/forums/17083/>
- Track 2 registration form (required): <https://forms.gle/qmMYaK5u9r86rPGKA>
- Training release: <https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data>
- Organizer contact: <realpde-competition@googlegroups.com>
- Original page snapshots were retained in the private archive, not this export.

For implementation details, treat the current Codabench Evaluation,
Submission, Rules, and FAQ pages as authoritative. The public overview is a
summary. If a live page disagrees with these dated participant notes, verify
the live source and update this reference without copying whole pages. Use
starting kit **v6** for the current Development Phase; older Track 2 scorers
and local evaluators are obsolete. The Files tab (starting kit download) is
visible only after sign-in and an approved participation request.

## Registration and administration

- **Completed:** the participant confirmed submission of both Track 1 and Track
  2 registration forms on 2026-08-14.
- Registration is closed. Teams that registered in time may continue to submit.
- One shared Codabench account is allowed per team. Multiple accounts can cause
  disqualification.
- Maximum team size: 3.
- Anyone who has had access to the hidden evaluation data, the scoring
  configuration, or the private test set may not compete or share a prize.
  Eligibility is about access, not affiliation.

## What Track 2 asks us to solve

Track 2 is Long-Term Test-Time Adaptation (LTTTA) for a real PIV flow stream
around a NACA4418 airfoil. A model repeatedly forecasts the next 20 frames and
then, one step later, receives the ground truth for its previous forecast. It
may use that revealed previous pair to adapt online before predicting the
current window.

The core deployment problem is not merely one-window forecasting. It is to
remain accurate, physically consistent, safe/calibrated, and fast while errors
and distribution shifts accumulate over a long ordered trajectory.

Permitted adaptation includes gradient updates, calibration, memory/retrieval,
expert/module selection, bounded controllers, and optional organizer-gateway
LLM decisions. All online computation counts toward runtime.

## Exact streaming protocol

- Strict trajectory order; batch size is always 1.
- Native real fields are `64 x 128`; evaluation uses 2x spatial subsampling to
  `32 x 64`.
- Channels are `[u, v, p]`. Real PIV measures `u` and `v`; `p` is zero-filled
  and is not scored.
- Every model input and prediction has shape `(1, 20, 32, 64, 3)`.
- Each step is 20 input frames to 20 target frames with stride 20.
- Window `k`'s target is window `k+1`'s input.
- The evaluator normalizes/denormalizes with the official `train_real`
  statistics. Participant code works in normalized space.
- `reset_ttt_state()` is called at each trajectory boundary. It must restore
  initial weights and clear all trajectory-specific caches/adaptation state.
- At step `t`, `prev_target_norm` is the target for step `t-1`, or `None` at a
  trajectory's first step. The current target is never available to the current
  prediction.

Canonical `ttt_step` order:

1. If a previous target is available, adapt on the cached previous input and
   the revealed previous target.
2. Predict from the current input without using its target.
3. Cache the current input/prediction for the next step.

Pairing the revealed previous target with the current input is a protocol bug.
Using the current step's ground truth by any means is a rule violation.

## Submission interface

The zip must contain `submission.py` at its archive root and define:

```python
def get_ttt_model(submission_dir: str, device: str):
    ...
```

The returned object is duck-typed and must provide:

```python
def reset_ttt_state(self) -> None:
    ...

def ttt_step(self, input_norm, prev_target_norm=None):
    # Return pred_norm and info.
    return pred_norm, {"adapt_loss": adapt_loss}
```

Contract details:

- `pred_norm` must have shape `(1, 20, 32, 64, 3)`.
- `info` must be a dictionary containing `adapt_loss`, which is a Python float
  or `None` when no adaptation occurred.
- Other `info` keys are ignored except optional `lower` and `upper` interval
  bounds.
- Use relative paths rooted at `submission_dir` for checkpoints/support files.
- Extracted archive size, including weights, must be **under 256 MB**.

### Optional uncertainty bounds

`ttt_step` may return `info["lower"]` and `info["upper"]` in the same normalized
space and exact shape as `pred_norm`.

- Return both keys or neither.
- Bounds must be finite and satisfy `lower <= upper` elementwise.
- The choice is run-wide: bounds must be present on every step or on no step.
  Starting or stopping bounds partway through causes ingestion failure.
- Without custom bounds, the scorer uses the physical-space default
  `pred ± 0.05 * abs(pred)`.

## Timing and platform limits

- Every `ttt_step` call is timed, including adaptation, controller logic, model
  forward, and LLM latency.
- Every `reset_ttt_state` call is timed and charged to the following step.
- Model construction and checkpoint loading are excluded from the per-step Time
  metric, but are included in the run's platform wall-clock limit.
- Warm-up and Development runs have a **10-minute container-execution limit**.
  Data download time is excluded. A timeout is Failed/no score. The FAQ says
  this wall limit may be recalibrated on the official baseline; any change
  will be announced there.
- The Time reference is `t_numerical = 0.72896 s` per step:

```text
r = mean_step_seconds / 0.72896
time_score = 100 / (1 + sqrt(r))
```

## Evaluation

The leaderboard uses a hidden `final_score` in `[0, 100]`. Its combination is
not published. The public leaderboard shows only `final_score`; a team's own
detailed submission result shows the five component scores.

All five component scores are clipped to `[0, 100]` and higher is better:

1. `rel_l2_score`: relative L2 data fidelity over `u, v`.
2. `tke_score`: relative L2 error of
   `0.5 * (var_t(u) + var_t(v))` over the 20-frame window.
3. `mvpe_score`: relative L2 error of time-mean `u, v` at fixed wake probes
   (4 streamwise stations and up to 9 rows).
4. `time_score`: mean timed per-step efficiency.
5. `sps_score`: safe prediction score combining accuracy, coverage, and interval
   tightness.

Official Rel-L2 per window is
`||pred - target||_2 / max(||target||_2, 1e-8)` over flattened `u, v`.
TKE and MVPE use the same relative-L2 form on their reduced fields.

For each of Rel-L2, TKE, and MVPE:

```text
component_score = 100 / (1 + 0.5 * mean_window_error)
```

### SPS summary

- Frozen width normalizer: `sigma_global = 0.0563870` (the more precise scorer
  constant is `0.0563870259`).
- For branch error `e`, `pm = e / (0.5 + e)`.
- A covered element receives `(1 - pm) * exp(-(upper-lower)/sigma_global)`;
  an uncovered element receives 0.
- Targets outside the measured PIV field or inside the airfoil body are excluded
  from SPS averaging.
- Branch weights are DM/Rel-L2 `0.5`, TKE `0.3`, MVPE `0.2`.
- `sps_score = 100 * weighted_branch_average`.
- An uncovered or confidently wrong element contributes 0, never a negative
  value, so a tight interval around a bad prediction cannot buy score.

The August 5 scorer update replaced the old logistic SPS mapping with this
linear 0-100 mapping. The current Development leaderboard was restarted empty,
so old archived-phase scores are not comparable.

### Invalid-output behavior

The official pages distinguish controlled zero scores from ingestion failure:

- Non-finite predictions, non-finite/reversed bounds, or scorer-level invalid
  output produce zero on all five subscores.
- Wrong `pred_norm` shape, bound-shape mismatch, returning only one bound, or
  changing bound presence mid-run can stop ingestion and yield no score.
- Guard every output with shape, finite-value, and interval-order checks before
  submission.

## Data

Physical setup:

- NACA4418 airfoil cross-sectional flow.
- Real: time-resolved PIV in a circulating water tunnel.
- Simulation: matched 3D CFD under the same geometry/operating conditions.
- Angles of attack: `0, 5, 10, 15, 20` degrees.
- Reynolds-number range: `2968` to `27975`.
- The overview describes 100 paired trajectories overall. The audited training
  release contains 100 simulation trajectories but only 82 real trajectories.

Prefer the Hugging Face copy of the release. The Google Drive mirror has a
per-file anonymous download quota that a shared link hits when many people
fetch it at once. Both hosts hold the same files.

Release contents:

- `train_sim.tar.gz`: simulated trajectories for pretraining.
- `train_real.tar.gz`: real PIV trajectories for finetuning.
- `example_data/3750_0.h5`: shape/data-layout example.
- `baseline_checkpoints/sim_pretrain/`: simulation-pretrained CNO/FNO/Transolver.
- `baseline_checkpoints/sim_real_ft/`: real-finetuned versions; normally the
  stronger Track 2 starting point.

Verified release layout: both extracted training sets store datasets at the HDF5
root. Real files contain `aoa,re,t,u,v,x,y`; simulation files additionally
contain `p`. See `docs/DATA_AUDIT.md` for dtypes, shapes, grid gaps, and anomalies.

### Mandatory exclusion

**Do not train on `train_real/7575_0.h5`.** It duplicates the measurement data
from `6300_0.h5` and is not a true Re=7575, AoA=0 case. Other AoAs for Re=7575
and the original `6300_0.h5` are valid. The bad file remains in the archive, so
our future dataset loader must explicitly skip it.

## Training-data and resource rules

- Training samples must trace back to the official released data.
- Official baseline weights and any checkpoint trained only from the release
  are permitted.
- Standard transformations/augmentation of released samples are permitted.
  Preserve clear provenance and reproducible augmentation code.
- No extra simulations/self-generated training data, external datasets, or
  pretrained models trained on other data.
- Shortlisted methods are retrained from scratch by organizers using only the
  release, so the full training pipeline must reproduce from it. The FAQ is
  explicit: the finalist package is **training code only** — no extra
  simulations, no code that runs new simulations, no model weights, and no
  external pretrained models. Augmentation must ship with that training code.
- Open-source software may be used during development.
- Do not exploit implementation bugs to compromise evaluation integrity.

There is a stale wording conflict: the August 5 pinned forum post says
"augmenting the release into extra training data" is disallowed, while the
later August 9 FAQ, current Rules-linked clarification, and dataset card
explicitly allow standard traceable augmentation. Use the later FAQ as the
current authority and keep augmentation as reproducible transforms of released
samples rather than unrelated generated data.

## Evaluation environment

Pinned image:

```text
w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972
```

- Python 3.10, PyTorch 2.2.2, CUDA 12.1, plus `torchvision` / `torchaudio`
  2.2.2.
- Offline; nothing is installed during evaluation.
- Common scientific packages are present, including NumPy 1.26, SciPy, pandas,
  matplotlib, h5py, einops, scikit-learn, tensorboard, safetensors, Pillow,
  SymPy, NetworkX, PyYAML, requests, tqdm, and the OpenAI client. Exact
  resolved versions are in `/opt/image_manifest.txt` inside the image.
- Vendor any additional pure-Python package inside the size-limited archive.
- The model runs in an isolated subprocess. Reading the submission, writing
  temp files, and using ordinary torch/CUDA/numpy caches is allowed. Only
  hidden evaluation ground truth is unreadable from disk; open attempts are
  denied by the kernel.
- Adaptation logs are recorded on the organizer side and may be audited.
  Suspicious score patterns are flagged for manual review.

### Optional LLM gateway

The organizer provides the only allowed evaluation-time network endpoint via:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `REALPDE_GATEWAY_MODEL` (listed as `gemini-3.5-flash` on 2026-08-14)

Only OpenAI-compatible chat completions are exposed; admin and key-management
routes are blocked. The gateway itself enforces fixed model versions, token
limits, timeouts, and request logging. Participant keys, private LLM services,
and all other direct outbound network access are prohibited. LLM latency is
timed, so calls need explicit frequency, token, timeout, and total budget
bounds. The kit's `agentic_demo` falls back to a deterministic rule controller
when gateway use fails.

## Baseline checkpoints and size strategy

Approximate local sizes:

| Model | fp32 | packed fp16 | Submission implication |
|---|---:|---:|---|
| CNO | 31 MB | not needed | Fits comfortably |
| Transolver | 49 MB | not needed | Fits comfortably |
| FNO | 385 MB | 193 MB | fp32 cannot fit; packed fp16 can fit narrowly |

Both simulation-pretrained and real-finetuned variants are downloaded. The
starter loader derives architecture hyperparameters from checkpoint shapes;
repository YAMLs are not reliable for these competition checkpoints.

FNO spectral weights are complex, so naive `.half()` is invalid. Use the kit's
complex-safe `pack_ckpt_fp16.py`/`unpack_fp16` path if FNO is used. Codabench
discourages relying on packing an oversized design; model/checkpoint size should
be budgeted from the start.

## Phases and deadlines

All boundaries are UTC, not Anywhere-on-Earth.

| Event | Date / interval |
|---|---|
| Launch | 2026-07-05 |
| Warm-up | 2026-07-05 through 2026-07-19 |
| Original Development phase | archived after scoring update |
| Current Development phase | 2026-08-05 00:00 through 2026-09-27 23:59:59 |
| Decision phase | 2026-09-28 through 2026-10-25 |
| Final results | 2026-11-10 |
| Code/fact-sheet deadline | 2026-11-25 |
| NeurIPS presentation | 2026-12-06 |

Current Development limits: **1 submission/day**, maximum **100 submissions in
the phase**. Warm-up was 3 submissions/day. The current phase is a
hidden-validation leaderboard. Each phase starts at `00:00:00 UTC` on its start
date and ends at `23:59:59 UTC` on its end date. The August 5 restart preserved
the earlier phase and leaderboard as frozen/archived records. The Overview
“Phases” table still lists the original Development start of 20 July; the
authoritative current window is the 5 August restart through 27 September.

## Final ranking and prizes

- The top 10 Development leaderboard teams are shortlisted.
- Organizers retrain each method from scratch on the competition GPU cluster
  and evaluate it on a private test set with unseen parameter regimes. Those
  Decision-phase private scores decide the top 3, not the Development
  leaderboard by itself. Organizers also verify code integrity, rule
  compliance, and robustness.
- A shortlisted solution that cannot be reproduced through retraining, or that
  shows improper use of evaluation data, is disqualified and replaced by the
  next-ranked team.
- Exact finalist training-code packaging is expected in the FAQ about five days
  before Development ends.
- Track 2 prizes: USD 6,000 / 3,000 / 1,500 for places 1/2/3, funded by
  Uniforce AI Ltd.
- Top 3: invited oral presentations. Top 5: joint results-paper invitation and
  certificates; each may nominate one advisor (for example a faculty
  supervisor) for authorship/certificate.
- Prize eligibility requires complete reproducible code and open-sourcing the
  winning solution before the NeurIPS results event. The winning team chooses
  the release license.
- Where the published Decision-phase procedure does not settle a question, the
  organizers decide and will publish the reasoning with the result.

## Submission/terms reminders

- Data is CC BY-NC 4.0 for non-commercial research and competition use.
- Do not read, infer, retain, encode, reverse-engineer labels of, or
  redistribute hidden evaluation data. Do not exploit implementation bugs.
- Prediction artifacts are withheld because previous ground truth is exposed
  during the TTT stream.
- Organizers may run/rerun/inspect submissions and describe approaches/scores in
  rankings and the results paper. Submission does not transfer ownership or
  authorize public release of non-winning code/weights.
- If Codabench reports a transient bundle-download/storage error, it is a known
  platform failure. Failed attempts do not consume quota; refresh and retry
  later. A submission stuck in `Submitting` may require organizer cleanup.
- Concrete Track 2 example: E024 submission `897439` reported a transient error
  fetching the organizer `ingestion_program` bundle. The live tooltip stated
  that participant code was not at fault, requested resubmission, and returned
  the attempt's quota.

## Practical development priorities

1. Build a strict streaming harness and test state reset/previous-pair alignment.
2. Establish no-adaptation and lightweight-adaptation baselines on real held-out
   trajectories, split by whole Re/AoA cases to mimic hidden regimes.
3. Track all five known subscores instead of optimizing only point error.
4. Treat uncertainty bounds as a calibrated model component; narrow uncalibrated
   bounds score zero where they miss.
5. Budget adaptation and reset costs in milliseconds per step and keep total run
   well below ten minutes.
6. Keep training deterministic/reproducible and every sample traceable to the
   official release in anticipation of organizer retraining.
7. Add hard pre-submission checks for archive size, output shape, finiteness,
   bounds, reset independence, and CPU/GPU compatibility.

## Source-change notes

- 2026-07-22: Track 2 bounds supported in kit v4.
- 2026-08-05: Development phase restarted; scorer updated; kit v6 required;
  reset time now charged; hidden composite retained.
- 2026-08-09: standard traceable data augmentation explicitly allowed.
- 2026-08-12: `train_real/7575_0.h5` announced invalid and must be excluded.
- 2026-08-16: local snapshots of every Codabench tab stored under
  `competition_docs/`. Added isolated-subprocess/temp-file rules, adaptation-log
  audit, hidden-data eligibility, finalist training-code-only package, Rel-L2
  `1e-8` floor, image `torchvision`/`torchaudio`, Uniforce prize sponsor, and
  Hugging Face vs Drive quota.
