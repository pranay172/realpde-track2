# Reproducing Track 2

**Status: research archive; no fresh post-cleanup end-to-end rerun.**
Full competition data, trained weights and generated submissions were deleted.
The organizer kit, its three example assets, and private development history
are intentionally absent from this public snapshot.

## Workspace and dependencies

Use the historical directory layout even if the GitHub names use hyphens:

```text
RealPDE/
  .venv/                         # recreate; not shipped
  .venv-gpu/                     # recreate; not shipped
  RealPDE-Competition-Data/       # reacquire from authorized release
  RealPDE_T1/                    # clone realpde-track1 here
  RealPDE_T2/                    # clone realpde-track2 here
```

Only the track you need must be cloned. Copy its `environment-locks/` folder
to the parent RealPDE root and follow [the installation guide](environment-locks/README.md).
Snapshots pin Python 3.10.20, all installed dependencies and PyTorch indexes.
They are online version snapshots, not archived wheels or cryptographic wheel
locks. Linux x86-64 is the supported historical platform. CPU evaluation used
Torch 2.2.2+cu121; local RTX 5050 work used Torch 2.7.1+cu128.

Obtain the **competition** release (not an arbitrary newer benchmark dataset)
from [the authorized dataset source](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data).
Restore `train_real/` and `baseline_checkpoints/sim_pretrain/` beneath the shared
data directory. Preserve release filenames and do not use `7575_0.h5`.
Data/checkpoint permissions remain separate from our code license.
No full dataset or trained solution checkpoint is included.

## Organizer dependencies (not bundled)

Before model imports, training, evaluation, or packaging, follow
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md) to obtain the exact historical
kit. Do not treat an absent dependency as a passing test. Some model imports
fail immediately without the kit. Kit downloads were not verified in this pass;
if you cannot obtain the required version legitimately, full reproduction is
blocked, while the source, results, and data-free checks remain usable.

## Data-free checks

From the track repository:

```bash
python3 scripts/check_archive.py
python3 scripts/publication_audit.py
```

These require only Python before Git initialization; they inspect source/JSON,
publication-document paths, requirement pins/checksums, and selected privacy
patterns. Once this snapshot has its own Git repository, the audit also checks
its reachable history. They do not
run models. GitHub CI runs the same archive check, not the full historical suite.


## Reconstruct the E033 recipe

After restoring the authorized data, official simulation FNO checkpoint and
environment, from `RealPDE_T2/`:

```bash
../.venv-gpu/bin/python scripts/train_dual_head_fno.py \
  --config configs/experiments/e033_full_real_sim_long.json --skip-eval
```

The historical run used 81 trajectories, 3341 stride-20 grid windows, 1800 updates
and about 327 seconds on RTX 5050. The initializer's recorded SHA and strict
conversion checks are in [E033_RESULTS.md](docs/E033_RESULTS.md).
This is full-data training; any evaluation on those public trajectories is not
held-out generalization evidence.

### Package a rebuilt checkpoint under a new identity

`configs/experiments/e033_submission.json` is historical evidence: its
`checkpoint_sha256` identifies the deleted original E033 checkpoint.
`package_submission.py` deliberately rejects different bytes. Retraining the
same recipe does not guarantee a byte-identical checkpoint, so do not use that
historical config unchanged or disable the hash guard.

After training completes, inspect `artifacts/e033/training_report.json`: verify
the intended config, successful strict initialization, 81 valid trajectories,
1800 updates and finite loss. Compute the new packed checkpoint's identity and
confirm it equals the report's `checkpoint_sha256`:

```bash
sha256sum artifacts/e033/fno_dual_head_fp16.pth
```

Make a local copy of `configs/experiments/e033_submission.json` at
`artifacts/e033/rebuild_submission.json` (do not overwrite an existing rebuild
config). Edit these fields, leaving all adaptation, interval and validation
settings unchanged:

```json
{
  "experiment": "E033_REBUILD",
  "checkpoint": "artifacts/e033/fno_dual_head_fp16.pth",
  "checkpoint_sha256": "<replace with the verified 64-character SHA-256 above>",
  "output": "artifacts/e033_rebuild",
  "archive_name": "e033_rebuild_candidate.zip"
}
```

This is a field-edit example, not a complete config. Replace the placeholder;
retain every other field from the copied historical config. Paths resolve from
the repository root, not the config's directory. The output directory must be
unused; choose another rebuild name throughout these instructions if it exists.
Keep the new config and training report together as provenance. Recording a new
hash identifies bytes; it does **not** establish model correctness or recover
E033's official score. Run the restored-environment tests below, then the
packaging and parity gates:

```bash
../.venv/bin/python scripts/package_submission.py \
  --config artifacts/e033/rebuild_submission.json --skip-full-eval --device cpu
../.venv/bin/python scripts/check_deployed_parity.py \
  --submission artifacts/e033_rebuild/extracted --output artifacts/e033_rebuild/parity_report.json
```

Inspect scripts/configuration before running: packaging invokes contract/container
checks and writes generated artifacts. Use a fresh output directory rather than
overwriting a valuable package. The original archive hash identifies historical
bytes; a rebuilt archive is a new artifact requiring verification. License-only
header additions also change wrapper/archive bytes without changing model logic.

## Restored-environment tests

```bash
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -v
```

Historical E033 verification recorded 108 passing tests. That suite was **not**
rerun during this publication pass because environments have been removed.
Tests using organizer examples require those assets; model/package/data tests
may need freshly generated dependencies. Do not count missing-asset skips as
successful end-to-end reproduction.

Verify previous-target alignment, full trajectory reset, finite ordered bounds,
and exact package/research parity. Use the pinned official Docker image from
[E033_RESULTS.md](docs/E033_RESULTS.md); CPU smoke results are not GPU runtime
or held-out quality estimates.

Historical corrected initialization provenance for E020/E021 must remain
visible; do not silently reinterpret those experiments as verified simulation
initialization. E031/E032 are the controlled recipe-selection evidence.
