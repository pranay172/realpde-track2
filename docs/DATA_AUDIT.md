# Track 2 release audit and validation split

Checked: **2026-08-15 (IST)**

This is the human-readable companion to the machine-readable
[`configs/data/release_v1.json`](../configs/data/release_v1.json) audit and
[`configs/splits/real_regime_v1.json`](../configs/splits/real_regime_v1.json)
split. The audit was generated from the official Hugging Face release after
safe extraction into `../RealPDE-Competition-Data/`. Prefer Hugging Face over
the Google Drive mirror; Drive has a per-file anonymous download quota.

## Integrity and extraction

Both compressed archives passed a complete gzip/tar read. Every member is a
regular `.h5` file below the expected single top-level directory; there are no
absolute paths, `..` components, duplicate members, links, devices, or other
unsafe member types.

| Archive | Compressed bytes | Files | Extracted bytes | SHA-256 |
|---|---:|---:|---:|---|
| `train_real.tar.gz` | 7,393,582,393 | 82 | 7,410,338,948 | `20f19e3910028d76573848064fc55d470a4fb4609bb56972dd7b387ca0d515ea` |
| `train_sim.tar.gz` | 8,826,593,178 | 100 | 9,844,735,200 | `09d0b94c3a23e0ee8dc954c4c903710ff565d1ed3e36f86c7a90816d32483657` |

Extraction used non-overwriting `tar` options after confirming that both
destination directories were absent. The extracted byte totals exactly match
the archive headers. About 305 GiB remained free afterward.

## HDF5 inventory

Every HDF5 structure, time vector, coordinate array, and channel value was read
and checked. No `u`, `v`, or `p` value is NaN or infinite.

| Property | Real | Simulation |
|---|---|---|
| Trajectories | 82 released; 81 valid | 100 |
| Parameter grid | 18 nominal Re values, 5 AoAs; incomplete | 20 Re values x 5 AoAs; complete |
| Frames | 79 x 868; one each x 607, 492, 282 | 100 x 1000 |
| Root datasets | `aoa,re,t,u,v,x,y` | `aoa,p,re,t,u,v,x,y` |
| Time dtype | `float32` | `float64` |
| Field dtype | `u,v`: `float64` | `u,v,p`: `float32` |
| Field shape | `(frames,64,128)` | `(frames,64,128)` |
| Coordinate shape/dtype | `(64,128)`, `float64` | `(64,128)`, `float64` |
| Coordinate grids | 2 hashes; one tiny numerical variant | 1 hash |

The three shorter real trajectories are `10125_0.h5` (607 frames),
`21600_5.h5` (492), and `24150_20.h5` (282). All other real trajectories have
868 frames. Real time spacing is approximately 0.02; simulation spacing is
`0.02002002002`.

The real release is not a 100-case full grid. It omits nominal Re 15225 and
27975 entirely, plus cases `3750_15`, `17775_0`, `22875_5`, `22875_20`,
`24150_5`, `25425_5`, `26700_5`, and `26700_20`. Counts by AoA 0/5/10/15/20
are 17/14/18/17/16. The simulation release is the complete 20 x 5 grid.

The older note that training fields live under `measured_data/` is not true for
these archives: all datasets are at the HDF5 root. Loaders should validate the
actual root schema above.

### Parameter metadata

Simulation filenames and stored Reynolds scalars agree. In real data, 78 of 82
files have a slightly different physical Reynolds scalar—for example nominal
`12675` stores `12698`. This is systematic, not corruption. Split membership
uses the nominal filename/grid value; the audit manifest preserves both
`nominal_re` and `stored_re` for modeling.

The second real coordinate hash belongs only to `3750_20.h5`; all coordinates
differ from the common real grid by less than `9.3e-8`. Code should read each
file's grid or tolerate this numerical variation rather than requiring a
byte-identical global grid.

### Mandatory duplicate exclusion

The organizer-announced bad case was verified independently. Chunked SHA-256
hashes of its full measured arrays exactly equal `train_real/6300_0.h5`:

| Array | SHA-256 |
|---|---|
| `u` | `a8bb298ada3fc7ba6b6055f743fd72ebd5e2a0d15822797923ae51b62d9f2d9a` |
| `v` | `09d45be9e25d673ecf4fc4562c4e095d4496b9bb26c3940c244445e75b6e5285` |

Always exclude `train_real/7575_0.h5`. Its scalar metadata does not make its
duplicated measurements valid.

## Validation split `real_regime_v1`

The deterministic split holds out nominal Reynolds levels 12675 and 21600 plus
AoA 15. The choices are separated interior Reynolds regimes and an interior,
high-angle regime. They exercise interpolation across unseen parameters while
retaining boundary regimes for training. No random seed is involved.

| Real partition | Rule | Trajectories |
|---|---|---:|
| `train` | Re not held out and AoA not held out | 56 |
| `val_re` | held-out Re, other AoAs | 8 |
| `val_aoa` | held-out AoA, other Re values | 15 |
| `val_joint` | held-out Re and held-out AoA | 2 |

The four real partitions are disjoint and cover all 81 valid trajectories.
Windows must be created only after selecting a partition, so no trajectory or
overlapping window can cross train/validation boundaries.

The manifest provides two simulation policies:

- `standard_train` contains all 100 simulations and matches the official
  sim-pretraining setup.
- `strict_regime_train` contains 72 simulations; its 28-case held-out complement
  removes every selected Re or AoA. Use this for the stronger test in which the
  validation parameters are unseen in both domains.

Released `sim_real_ft` checkpoints were trained on the full real release. Their
scores on this split are useful reproduction diagnostics, but are not
leakage-safe validation scores. A genuinely leakage-safe comparison requires
retraining from a permitted initialization using only the manifest's training
partitions.

## Reproduction

From `RealPDE_T2/` with the evaluator-compatible environment available at the
project root:

```bash
../.venv/bin/python scripts/audit_release.py \
  --full-value-scan --verify-archives
../.venv/bin/python -m unittest discover -s tests -v
```

The exhaustive audit took 2:10 wall time on the local CPU/storage path and
peaked at about 74 MiB resident memory. Running without the two flags performs
the structural audit and regenerates the same split more quickly, but does not
re-read every field value or independently re-hash the archives.
