# E029 — Hardened emaV resubmission

Date: **2026-08-23 (IST)**. Status: **complete — official score 76.760173**.

## Why E029 exists

E024 did not fail in participant code. Codabench submission `897439` is marked
with a transient platform storage error while fetching the organizer
`ingestion_program` bundle; its live tooltip explicitly requests resubmission
and the attempt returned its quota.

E029 nevertheless keeps two low-risk hardening changes prepared during the
initial diagnosis:

1. The emaV 3×3, σ=0.5 reflect smoothing is expressed as separable sliced
   weighted sums instead of dynamic `F.pad` plus grouped `conv2d`.
2. Exceptions inside emaV variance-state update or scale construction clear
   the emaV state and use fixed 1.5 fluctuation scaling for that and subsequent
   steps of the trajectory. A normal first window remains identity-scaled
   because no previous target variance is available.

The E021 full-release dual-head checkpoint, mean-bias EMA, first-window
behavior, and frozen SPS bounds `(0.02, 0.14)` are unchanged.

## Exact uploaded package

- Archive: `artifacts/e029/e029_candidate.zip`
- SHA-256:
  `bdf06b5af726b99b97fe333fb55f3f5968c5cff3c0e890d5d3cbca7525ca08c4`
- Compressed size: `185,748,575` bytes
- Extracted size: `201,433,697` bytes, below the `256,000,000`-byte gate
- Wrapper SHA-256:
  `d959da0294349e3969f2b5631f6a504213d67f904ee538582e432323ed4ca8a3`
- E021 checkpoint SHA-256:
  `3d9a6b93b8634db08b303d220a6a5df48ef35beb84628d8456d8fddc6d903dfe`
- Config SHA-256:
  `eaa21dae322db11c6e79ccf9374672d84d2298706e3a0851e58555b611db02f7`

The ZIP contains exactly ten files: root `submission.py`, `model.pth`,
`load_baseline.py`, and the slim bundled `rpde_baselines` FNO dependency tree.
The source wrapper, extracted wrapper, and archived wrapper are byte-identical;
the same is true of the configured, extracted, and archived checkpoint.

## Quality preservation

Leakage-safe deployed validation with E020 weights reproduces E024:

| Rel-L2 | TKE | MVPE | SPS | Coverage |
|---:|---:|---:|---:|---:|
| 94.094471 | 77.317891 | 95.664647 | 36.450294 | 71.90% |

An independent comparison of the exact E024 and E029 extracted archives over
one complete 42-window CUDA trajectory found maximum absolute differences of:

| Output | Maximum absolute difference |
|---|---:|
| Prediction | `3.58e-7` |
| Lower bound | `3.58e-7` |
| Upper bound | `4.77e-7` |

No fallback activated. These differences are ordinary float32 operation-order
roundoff and do not change the validated scores.

## Independent gates

- Pre-upload evaluator-environment suite: **89/89 passed**; two cleanup
  regressions for the standalone E029 constants/arithmetic and injected
  fallback raise the current suite to **91/91 passed**.
- ZIP CRC/integrity and root-file contract: passed.
- Host starting-kit `local_eval`: passed.
- Synthetic exception injected into the exact archived wrapper: prediction and
  bounds remained finite, fallback counter became 1, and reset cleared it.
- Exact extracted archive on local RTX CUDA: **168/168** real windows across
  four trajectories, zero failures.
- Exact extracted archive in the pinned official Torch 2.2.2 CPU image:
  **168/168** real windows across four trajectories, zero failures.
- Reset, finite output, exact shapes, bounds-on-every-step, and
  `adapt_loss=None` checks passed throughout.

The unavailable local matrix cell is Torch 2.2.2 on CUDA: that wheel cannot
execute on the RTX 5050 (`sm_120`). E029 removes the only emaV operation surface
not inherited from the already successful E022 package, while the guarded path
prevents ordinary emaV exceptions from terminating a trajectory.

## Official result

The unchanged archive later scored successfully after the evaluation-service
failures. E022 is the direct comparator because it uses the same E021 model,
mean-bias tracker, and uncertainty recipe, with fixed fluctuation scaling in
place of emaV.

| Metric | E022 | E029 | Delta |
|---|---:|---:|---:|
| Rel-L2 | 93.757063 | **93.195166** | −0.561897 |
| TKE | 68.393174 | **75.031242** | **+6.638068** |
| MVPE | 92.971467 | **92.971467** | 0.000000 |
| Time | 89.017010 | **88.743410** | −0.273600 |
| SPS | 30.855131 | **30.378171** | −0.476960 |
| Final | 76.499614 | **76.760173** | **+0.260559** |

The exact MVPE match confirms that the unchanged mean path behaved as intended.
emaV recovered substantially more hidden-stream fluctuation energy, validating
the mechanism behind E023. The Rel-L2 decrease shows the remaining limitation:
better amplitude does not guarantee better phase or pointwise fluctuation
alignment. SPS moved with the changed point predictions despite unchanged
bounds, and runtime remained safely above the established floor.

## Decision

D042 adopts E029 as the official baseline. Preserve the exact archive and its
hash. The next mechanism should improve coherent fluctuation phase/structure
while retaining emaV's amplitude gain, rather than retuning variance magnitude,
mean correction, or uncertainty widths.
