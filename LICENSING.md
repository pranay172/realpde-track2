# License scope

The [root MIT license](LICENSE) applies only to original contributions by this
project's contributors, to the extent they own the rights. It does not relicense
upstream-derived implementations, data, weights, or separately obtained
dependencies. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Retained adapted source — CC BY-NC 4.0

The following files include FNO forward/trunk logic adapted from RealPDEBench:

- `src/realpde_t2/dual_head_fno.py`
- `src/realpde_t2/variance_head_fno.py`
- `submission/e021/submission.py`
- `submission/e024/submission.py`
- `submission/e029/submission.py`

These five adapted files are distributed under **CC BY-NC 4.0**, not the root
MIT grant. Credit: Zongyi Li (original FNO implementation) and RealPDEBench
contributors. Our changes add separate temporal-mean/fluctuation projections,
checkpoint conversion, an experimental variance head, and streaming adaptation.
The three standalone submission wrappers embed adapted FNO forward logic;
they are not merely import-only clients of the external model. All five files
carry explicit attribution, modification, and license headers. Their external
base implementation is not bundled.

Publication headers do not change Python implementation ASTs, but do change
source bytes. Historical wrapper/archive hashes identify the original scored
artifacts, not packages rebuilt from this public snapshot.

The kit's FNO source was compared with the locally pinned benchmark source:
only the two package import paths differed. Benchmark source commit:
`62f4c80ab17f78933d046f2b038531dbc6a478a0`.
[Upstream file](https://github.com/AI4Science-WestlakeU/RealPDEBench/blob/62f4c80ab17f78933d046f2b038531dbc6a478a0/realpdebench/model/fno.py).

## External material is not bundled

Organizer kits, page snapshots, example arrays, and model/data artifacts were
omitted from this public snapshot. Acquire required dependencies from their
authorized sources as described in [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).
Restoring them locally does not make them part of our MIT grant or authorize
uploading them in a public release.

RealPDEBench is [CC BY-NC 4.0](https://github.com/AI4Science-WestlakeU/RealPDEBench/blob/main/LICENSE).
NVIDIA-derived CNO utilities use the
[StyleGAN3 source license](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt),
including research/evaluation-only limitations. einops has its own MIT notice.
Retain the relevant upstream notices when using those dependencies. The complete
model/data stack must not be presented as unrestricted MIT software.

No right to redistribute competition-specific files with unclear terms is
assumed. This inventory records the observed scope; it is not a legal warranty.
