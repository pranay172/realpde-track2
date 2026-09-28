# E019 train-only spectrum/TKE route diagnostic

Run: **2026-08-21 (IST)**

E019 is a diagnostic-only audit of frozen leakage-safe E004. It determines
whether the next isolated point-model experiment should be a train-scaled
binned spatial-spectrum auxiliary, an identity-initialized dual-head temporal
mean/fluctuation decoder, or neither. It does **not** train, adapt, alter
intervals, inspect validation/complementary fields, package, or submit.

## Pre-registered setup and decision rule

- Base: E004 packed FNO, SHA-256
  `75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`.
- Config: [`e019_spectrum_tke_diagnostic.json`](../configs/experiments/e019_spectrum_tke_diagnostic.json),
  SHA-256 `b91c88b6ee88bcfaec8f89b1ab756cb639225553a30a9f542c27a75bbbd205c0`.
- Train-only split: the unchanged E006/E018 calibration Re
  `{3750, 8850, 16500, 24150}` (15 trajectories / 601 windows), with every
  other `real_regime_v1/train` case as the disjoint audit (41 / 1,709). The
  25 `v1` validation trajectories were not opened.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, NumPy 1.26.4, RTX 5050,
  CUDA 12.8. Working-tree base commit: `eccc6e1`.
- The loader reads one released trajectory at a time and emits its contiguous
  stride-20 20→20 windows in manifest order. It reproduces E018's frozen-E004
  mean train-side TKE errors: calibration `0.793300` (score 71.600) and audit
  `0.818748` (70.953).

For every physical-space `u/v` window, E019 reports the exact orthogonal
decomposition

\[
\sum e^2 = T\sum(\operatorname{mean}_t e)^2 +
\sum(e-\operatorname{mean}_t e)^2,
\]

the official TKE map `0.5(var_t(u)+var_t(v))`, and an rFFT radial spatial
spectrum of `u'=u-mean_t(u)` and `v'=v-mean_t(v)`. rFFT modes are
Hermitian-weighted and divided by `(H*W)^2`; summed spectrum energy therefore
equals mean temporal-fluctuation energy in physical velocity-squared units.
Frequency is cycles per grid cell. The zero spatial mode is reported separately
and excluded from the low/mid/high comparison.

Before data output, E019 fixed these route gates:

1. **Spectral auxiliary:** on *both* partitions, combined full-domain high-band
   prediction/target energy ratio ≤0.90; high ratio at least 0.10 below low;
   and Spearman correlation between per-window binned-log-spectrum RMSE and
   organizer per-window TKE Rel-L2 ≥0.25.
2. **Dual head:** only if the spectral route fails, and either centered
   full-domain fluctuation SSE share ≥0.60 on both partitions, or the combined
   TKE-map amplitude ratio is consistently ≤0.90 or ≥1.10 on both.
3. Otherwise choose neither. Thresholds were not changed after output.

The wake breakdown uses only the explicit bounding rectangle of the official
MVPE probes: `y=8:25`, `x=13:38` on the 32×64 grid. A wake-only FFT is not
reported because applying a new spatial window would make its spectrum
ambiguous.

## Results

### Mean versus temporal fluctuation error

| Partition / region | Combined fluctuation SSE share | u share | v share |
|---|---:|---:|---:|
| Calibration, full domain | 66.607% | 63.794% | 75.654% |
| Audit, full domain | 66.991% | 64.807% | 75.174% |
| Calibration, wake rectangle | 63.743% | 62.818% | 69.792% |
| Audit, wake rectangle | 65.474% | 64.765% | 71.017% |

The decomposition closes to relative numerical error below `6e-14` in every
reported combined region. Centered temporal fluctuations therefore account for
about two-thirds of E004 field SSE on both disjoint train-side partitions.

### Official TKE-map amplitude

| Partition / region | Energy ratio (pred/target) | L2-amplitude ratio | TKE-map global Rel-L2 |
|---|---:|---:|---:|
| Calibration, full domain | 0.241 | 0.216 | 0.849 |
| Audit, full domain | 0.232 | 0.215 | 0.854 |
| Calibration, wake rectangle | 0.152 | 0.131 | 0.912 |
| Audit, wake rectangle | 0.142 | 0.122 | 0.919 |

The full-domain TKE-map amplitude ratio is consistently about **0.22**, far
below the locked 0.90 underprediction gate. Wake under-amplitude is stronger.
For full-domain components, u amplitude ratios are 0.189/0.191 and v ratios
are 0.254/0.252 (calibration/audit).

### Spatial spectrum and TKE association

| Partition | Low ratio | Mid ratio | High ratio | Low − high | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Calibration | 0.261 | 0.201 | 0.160 | 0.1005 | −0.416 | −0.474 |
| Audit | 0.252 | 0.187 | 0.158 | 0.0944 | −0.290 | −0.337 |

All named spectrum bands are underpowered, with high-frequency energy lower
than low-frequency energy. But the required association is not merely weak: it
has the **opposite sign** on both partitions. Further, the audit low-minus-high
gap misses the fixed 0.10 threshold. Thus a smaller binned log-spectrum error
does not identify lower organizer TKE error in this data/model setting.

## Locked decision

**Choose the identity-initialized dual-head mean/fluctuation decoder as the
next candidate route. Do not train a spectral auxiliary from E019 evidence.**

The spectral mechanism fails its route gate despite global high-band
underpower: correcting a spatial spectrum is not shown to improve the official
temporal-variance TKE map. The non-spectral evidence is strong and stable:
centered fluctuation error dominates on calibration and audit, and TKE-map
amplitude is underpredicted by roughly 80% in the same direction everywhere.

The ignored raw report, including per-window IDs, official TKE errors, and log
spectrum errors, is [`artifacts/e019/diagnostic.json`](../artifacts/e019/diagnostic.json),
SHA-256 `a1f954905b4f67112f70364efbdf8d1086070d37e8ea32b6cb03c31bc7c8a5ea`
(242 KiB). Runtime was 22.61 s calibration + 64.29 s audit, 86.90 s total;
peak allocated GPU memory was 1,574,978,048 bytes.

## Verification and next action

- Focused tests cover exact mean/fluctuation closure, official TKE equivalence
  including the combined-map cross-term, rFFT Parseval normalization and bands,
  zero-safe log spectra, tie-aware Spearman, trajectory batching, train-only
  membership, and spectral-route priority.
- The GPU run loaded E004 strictly, kept pressure excluded, and recorded an
  empty `validation_field_values_accessed` list.

Next, if requested, reserve one dual-head decoder experiment that is exactly
E004 at initialization, preserves the base physical field loss, and has
pre-registered Rel-L2/MVPE, pack-size, Time, and complementary-fold gates.
