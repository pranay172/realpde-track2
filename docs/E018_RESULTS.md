# E018 static FNO/persistence blend

Run: **2026-08-21 (IST)**

E018 tested a fixed convex mixture of the leakage-safe E004 FNO and the
current measured input window (persistence). The train-side gates rejected the
entire pre-registered grid, so no validation or complementary-fold fields were
opened and no candidate was packaged or submitted. The frozen E007/E016 paths
remain the references.

## Hypothesis and locked decision rule

The hypothesis was that a small current-input mixture would recover TKE while
preserving the E004 Rel-L2 and MVPE advantages over persistence. The only
candidate weights were:

```text
w ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25}
y_w = (1 − w) FNO(x) + w x
```

The mixture was formed in physical space using the official input and target
normalization statistics, then converted back to target-normalized space. The
wrapper discards `prev_target_norm`, performs no adaptation, and keeps the
E006 interval fixed at `(alpha=0.025, beta=0.1)`. SPS was reported but never
used for weight selection.

Selection required the smallest weight satisfying all three calibration gates:

- TKE score gain ≥ 0.50 versus `w=0`;
- Rel-L2 score drop ≤ 0.15;
- MVPE score drop ≤ 0.15.

The same gates had to pass on the disjoint train audit. Only then would the
selected weight be evaluated once on `real_regime_v1` and once on the existing
E010 complementary-fold checkpoint. The locked validation gates versus E007
also required Time ≥ 86 and SPS score drop ≤ 0.5.

## Leakage-safe setup

- Base checkpoint: E004 packed FNO, SHA-256
  `75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`.
- Config: [`e018_static_blend.json`](../configs/experiments/e018_static_blend.json),
  SHA-256 `9689281812b93d58b98cb6e7efebf2d092aaf7e56904e0a964fec0c5cc6d341a`.
- Calibration: 15 `real_regime_v1/train` trajectories at nominal Re
  3750/8850/16500/24150, 601 windows.
- Audit: the remaining 41 `real_regime_v1/train` trajectories, 1,709 windows.
- Validation: all 25 `val_re`/`val_aoa`/`val_joint` trajectories were excluded
  from both train-side partitions and were never read by the run.
- Seed: 0.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, NumPy 1.26.4, RTX 5050,
  `.venv-gpu`, CUDA 12.8.
- Working-tree Git HEAD at execution: `8d6d40448bf4701ab98b3e911d16cd2646286dfa`.
- GPU train-side scoring wall time: 41.83 s calibration + 103.66 s audit.

## Results

Higher scores are better. The train-side `Time` field is not meaningful here
because calibration uses zero elapsed values; validation timing was never run.
The fixed E006 bounds are shown through SPS only as a diagnostic.

### Calibration (601 windows)

| w | Rel-L2 | TKE | MVPE | SPS | Coverage | ΔRel-L2 | ΔTKE | ΔMVPE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 95.706 | 71.600 | 95.897 | 44.234 | 80.431% | 0.000 | 0.000 | 0.000 |
| 0.05 | **95.712** | 71.134 | **95.922** | **44.271** | 80.602% | +0.006 | −0.465 | +0.025 |
| 0.10 | 95.697 | 70.747 | 95.933 | 44.232 | 80.496% | −0.010 | −0.853 | +0.037 |
| 0.15 | 95.661 | 70.450 | 95.931 | 44.125 | 80.371% | −0.045 | −1.150 | +0.034 |
| 0.20 | 95.606 | 70.250 | 95.915 | 43.965 | 80.218% | −0.100 | −1.350 | +0.019 |
| 0.25 | 95.532 | 70.149 | 95.888 | 43.763 | 80.078% | −0.174 | −1.451 | −0.009 |

The best nonzero candidate for TKE was still `w=0.05`, with a **−0.465**
TKE-score change rather than the required +0.50 gain.

### Audit (1,709 windows)

| w | Rel-L2 | TKE | MVPE | SPS | Coverage | ΔRel-L2 | ΔTKE | ΔMVPE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 95.781 | 70.953 | 96.107 | 41.345 | 75.953% | 0.000 | 0.000 | 0.000 |
| 0.05 | 95.772 | 70.524 | **96.131** | **41.415** | 76.062% | −0.008 | −0.429 | +0.024 |
| 0.10 | 95.745 | 70.192 | 96.144 | 41.417 | 76.059% | −0.035 | −0.762 | +0.037 |
| 0.15 | 95.700 | 69.963 | 96.147 | 41.365 | 76.002% | −0.080 | −0.991 | +0.040 |
| 0.20 | 95.638 | 69.840 | 96.140 | 41.271 | 75.899% | −0.143 | −1.113 | +0.033 |
| 0.25 | 95.559 | 69.826 | 96.123 | 41.145 | 75.779% | −0.221 | −1.128 | +0.016 |

The audit confirms the calibration direction: every nonzero mixture loses TKE.
The calibration report is the ignored artifact
`artifacts/e018/calibration.json`, SHA-256 `289a2e340637c20a57220274ac98382bf83d50cd418fb7a3aac62404145d61e8`.

## Implementation and verification

- Added [`src/realpde_t2/blending.py`](../src/realpde_t2/blending.py) with a
  reusable physical-space `StaticBlendPredictor`.
- Added [`scripts/evaluate_static_blend.py`](../scripts/evaluate_static_blend.py),
  which enforces calibration/audit-before-validation ordering and writes
  reproducible JSON provenance.
- Added the focused [`tests/test_blending.py`](../tests/test_blending.py) suite
  covering physical-space mixing, normalization conversion, reset delegation,
  previous-target exclusion, and invalid inputs.
- Added [`tests/test_e018.py`](../tests/test_e018.py) to lock the candidate grid,
  train-only calibration/audit membership, and complementary stress provenance.
- CPU smoke and all seven focused E018/blending tests passed. The full suite
  passed 62/62 tests under `.venv`. The full GPU run loaded the packed
  checkpoint strictly, produced finite exact-shape outputs, and stopped before
  validation when the locked train-side gate failed.

## Decision

**Reject E018.** Static FNO/current-input blending does not recover the TKE
deficit on either train-side partition; do not relax the gate, search more
weights, or open v1 after seeing this result. Keep E007/E016 frozen. The next
evidence should be I017's pre-registered spectrum/TKE diagnostic, which can
distinguish temporal-amplitude loss from spatial-spectrum underpower before a
new trained point model is attempted.
