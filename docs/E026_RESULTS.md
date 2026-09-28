# E026 — Coverage-feedback SPS bounds + input mean-map blend: both negative

Date: 2026-08-22. Status: **Complete — both mechanisms rejected (D037).**

## Hypotheses

Against the two largest unattacked official↔local gaps (SPS −6.0, MVPE −2.7),
on top of the frozen E020 + emaV point model:

1. **I029 — online coverage-feedback bounds.** Track the empirical coverage of
   each revealed window (low-variance integral statistic over ~82k elements,
   unlike E017's noisy per-element residuals) and multiplicatively steer the
   width factor `f ← clip(f·(target/ĉ)^κ, 0.6, 3)` toward a target coverage.
   Self-calibration from revealed targets is the pattern that produced both
   official wins so far (bias EMA, emaV).
2. **I030 — input mean-map blend.** Blend the model temporal mean with the
   measured input-window mean (`w ∈ [0.1, 0.5]`, bias EMA re-tracked against
   the blended estimator).

## Setup

Offline replay from the E023 caches (cal 601 / audit 1,709 / one locked val
evaluation for I029 only). Grid: target {0.75, 0.80, 0.85} × κ {0.25, 0.5,
1.0}. Tracked probe: `scripts/probe_e026.py`; ignored reports:
`artifacts/prelim_probe/e026_probe.json`, `e026_val.json`.

## Results

| Variant | cal SPS | audit SPS | val SPS | val coverage |
|---|---:|---:|---:|---:|
| static (0.02, 0.14) | 44.252 | 42.161 | 36.450 | 71.9% |
| cov-t0.75-k0.25 | **45.091** | 42.223 | 35.601 | 75.5% |
| cov-t0.85-k1.0 | 43.334 | 39.783 | — | — |

| Variant | cal MVPE | audit MVPE |
|---|---:|---:|
| w=0 (reference) | 96.303 | 96.393 |
| mblend w=0.1 | 96.199 | 96.316 |
| mblend w=0.5 | 95.367 | 95.662 |

- **I030**: monotone degradation on both train-side splits at every weight →
  closed without opening validation. The E020 mean + bias EMA already
  dominates the measured input mean; blending only injects window noise.
- **I029**: cal +0.84 (controller tightens over-coverage), audit +0.06
  (missed the +0.5 gate), and the one locked val evaluation — run because
  audit is in-distribution and cannot exercise the regime-shift case the
  controller exists for, with the pass rule fixed in advance (SPS ≥ 36.450) —
  failed: coverage moved to target (75.5%) but SPS fell −0.85. Widening
  uniformly buys few marginal elements while paying `exp(−width/σ)` on every
  already-covered element; on val the static widths already sit near the SPS
  optimum.

## Conclusion (D037)

Three independent strikes in bound-adaptivity space — E017 (residual-EMA
scalar), E026 (per-position conformal), E026 (coverage feedback) — establish
that the frozen calibrated `(0.02, 0.14)` family is a genuine local optimum
on this point model. Remaining official SPS points must come from better
point predictions (the SPS branches inherit DM/TKE/MVPE errors), not from
smarter widths. Do not reopen interval adaptivity without a fundamentally
different point model.

## Postscript (E026 val run note)

The locked val evaluation also re-confirmed the E024-local baseline row
(94.094/77.318/95.665/36.450) through an independent replay path.
