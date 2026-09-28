# E025 — Frozen-trunk variance-map prediction head (I028): negative

Date: 2026-08-22. Status: **Complete — negative; validation not opened (D036).**

## Hypothesis

E023 measured a per-pixel map-oracle TKE ceiling of 89.8 (val) / 89.5 (cal)
vs 77.3 deployable (emaV). I028 asked whether a *learned* predictor of the
current window's variance map could capture part of that gap. The isolated,
lowest-risk instantiation: a zero-initialized linear head `fc_var`
(128→3, 393 parameters) on **detached frozen-E020 trunk features**, trained
only with the official TKE-form relative loss on the variance map
`V̂ = V_x · exp(δ(x))`, deployed exactly like emaV as the fluctuation scale
`sqrt(V̂ / V_model)` (σ=0.5 smoothing, clip [0.2, 5]). Because no point-path
parameter receives gradients, the point model is bitwise-E020 — the E013/E014
failure mode (point-loss degradation) is structurally impossible.

Pre-registered gate: open `v1_val` only if head or blend beats emaV on cal by
> 0.10 TKE with Rel-L2 within 0.15 of the fixed-1.5 baseline.

## Setup

- Trunk: frozen E020 packed SHA `b371a2ce3265` (leakage-safe).
- Train: `real_regime_v1/train` (56 trajectories / 2,310 windows), Adam 3e-4,
  600 updates, batch 4, seed 0; 26.1 s on RTX 5050 (`.venv-gpu`).
- Selection: cal subset (601 windows) via the E023 offline replay; variants:
  baseline fixed-1.5, emaV, head, geometric blend sqrt(V̂·V_ema).
- Code: `src/realpde_t2/variance_head_fno.py`, `scripts/train_variance_head.py`,
  `scripts/evaluate_variance_head.py`; 4 new model tests (trunk
  bitwise-consistency, zero-init identity, temporal-mean semantics, gradient
  isolation); full suite 88/88 in `.venv`.

## Results

Training loss moved only 0.813 → 0.738 (−9%): trunk features carry very
little signal about the target variance map beyond `V_x` itself.

| Variant (cal, 601) | Rel-L2 | TKE | Composite* |
|---|---:|---:|---:|
| fixed 1.5 (reference) | 95.751 | 74.634 | 74.839 |
| **emaV (E023)** | 95.445 | **75.976** | 75.057 |
| trained head | 95.096 | 64.257 | 72.450 |
| blend sqrt(V̂·V_ema) | 95.396 | 73.617 | 74.565 |

\* with default-width SPS (selection phase convention).

The trained head is ~10 TKE *below its own zero-initialization* (the
input-map estimate scores ~73–74 on cal, E023): the modest objective gain it
did find misallocates the deployed scale map. Both candidates fail the
pre-registered gate; `v1_val` was not opened.

## Conclusion

1. The current window's variance map is **not linearly predictable** from
   frozen trunk features — consistent with E023's finding that even an EMA of
   the *true* past variance maps saturates at the per-channel-oracle
   information content. The oracle-map gap (≈ +12 TKE) is dominated by
   window-level turbulent intermittency, which is not available in the input
   window, the past windows, or the trunk features.
2. Close the frozen-trunk linear-head family (D036). A deeper convolutional
   head is P3 at most and requires new evidence before any run.
3. E024 (E021 weights + emaV) remains the submission path; the realistic TKE
   frontier for streaming correction on this base model is ≈77.3 local.
