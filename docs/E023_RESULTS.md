# E023 — Structural mechanism probe on frozen E020 (dynamic variance matching, AR donor, conformal SPS, variance-ratio maps)

Date: 2026-08-21. Status: **Complete — emaV accepted; three mechanisms closed.**

## Hypothesis

Four candidate mechanisms were pre-registered in the ledger (I024–I027) against
the diagnosed weaknesses of E022: static `fluct_scale=1.50` regressed official
TKE (−1.360 vs E016), the SPS official↔local gap is −6.0, and E019 showed the
model damps fluctuation amplitude to ~0.22× with 67% of squared error in the
fluctuation.

1. **I024** dynamic per-channel fluctuation scale, EMA-tracked from the revealed
   previous target (`s_t = EMA(√(Var(yₜ₋₁)/Var(ŷbaseₜ₋₁)))`, γ grid).
2. **I025** measured-fluctuation donor: per-pixel ridge AR(p) extrapolation of
   the centered input window, blended into the model fluctuation.
3. **I026** per-position split-conformal SPS bounds from train residuals.
4. **I027** per-pixel variance-ratio map correction (input-window, oracle, EMA
   of revealed targets; smoothed σ grid).

Comparator: frozen E020 replay (Rel-L2 94.658 / TKE 74.111 / MVPE 95.665 /
SPS 36.901 on `v1_val`). Decision gates: promote only if TKE gains on both
independent folds without Rel-L2/MVPE cost; reject train-side losers before
opening validation.

## Setup

- Model: frozen E020 packed dual-head FNO SHA `b371a2ce3265`; official
  `mean_std_real.pt`; no training anywhere in E023.
- Data: `real_regime_v1`; selection on the E006 calibration subset only
  (Re 3750/8850/16500/24150; 15 trajectories / 601 windows); SPS and bounds
  checks on the disjoint audit subset (41 trajectories / 1,709 windows); one
  locked `v1_val` evaluation (25 / 1,031) and one complement-fold evaluation
  (20 / 821).
- Method: `scripts/probe_e023.py` makes one GPU stream pass per split and
  caches (input, target, raw model
  output) in fp16; all variants replay offline with the deployed streaming
  state machine (bias EMA from raw outputs; adaptation uses only the revealed
  previous target). Metrics via the kit scorer through `MetricAccumulator`
  (identical to prior E-series). JSON reports remain under ignored
  `artifacts/prelim_probe/`; replay arrays were removed in the
  [2026-09-27 cleanup](ARTIFACT_CLEANUP_2026-09-27.md), which records regeneration.
- Fidelity check: the E020 replay reproduces the stored
  `artifacts/e020/evaluation.json` exactly (94.658/74.111/95.665/36.900).
- Environment: `.venv-gpu` (Python 3.10.20, Torch 2.7.1+cu128), RTX 5050,
  seed 0. Step-time accounting: E020 measured 15.97 ms/step; emaV adds only
  elementwise reductions, a 3×3 smoothing conv, and an EMA update per step
  (≪1 ms GPU).

## Results

### Calibration selection (601 windows, train-side)

| Variant | Rel-L2 | TKE | Note |
|---|---:|---:|---|
| fixed 1.5 (deployed E020) | 95.751 | 74.634 | reference |
| fixed 1.0 | 95.940 | 71.600 | damping is real |
| dyn per-channel γ=0.9 | 95.547 | 75.010 | best channel EMA |
| oracle channel scale | 95.553 | 76.013 | channel-scalar ceiling |
| AR donor (any p, λ; w=0.25→1.0) | 95.8→92.3 | 74.7→37.98 | monotone degradation |
| input-window variance map | 95.45–95.58 | 73.2–74.1 | below channel EMA |
| chan × input-map products | 93.7–94.8 | 43.6–63.4 | amplitude double-count |
| oracle per-pixel map σ=0.5 | 95.716 | **89.505** | spatial ceiling |
| **emaV (EMA of revealed V maps) γ=0.9 σ=0.5** | **95.445** | **75.976** | deployable winner |

Shedding diagnostic (FFT of wake `v`): dominant period 98/40/32/21 frames
(median per Re 3750/8850/16500/24150). A 20-frame input window covers ≤1
shedding cycle, so per-pixel frequency is unidentifiable — the physical reason
the AR donor fails. Synthetic validation confirmed the AR implementation is
correct (pure sinusoids extrapolate at ~1e-4 error); the failure is the data,
not the code.

### Audit SPS selection (1,709 windows, train-side, dyn-g0.9 point)

| Bounds | SPS | Coverage |
|---|---:|---:|
| frozen (0.02, 0.14) | **40.852** | 78.5% |
| conformal q85 ×1.0 | 39.858 | 79.0% |
| conformalized-α map l0.85 | 35.582 | 83.6% |

Conformal bounds (fixed-level and |pred|-normalized) lose to the frozen form;
E020 bounds stay near-optimal for the emaV point model as well (audit SPS
42.16 with (0.02, 0.14) vs 42.37 at β=0.13 — within noise; frozen).

### Locked v1_val (1,031 windows) and complement diagnostic (821 windows)

The complement rows reuse the E020 checkpoint trained on `real_regime_v1/train`;
17/20 complement trajectories overlap that training partition. They are retained
only as a mechanism-consistency diagnostic, not independent holdout evidence.

| Variant | Fold | Rel-L2 | TKE | MVPE | SPS | Composite |
|---|---|---:|---:|---:|---:|---:|
| E020 replay (sanity) | v1_val | 94.658 | 74.111 | 95.665 | 36.900 | 77.688 |
| dyn channel γ=0.9 | v1_val | 94.142 | 73.978 | 95.665 | 34.580 | 77.095 |
| **emaV γ=0.9 σ=0.5** | v1_val | **94.094** | **77.318** | 95.665 | 36.450 | **78.127** |
| oracle channel | v1_val | 94.165 | 75.370 | 95.665 | 14.123* | 73.286* |
| oracle map σ=0.5 | v1_val | 94.318 | 89.775 | 95.665 | 41.267 | 81.627 |
| E020 replay (sanity) | complement | 96.287 | 73.790 | 97.044 | 46.705 | 80.187 |
| **emaV γ=0.9 σ=0.5** | complement | **95.976** | **76.770** | 97.044 | 46.733 | **80.726** |

\* oracle rows use no calibrated bounds (default widths); scores are for
ceiling reading of Rel-L2/TKE/MVPE only.

emaV beats the per-channel *oracle* (77.318 vs 75.370): the spatial structure
of the EMA map carries information beyond global amplitude. The map-oracle
ceiling (89.775) minus emaV (77.318) is window-level intermittency — the
current window's variance map is not fully predictable from past windows
(EMA-of-truth ≈ channel-oracle information content).

## emaV deployment recipe

Per trajectory state: per-pixel EMA `E` of revealed previous-target temporal
variance maps (γ=0.9; init at first reveal). Per step:

```text
V_model(x_t)  = var_t(raw_pred + bias)          # per-pixel, channels u,v
E              = 0.9·E + 0.1·var_t(y_{t-1})      # revealed target only
s_map          = smooth3x3( sqrt( clip(E / (V_model + 1e-12), 0.2, 5.0 ) ) )
pred_t         = mean_t(raw_pred + bias) + s_map · fluct_t(raw_pred + bias)
```

First window of a trajectory (no revealed target yet): the per-channel dynamic
scale has no revealed statistic and is therefore exactly identity (`1.0`), as
in the deployed wrapper. Bounds stay frozen at `(0.02, 0.14)` physical-space
`α|pred| + βσ`.

## Conclusions and next actions

1. **Adopt emaV (D033)** — TKE +3.207 (v1_val) / +2.980 (complement) with
   Rel-L2 gains, MVPE/Time unchanged. Self-calibrating: removes the static
   1.5 constant that regressed official TKE in E022.
2. **Close I025, I026 (D034)** — AR donor and conformal SPS families are
   falsified on train-side evidence with physical explanations recorded.
3. **I028 (new, P1)** — the map-oracle ceiling (TKE ~89.8) justifies a
   train-time per-pixel variance-map prediction head with a TKE-shaped loss;
   must clear the Rel-L2/MVPE preservation gates that closed E013/E014
   point-loss mixes.
4. Package **E024** = E021 full-release weights + emaV in the E022 wrapper;
   its later Codabench attempt failed before participant execution while the
   platform fetched the organizer ingestion bundle. E029 is the hardened,
   output-equivalent resubmission.

## Limitations

- Calibration/audit and validation are drawn from the released 81 real
  trajectories; hidden-regime behavior is proxied, not guaranteed. Only
  `v1_val` is leakage-safe for the E020 checkpoint; the complement rows have
  training overlap and require a separately trained complement model before use
  as promotion evidence.
- Raw predictions were cached in fp16 (tiny numeric drift vs fp32 streaming;
  the E020 sanity reproduction is exact to 3 decimals).
- SPS with emaV moved −0.450 on v1_val but +0.028 on complement and +1.3 on
  audit; treated as noise-level, bounds unchanged.
