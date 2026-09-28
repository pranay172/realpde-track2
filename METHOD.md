# Method — Track 2

## Streaming task and integrity

Track 2 evaluates an ordered stream. A previous target may update state only
for subsequent predictions; current targets cannot influence their own forecast.
Every trajectory resets model-side mutable state, cached predictions, correction
statistics and failure counters. No cross-trajectory or cross-window retrieval
is used.

## Best officially scored recipe: E033

The point predictor is an organizer-derived FNO with separate temporal-mean and
fluctuation projection heads. The mean head is averaged over forecast time;
the fluctuation head is centered over that time axis. Their sum is the forecast.
Conversion from the official single-head simulation checkpoint copies its final
projection to both heads so the initial combined output preserves the original
mapping. E031 added strict checkpoint conversion/tensor-equality verification.

**Training:** all 81 valid real trajectories, excluding `7575_0.h5`; seed 0,
batch 4, float32, Adam 3e-4, cosine decay, 1800 updates. Uniform trajectory sampling
with window starts on a stride-20 grid prevents long trajectories from dominating.
The loss is per-window physical u/v relative MSE. Official normalization remains
unchanged. Controlled two-fold/two-seed experiments selected this recipe.

**Streaming inference:** the E029 wrapper is reused unchanged with E033 weights.

1. Use the newly revealed previous target with the matching cached previous
   forecast to update a temporal-mean bias EMA (decay 0.90).
2. Update a per-pixel target temporal-variance EMA (decay 0.90).
3. Produce the current FNO forecast, add the bias correction, and rescale
   centered fluctuations using a bounded, spatially smoothed variance ratio.
   The smoothing uses the sigma-0.5 three-tap separable kernel.
4. Use finite-value guards and a defined fallback if variance adaptation fails.
5. Return symmetric intervals with half-width
   `0.020*abs(prediction)+0.140*SIGMA_GLOBAL`; pressure remains zero.
6. Reset all adaptation state at trajectory boundaries.

This successful adaptation is statistic-based, not an online gradient optimizer
or LLM agent. Gradient-based TTT variants were studied but not selected.

## Controls and evidence

- Point model: `src/realpde_t2/dual_head_fno.py`.
- Research adaptation: `src/realpde_t2/streaming_state.py`.
- Deployed wrapper: `submission/e029/submission.py`.
- Full-data config: `configs/experiments/e033_full_real_sim_long.json`.
- Package config: `configs/experiments/e033_submission.json`.
- Detailed evidence: [E031](docs/E031_RESULTS.md), [E032](docs/E032_RESULTS.md),
  [E033](docs/E033_RESULTS.md).

E032 measured a small TKE tradeoff against matched short-budget controls.
E033's server improvement combines initialization, sampling and budget changes;
it cannot attribute the gain to any one intervention.

## Provenance caveats

Historical packed-checkpoint loading had initialization problems. The E021
record was corrected to random initialization; E020's exact history remains
qualified by contradictory checkpoint evidence. They must not be presented as
clean simulation-initialized controls. E031/E033 use verified initialization.

Full-data evaluations are contaminated as generalization evidence. Example-data
container smoke scores establish interface behavior, not leaderboard quality.
No pretrained E033 checkpoint remains after cleanup.
