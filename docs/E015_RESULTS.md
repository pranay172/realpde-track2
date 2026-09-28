# E015 full-release E004 recipe

Run: **2026-08-16 (IST)**

E015 repeats the E004 fine-tune with one isolated change: the train set is
every valid released real trajectory (81 files / 3,341 windows), the union of
`real_regime_v1` train and validation. `train_real/7575_0.h5` stays excluded.
There is no leakage-safe local validation stream. Ignored artifacts live under
`artifacts/e015/`. Packed checkpoint SHA-256
`1e61d570bf941985d5caa0a033ada7c59a1d609c38eef918ae0058583ba73343`.

## Hypothesis and decision

The same E004 recipe on all 81 valid real files will raise official Codabench
Rel-L2 or MVPE versus E007 (93.780 / 91.190), or official `final_score` versus
75.423409, with Time still ≥86.

| Gate | Threshold | E015 | Result |
|---|---|---|---|
| Local quality on `v1_val` | not a gate | not scored | — |
| Train completes | 81 files, 3341 windows, finite loss | 81 / 3341 / yes | **pass** |
| Packed size | < 256 MB | 201,396,069 B | pass |
| Official Rel-L2 vs E007 93.780 | +≥0.30 **or** rely on MVPE/final | 93.799842 (+0.020) | fail +0.30 |
| Official MVPE vs E007 91.190 | +≥0.30 | 91.710703 (+0.521) | **pass** |
| Official `final_score` vs 75.423409 | raise | 75.496987 (+0.074) | **pass** |
| Official Time | ≥ 86 | 89.478629 | **pass** |

Local training hypothesis is accepted as a completed full-release fit. Official
quality, via the E016 zip, is a **slight** accept: MVPE and `final_score` rose;
Rel-L2/SPS did not meaningfully move. Do not treat any later `v1_val` number as
unseen.

## Setup

- Isolated change versus E004: train membership 56 → 81 valid real files
  (`configs/splits/real_all_valid_v1.json`, SHA-256
  `a4b541f24378b5cc65618f0a179dd85a8329a3e169b49652e87089ae7052925a`).
- Controls held equal: sim-pretrain FNO init, official `mean_std_real`, Adam
  `3e-4`, cosine, batch 4, 600 updates, seed 0, physical `u/v` relative MSE.
- Effective epochs 0.718 versus E004 1.039. Same 2,400 examples seen.
- Eval: skipped. `local_bench_v1` is contaminated for this model.
- Package interval if submitted later: frozen E006 `(0.025, 0.1)`.
- Training 106.94 s; first/final mean-25 loss `0.16278/0.01061` (E004 final
  mean-25 was 0.00945). Peak allocated 2.464 GiB.

Config SHA-256
`609f755bd631e90a35e018701ca104b20a7aa53a6f749ddb8e6254a507263165`.

## Interpretation and decision

1. The isolated full-release fit completed. Train loss is in the same range as
   E004 despite mixing in the former holdout regimes and seeing fewer epochs.
2. Official E016 scores accept the hypothesis slightly: adding the 25 held-out
   regimes recovered 0.52 MVPE and 0.07 `final_score` versus E007, not Rel-L2
   or SPS. Freeze that official row. Do not retune this fit to the hidden set.
3. Never score these weights on `v1_val` or `complement_val` as if those
   streams were held out. E007 stays the leakage-safe local reference.
