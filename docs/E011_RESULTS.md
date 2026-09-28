# E011 v/MVPE-aware loss on the E004 recipe

Run: **2026-08-16 (IST)**

E011 repeats the E004 fine-tune with one isolated change: the joint physical
`u/v` relative MSE is replaced by
`0.5 * channel_balanced_physical_relative_mse + 0.5 * official_mvpe_probe_relative_mse`.
Every other control is held fixed. Evaluation uses the frozen E006 interval.
Ignored artifacts live under `artifacts/e011/`. The packed checkpoint SHA-256
is `df54ea7b90b48530f2b186f5becf3b5215c96b23ac91f9e2137a21b174d1f4b5`.
The evaluation JSON SHA-256 is
`288e74a756cd69c6c3032067cd31c42661f32ff6e001a578775d7dfb4f38d5b9`.

## Hypothesis and decision

The isolated loss change will raise Rel-L2 **or** MVPE by ≥0.30 versus E004,
keep the other within 0.25 of E004, keep TKE ≥70.247, and keep Time ≥86.

| Gate | Threshold | E011 | Result |
|---|---:|---:|---|
| Rel-L2 vs E004 | ≥ 95.020 (+0.30) **or** drop ≤0.25 | 94.256 (−0.464) | **fail** |
| MVPE vs E004 | ≥ 95.443 (+0.30) **or** drop ≤0.25 | 94.686 (−0.457) | **fail** |
| TKE floor | ≥ 70.247 | 71.997 | pass |
| Time floor | ≥ 86.000 | 88.021 | pass |

Hypothesis rejected. Both Rel-L2 and MVPE moved the wrong way. TKE rose 1.500,
similar in size to E009's longer-budget TKE gain.

## Setup

- Controls held equal to E004: sim-pretrain FNO init, official `mean_std_real`,
  56 `real_regime_v1/train` trajectories / 2,310 windows, Adam `3e-4`, cosine,
  batch 4, 600 updates, seed 0, no early stopping.
- Isolated change: joint physical `u/v` relative MSE → equal mix of
  per-channel physical relative MSE and official-geometry wake-probe relative
  MSE (`y = [8,10,…,24]`, `x = [13,21,29,37]`, time-mean probes, then
  per-station `||·||^2 / ||tgt||^2`).
- Eval interval: frozen E006 `(0.025, 0.1)`.
- Training 107.76 s; first/final mean-25 loss `0.19110/0.05835`. Channel
  term `0.28466 → 0.10805`; probe term `0.09754 → 0.00865`. Peak allocated
  2.463 GiB.

Config SHA-256:
`f70f4975ba330b3564d5f2d6b21489ec64bd85836a4d92edf6671dd51a609a24`.

## Overall results

Primary column is `local_bench_v1` / `v1_val`. E011 Time/SPS are the eval
harness with the same interval as E007.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | **72.697** | 94.427 | **99.631** | 13.772 |
| E004 / E007 | **94.720** | 70.497 | **95.143** | 88.013 | **35.067** |
| **E011 v/MVPE loss** | 94.256 | 71.997 | 94.686 | 88.013 | 33.886 |

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 95.317 | 71.742 | 96.398 | 88.002 | 37.783 |
| AoA-only | 93.742 | 72.152 | 93.858 | 88.027 | 32.262 |
| Joint | 94.171 | 71.806 | 94.601 | 88.052 | 31.335 |

`complement_val` is a stress column only (E011 trained on `v1` train, which
includes most of those files): E007 96.196/70.939/96.442/88.117/45.979 versus
E011 95.425/72.728/96.513/88.101/43.917.

## Interpretation and decision

1. Re-weighting `v` and adding the official wake-probe term improved TKE and
   collapsed **train** probe MSE, but official val Rel-L2 and MVPE both fell
   about 0.46. The probe term did not transfer to the scored MVPE.
2. Do not promote E011 over E007. Do not retune the 0.5/0.5 mix, and do not
   train this recipe longer, on `real_regime_v1` validation.
3. Keep E007 as the submission baseline. Next unused train-side idea is I010:
   apply the already-computed coverage-targeted interval `(0.025, 0.15)` to
   frozen E004 on `local_bench_v1`.
