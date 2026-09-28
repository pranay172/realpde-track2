# E027 — Joint sim+real fine-tune (I031): negative

Date: 2026-08-22. Status: **Complete — negative; complement fold not opened (D038).**

## Hypothesis

The 100 released simulation trajectories are legal training data used only as
the *initialization* in the E004→E020 lineage. Jointly fine-tuning the E020
dual-head FNO on mixed batches (real weight 1.0, sim weight 0.3, sim
restricted to the 72 `strict_regime_train` trajectories disjoint from the
validation regimes) with the otherwise-unchanged E020 recipe would improve
unseen-regime generalization. Gates: v1 parity with E024-local (Rel-L2 ≥
93.944, TKE ≥ 77.018, MVPE ≥ 95.515) AND either a +0.30 point-component gain
or complement composite ≥ 80.93.

## Setup

- `scripts/train_joint_sim_real.py`, config
  `configs/experiments/e027_joint_sim_real.json`; init E020 packed SHA
  `b371a2ce3265`; 600 updates, batch 4+4, sequential per-source backward
  (gradient-equivalent, halves peak memory), Adam 3e-4 + cosine, seed 0;
  194.1 s on RTX 5050. Loss = physical relative MSE per source.
- Evaluation: deployed emaV streaming predictor on the locked `v1_val` stream
  (1,031 windows).

## Result

| Model | Rel-L2 | TKE | MVPE | SPS | Coverage |
|---|---:|---:|---:|---:|---:|
| E024-local (E020 + emaV) | 94.094 | 77.318 | 95.665 | 36.450 | 71.9% |
| E027 (joint sim+real + emaV) | 93.919 | **67.478** | 95.094 | 34.142 | 71.4% |

All three parity gates missed; TKE collapses by −9.8. The simulation's
cleaner fluctuation statistics pull the dual-head fluctuation representation
toward the sim distribution even at weight 0.3 — the reverse of Toma's
"source quality" principle: here the added source is *mismatched* in exactly
the statistic our focal metric scores. The complement fold was not opened
(parity failed; no decision could change).

## Conclusion (D038)

Close I031. Positive-weight simulation batches are rejected on this
architecture unless paired with an explicit domain-alignment mechanism (e.g.,
sim-only auxiliary heads or distribution-matched losses), which would be a new
hypothesis requiring its own evidence. Simulation data remains useful as
pretraining initialization only (E004 lineage).

With E025 (variance head), E026 (bounds/mean mechanisms), and E027 (data
mixing) all negative, the local-evidence frontier on the E020/E021 base
stands at **E024: composite 78.127 v1 / 80.726 complement**. The next
information gain is the official E024 leaderboard row, which tests whether
emaV's self-calibration transfers to hidden regimes.
