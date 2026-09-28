# E004 split-safe packed-FNO fine-tune

Run: **2026-08-15 (IST)**

E004 produces a leakage-safe learned FNO by fine-tuning only the 56
[`real_regime_v1`](../configs/splits/real_regime_v1.json) training trajectories.
Ignored artifacts live under `artifacts/e004/`. The packed evaluation
checkpoint is `artifacts/e004/fno_fp16.pth` (201,396,069 bytes, SHA-256
`75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c`). The evaluation JSON has SHA-256
`49acb99a0021d707485b7973bc0173604689b3fc44ec2277a12e473c9565951a`.

## Hypothesis and decision

A packed FNO initialized from the released simulation-only checkpoint and
fine-tuned with official `mean_std_real` normalization plus per-window physical
relative MSE on scored `u/v` will beat E003 persistence on Rel-L2 and MVPE, and
will not drop TKE below E003 sim-pretrain FNO.

| Gate | Threshold | E004 | Result |
|---|---:|---:|---|
| Rel-L2 vs persistence | > 92.164 | 94.720 | pass |
| MVPE vs persistence | > 94.427 | 95.143 | pass |
| TKE floor vs sim-pretrain FNO | ≥ 66.969 | 70.497 | pass |

Hypothesis accepted. Time and SPS were reported, not gated.

## Setup

- Train: 56 whole trajectories, 2,310 non-overlapping 20-to-20 windows, stride 20.
- Held out: 25 trajectories / 1,031 windows (`val_re` 317, `val_aoa` 630,
  `val_joint` 84). Training never read those field values.
- Init: `baseline_checkpoints/sim_pretrain/sim_fno_fp16.pth` SHA-256
  `46cedd9a0bf7802b73253b4bcfdfb2855aef600d7a4b4a4f1755abe591c2eb7d`.
- Normalization: official kit `mean_std_real.pt`, not recomputed train-only
  statistics, because the evaluator works in that space.
- Loss: per-window physical relative squared error on `u/v` only.
- Optimizer: Adam `3e-4`, cosine to 0, 600 fixed updates, batch 4, seed 0,
  float32, no gradient clipping, no validation-based early stopping.
- Environment: Python 3.10.20, Torch 2.7.1+cu128, RTX 5050, 2.645 GiB peak
  allocated during training. Training wall time 106.28 s; first/final mean-25
  loss `0.10436/0.00945`.
- Evaluation: packed fp16 checkpoint, no-adaptation `TorchPredictor`, official
  v6 scorer, default ±5% SPS band, synchronized reset/predict timing.

Config SHA-256:
`0f870e9c4e18d3663318702fa74685cd0976864fcc96ea1399fd18d493b5a0d6`.

## Overall results

Higher is better. Time is local-hardware.

| Model | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| E003 persistence | 92.164 | **72.697** | 94.427 | **99.510** | 13.772 |
| E003 sim-pretrain FNO | 77.876 | 66.969 | 71.700 | 87.633 | 0.790 |
| **E004 split-safe FNO** | **94.720** | 70.497 | **95.143** | 88.167 | **15.024** |
| E003 sim-real-ft FNO (leaky) | 96.348 | 78.629 | 96.956 | 87.833 | 20.614 |

Underlying aggregates for E004: Rel-L2 error 0.111477, TKE error 0.836993,
MVPE error 0.102101, SPS 0.150237, coverage 29.982%. Mean step 13.13 ms over
1,031 windows. Peak eval GPU 2.185 GiB. All outputs finite and exact-shape.

## Regime breakdown

| Partition | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---:|---:|---:|---:|---:|
| Re-only | 95.841 | 70.057 | 96.349 | 88.193 | 18.358 |
| AoA-only | 94.197 | 70.751 | 94.550 | 88.155 | 13.506 |
| Joint | 94.488 | 70.277 | 95.124 | 88.155 | 13.809 |

AoA-only remains the hardest Rel-L2/MVPE/SPS partition, matching E003.

## Interpretation and decision

1. Fine-tuning only the 56 train trajectories closes most of the sim-to-real
   gap that made every simulation-only E003 model lose to persistence.
2. Rel-L2 (+2.556) and MVPE (+0.716) beat the leakage-free persistence anchor.
   TKE improved 3.528 over sim-pretrain FNO but remains 2.200 below persistence.
3. Default SPS coverage is still only 30.0%. Calibration stays a separate
   experiment; this run did not retune intervals.
4. The packed checkpoint is 201.4 MB and loads strictly on Torch 2.2.2 CPU with
   bitwise-identical replay of a training window. It is the leakage-safe learned
   base for later TTT comparisons.
5. Released `sim_real_ft` FNO remains a diagnostic upper bound only.

Next: keep this checkpoint frozen and test lightweight test-time adaptation
against the no-adaptation E004 scores above.
