# E033 — Full-release verified-initialization, longer-training candidate

Status: officially scored — new baseline, final **77.615784**.
Scores supplied by the user on 2026-09-21; submission was performed by the user.

## Basis and fixed controls

E031 accepted verified simulation initialization in fresh two-fold/two-seed
comparisons. E032 then accepted trajectory-balanced stride20 sampling with
1,800 Adam updates, confirmed with a second seed on both folds. Its mean
effect versus matched 600-update controls was +0.274878 Rel-L2, −0.129644 TKE,
+0.156858 MVPE, +0.927519 SPS. See the corresponding result notes for controls,
rejected dense sampling, preservation limits, and limitations.

E033 refits that recipe on all 81 valid real trajectories: seed 0, batch 4,
3e-4 learning rate, cosine decay, physical relative u/v MSE, float32, fixed
final update, and unchanged official normalization. The expected grid is
3,341 windows; the invalid duplicate `train_real/7575_0.h5` is excluded.
The optimizer/loss/sampling controls were checked against the selected E032
resolved configuration before running.

No released-real split remains leakage-safe for this full-data model. Do not
report local evaluations of these trajectories as unseen quality evidence.
The E032 split models establish recipe evidence; the official server must
establish the candidate's generalization and final score. **>80 is a target,
not a result inferred from local components.**

## Provenance

- Registration commit: `90f11bd`; training began only after the E032 seed-1
  report selected `stride20_u1800` as confirmed.
- Config: `configs/experiments/e033_full_real_sim_long.json`, SHA-256
  `85cd3264f68887fc504e3d4ae30ad5125ea90195854054d6aaf1ee02e6af6e34`.
- Official simulation initializer:
  `../RealPDE-Competition-Data/baseline_checkpoints/sim_pretrain/sim_fno_fp16.pth`,
  SHA-256 `46cedd9a0bf7802b73253b4bcfdfb2855aef600d7a4b4a4f1755abe591c2eb7d`.
- Inference wrapper remains `submission/e029/submission.py`, SHA-256
  `d959da0294349e3969f2b5631f6a504213d67f904ee538582e432323ed4ca8a3`.
  Mean-bias EMA, emaV, first-window behavior, reset, fallback, and calibrated
  bounds remain unchanged; only model weights differ from E029.

Training command:

```bash
../.venv-gpu/bin/python scripts/train_dual_head_fno.py --config configs/experiments/e033_full_real_sim_long.json --skip-eval
```

## Completed training

- 81 unique files exactly match the all-valid manifest; 3,341 grid windows,
  1,800 updates, and all losses/packed tensors finite.
- Strict single-head conversion passed, with no missing/unexpected keys and
  exact loaded-tensor equality. All four saved BatchNorm counters are 6,800
  (5,000 inherited simulation updates + 1,800 real updates).
- RTX 5050, Python 3.10.20, Torch 2.7.1+cu128; 50,358,342 trainable parameters,
  float32; training 326.64 seconds, peak CUDA allocation 2,647,842,304 bytes.
- Loss: first 0.108360; final 0.005622; trailing-50 mean 0.007771.
- Recorded training HEAD: `cbbf4b5`; FNO architecture, loss, and optimizer
  implementation were unchanged across the compared cells. Initialization,
  sampling, and budget varied only as specified in the registered designs.
- Packed checkpoint SHA-256:
  `2e7431c3e24587d5ad94ef6335908cbfb5ba034f86bccb0ec9566a67ddd81080`.

## Exact submission artifact and gates

Candidate: `artifacts/e033/e033_candidate.zip`.

- Archive bytes: **186,147,375**; extracted bytes: **201,433,697** (<256,000,000).
- Archive SHA-256:
  `50a5b3dc2550b4129eabfe917085e98aedceb0224188c21e269856ae60b6ba93`.
- Package config: `configs/experiments/e033_submission.json`, SHA-256
  `66c717bf0ecf22c0abe4800c8a93dfa6937c1e3e4fb5c7a77f3ca605c4764f05`.
- Ten expected root/dependency files; root `submission.py`; unchanged E029
  wrapper hash; no training data, logs, caches, or additional weights packaged.
- Evaluator-compatible host local_eval: passed, four steps over two trajectories.
- Reset after a revealed-target update: prediction and both bounds replay
  bitwise identically; all checked outputs finite, correctly shaped and ordered.
- Pinned official-container **CPU** local_eval: passed, four steps over two
  trajectories; process wall time 6.08 seconds. Image:
  `w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972`.
  Image pulls forbidden, network disabled, read-only filesystem and mounts.
- Exact extracted-package vs research-predictor parity: six real windows over
  two trajectories, **zero** maximum physical prediction/lower/upper difference,
  zero emaV fallbacks, evaluator-compatible Torch 2.2.2+cu121 on CPU.
- Full evaluator-compatible regression suite: **108/108 passed**; compilation
  and whitespace checks passed.

Container example scores (Rel-L2 81.248, TKE 73.654, MVPE 85.702, SPS 8.237)
are example-data smoke outputs, **not holdout or leaderboard estimates**.
CPU step time (573 ms in that smoke) is not target-GPU runtime. Official CUDA
execution remains untested locally on the old evaluator stack; the wrapper's
operation graph is unchanged from successfully scored E029, and E032 CUDA
evaluations took approximately 16–18 ms per step on the development GPU.

Reproduction after training:

```bash
../.venv/bin/python scripts/package_submission.py --config configs/experiments/e033_submission.json --skip-full-eval --device cpu
../.venv/bin/python scripts/check_deployed_parity.py --submission artifacts/e033/extracted --output artifacts/e033/parity_report.json
```

Retain `artifacts/e033/training_report.json`, `report.json`, `parity_report.json`,
the packed checkpoint, exact archive, and extracted validation copy. E029's
archive was rehashed and still matches `bdf06b5a…ca08c4`; it was not replaced.

## Official result and decision

The user reported the following Codabench scores. These are recorded as
user-supplied official results, not independently retrieved server logs.

| Metric | E029 | E033 | Delta |
|---|---:|---:|---:|
| Rel-L2 | 93.195166 | 93.614317 | +0.419151 |
| TKE | 75.031242 | 77.141376 | +2.110134 |
| MVPE | 92.971467 | 93.391126 | +0.419659 |
| Time | 88.743410 | 88.185671 | −0.557739 |
| SPS | 30.378171 | 32.109079 | +1.730908 |
| **Final** | **76.760173** | **77.615784** | **+0.855611** |

Promote E033 to the official baseline. All four quality components improved;
the final gain confirms that the combined new training recipe transferred to
the hidden development stream. This comparison changes initialization,
sampling and training budget together, so it cannot attribute the official
gain to any one factor. The local E031/E032 studies supply those controlled
comparisons. In particular, the official TKE gain is not evidence that longer
training alone improved TKE: E032 measured a small negative TKE budget effect.

The Time score declined despite an unchanged inference graph. A single server
run cannot distinguish platform/runtime variation from a reproducible slowdown;
retain the measured decline rather than assuming a cause. The SPS gain came
with the same interval rule, not a newly learned uncertainty model.

The remaining gap to 80 is **2.384216 final-score points**. No linear projection
from component gains is justified because the official aggregation is hidden.
Preserve E033's exact archive/checkpoint and E029 as a historical comparator.

Next priority remains feature-conditioned uncertainty: freeze the accepted
point predictor and streaming adaptation, train on properly nested train-only
out-of-fold residuals, and test interval-only changes on the preserved outer
folds. Require unchanged point metrics, improved held-out SPS and bounded
runtime overhead. Optimize the competition's coverage/tightness tradeoff,
not nominal coverage alone. No new experiment was launched while recording
these scores. Longer observed history remains a subsequent point-model route.
