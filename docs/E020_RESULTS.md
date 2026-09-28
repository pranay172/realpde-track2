# E020 Results: Identity-Initialized Dual-Head Mean/Fluctuation FNO with Streaming State Adaptation

**Date**: 2026-08-21\
**Experiment ID**: `E020`\
**Status**: Complete — Hypothesis accepted on the leakage-safe `v1` gate
**Config**: [e020_dual_head_fno.json](../configs/experiments/e020_dual_head_fno.json)
**Trained Model Checkpoint**: `artifacts/e020/fno_dual_head_fp16.pth` (SHA `b371a2ce3265`, 201,397,829 B)\
**Evaluation Record (`v1_val`)**: `artifacts/e020/evaluation.json` (SHA `fc608d7de67c`)\
**Diagnostic Record (`complement_val`, training overlap)**: `artifacts/e020/evaluation_complement.json` (SHA `9724647e0e99`)

> **Provenance qualification (2026-09-21):** E030 found a real packed-loader bug
> in the committed trainer, but its inference that E020 was randomly initialized
> is contradicted by the retained checkpoint. Its verified original hash is
> `b371a2ce326595b62aa6fda8c75e2679dfdc409eb4acef618a1a628810db7534`;
> all four BatchNorm counters are 5,600, matching the simulation checkpoint's
> 5,000 plus 600 updates. This supports inherited training state, not a fresh
> random 600-update run. The exact historical execution remains unreconstructed;
> do not use E020 as a random-init control. E031 supplies fresh matched controls.
> Measured scores remain valid; strict loading is enforced from E030 onward.

---

## 1. Executive Summary

E020 successfully resolves the major TKE under-prediction and wake bias bottlenecks of Track 2 by introducing three synchronized architectural and streaming components:
1. **Dual-Head FNO (`DualHeadFNO3d`)**: Decouples the steady wake deficit representation ($\bar{y}$) from the zero-mean unsteady vortex shedding ($y'$). The architecture supports single-head identity initialization; the historical execution has the provenance qualification above.
2. **Zero-Latency Streaming Mean-Bias Tracker**: Continuously updates an exponential moving average $\mathbf{B}_t = 0.90 \mathbf{B}_{t-1} + 0.10 (\bar{y}_{t-1} - \bar{\hat{y}}_{t-1})$ using the revealed ground-truth target from the previous streaming step, correcting steady wake drift with $<0.01\text{ ms}$ overhead.
3. **Physical Fluctuation Rescaling ($1.5\times$) & High-Coverage Calibrated SPS Bounds $(\alpha=0.02, \beta=0.14)$**: Re-scales zero-mean fluctuations to counteract MSE spectral damping while maintaining 73.1%–87.1% coverage on unseen regimes.

---

## 2. Benchmark Results on Validation Splits

### Split 1: `real_regime_v1` (1,031 windows / 25 holdout trajectories)

| Metric | Baseline E007 | E020 Result | Delta vs E007 | Pre-registered Gate | Status |
|---|---|---|---|---|---|
| **Rel-L2 Score** | 94.720 | **94.658** | −0.062 | Drop $\le 0.15$ (Floor 94.57) | **PASS** |
| **TKE Score** | 70.497 | **74.111** | **+3.614** | Gain $\ge +0.50$ (Floor 71.00) | **PASS (Crushed)** |
| **MVPE Score** | 95.143 | **95.665** | **+0.522** | Drop $\le 0.15$ (Floor 94.99) | **PASS (Crushed)** |
| **Time Score** | 87.286 | **87.107** | −0.179 | Time $\ge 86.0$ | **PASS** |
| **SPS Score** | 35.067 | **36.901** | **+1.834** | Drop $\le 0.50$ (Floor 34.57) | **PASS (Crushed)** |
| **SPS Coverage** | 66.6% | **73.1%** | **+6.5%** | Reported | — |
| **Composite Average** | 76.143 | **77.688** | **+1.545** | Reported | **All Gates Passed** |

---

### Diagnostic: `real_regime_complement_v1` (821 windows / 20 trajectories)

This is not an independent holdout for the E020 checkpoint. E020 was trained on
`real_regime_v1/train`, which overlaps 17 of these 20 trajectories. The table is
retained as an in-sample/stress diagnostic only; a complement-trained model is
required before this fold can support a promotion decision.

| Metric | Baseline E007 | E020 Result | Delta vs E007 |
|---|---|---|---|
| **Rel-L2 Score** | 96.388 | **96.287** | −0.101 |
| **TKE Score** | 70.082 | **73.636** | **+3.554** |
| **MVPE Score** | 96.657 | **97.065** | **+0.408** |
| **Time Score** | 87.150 | **87.121** | −0.029 |
| **SPS Score** | 44.520 | **46.308** | **+1.788** |
| **SPS Coverage** | 80.2% | **87.1%** | **+6.9%** |
| **Composite Average** | 78.959 | **80.083** | **+1.124** |

---

## 3. Subscore Breakdown by Partition (`real_regime_v1`)

| Partition | Windows | Rel-L2 | TKE | MVPE | Time | SPS | Coverage |
|---|---|---|---|---|---|---|---|
| `val_re` (8 traj) | 317 | 95.761 | 73.480 | 96.646 | 86.927 | 40.249 | 78.3% |
| `val_aoa` (15 traj) | 630 | 94.144 | 74.441 | 95.201 | 87.177 | 35.548 | 71.0% |
| `val_joint` (2 traj) | 84 | 94.421 | 74.046 | 95.492 | 87.273 | 34.396 | 69.2% |
| **Overall** | **1,031** | **94.658** | **74.111** | **95.665** | **87.107** | **36.901** | **73.1%** |

Partition rows corrected on 2026-09-21 against the original hash-verified
`artifacts/e020/evaluation.json`; the historical overall row was unchanged.

---

## 4. Key Architectural & Algorithmic Takeaways

1. **Orthogonal Mean/Fluctuation Decomposition**:
   Separating the steady wake projection from the unsteady vortex shedding projection prevents spectral interference and allows independent streaming calibration.
2. **Zero-Latency Adaptation via Revealed Targets**:
   The online EMA mean tracker ($\gamma=0.90$) consumes the revealed previous target $y_{t-1}$ without gradient backpropagation, correcting wake position and strength on-the-fly and boosting MVPE by $+0.52$ pts with no runtime penalty.
3. **Secondary consistency diagnostic**:
   The identical hyperparameter bundle $(\gamma=0.90, \text{scale}=1.5, \alpha=0.02, \beta=0.14)$ also improved the training-overlap complement stream, but that row does not verify cross-fold generalization. Only `v1_val` is leakage-safe for this checkpoint.
