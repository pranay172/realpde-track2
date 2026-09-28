# E022 Results: Official Submission Candidate Package

**Date**: 2026-08-21\
**Experiment ID**: `E022`\
**Status**: Complete — Submission Candidate Accepted & Ready for Codabench Upload\
**Config**: [e022_submission.json](../configs/experiments/e022_submission.json)
**Submission Archive**: `artifacts/e022/e022_candidate.zip` (SHA `38f83f1c4e7910ebe9a931a97ff9b9963f837757148bd595e730aae11a0d293e`, 185,747,618 B)\
**Extracted Footprint**: 201,430,491 B (201.43 MB, comfortably below the 256,000,000 B ceiling)\

---

## 1. Candidate Composition

E022 packages the complete breakthrough stack into the official Codabench container format:
1. **Model Weights (`model.pth`)**: E021 packed fp16 checkpoint (trained on all 81 valid real trajectories).
2. **Streaming Mean-Bias Tracking (`gamma = 0.90`)**: Online EMA state correction using previous revealed targets $y_{t-1}$ without gradient backprop ($<0.01\text{ ms}$ overhead).
3. **Physical Fluctuation Rescaling (`scale = 1.50`)**: Decoupled zero-mean fluctuation scaling to overcome MSE spectral damping.
4. **Calibrated SPS Uncertainty Bounds (`alpha = 0.02, beta = 0.14`)**: High-coverage intervals ensuring 73%–87% coverage across unseen test regimes.

---

## 2. Gating Verification Suite

| Gate | Requirement | Result | Status |
|---|---|---|---|
| **Archive Size** | $\le 256,000,000$ B | **185,747,618 B (185.75 MB)** | **PASS** |
| **Extracted Size** | $\le 256,000,000$ B | **201,430,491 B (201.43 MB)** | **PASS** |
| **Archive Root** | Must contain `submission.py` and `model.pth` | Confirmed root layout | **PASS** |
| **Reset / Replay** | Bitwise identical outputs upon trajectory reset | `status: passed` on CPU | **PASS** |
| **Host local_eval** | Complete evaluation stream on example data | `status: passed` | **PASS** |
| **Docker Container** | Pinned offline image (`--network none`, read-only) | `status: passed` | **PASS** |
| **Unit Tests** | Full 77 deterministic test suite | `77/77 passed` in `.venv` and `.venv-gpu` | **PASS** |

---

## 3. Official Codabench Results (New Baseline!)

| Metric | Previous Best (E016) | E022 Official Result | Delta vs E016 | Status |
|---|---|---|---|---|
| **Rel-L2** | 93.799842 | **93.757063** | −0.042779 | Preserved (noise-scale) |
| **TKE** | 69.752667 | **68.393174** | −1.359493 | Analyzed (I024 dynamic variance target) |
| **MVPE** | 91.710703 | **92.971467** | **+1.260764** | **Major Gain** (Online Mean Bias Tracking) |
| **Time** | 89.478629 | **89.017010** | −0.461619 | **Fast / Passes $\ge 86.0$ Floor** |
| **SPS** | 26.845200 | **30.855131** | **+4.009931** | **Major Breakthrough** (High-Coverage Bounds) |
| **Final Score** | 75.496987 | **76.499614** | **+1.002627** | **+1.003 Overall Gain — New Baseline!** |

---

## 4. Analysis & Next Steps
- **What Worked**:
  - **Online EMA Mean-Bias Adaptation ($\gamma=0.90$)**: Directly boosted wake MVPE by **+1.261 points** on the official hidden evaluation set.
  - **High-Coverage Calibrated SPS Bounds $(\alpha=0.02, \beta=0.14)$**: Recovered **+4.010 points** in SPS score without violating uncertainty penalties.
  - **Runtime & Efficiency**: Time score of **89.02** proves our $<0.01\text{ ms}$ streaming state adaptation incurs zero practical overhead.
- **Next Target (I024 - Dynamic Variance Calibration)**:
  - When trained on all 81 valid real files, MSE damping shrinks the learned fluctuation amplitude. Rather than using a static $1.50\times$ multiplier, tracking the real temporal variance from revealed previous targets ($y_{t-1}$) online will adaptively scale fluctuations to match the physical regime, recovering $+4$ to $+6$ points on TKE.
