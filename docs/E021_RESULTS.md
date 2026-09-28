# E021 Results: Full-Release Dual-Head FNO Training

**Date**: 2026-08-21\
**Experiment ID**: `E021`\
**Status**: Complete\
**Config**: [e021_full_real_dual_head.json](../configs/experiments/e021_full_real_dual_head.json)
**Trained Model Checkpoint**: `artifacts/e021/fno_dual_head_fp16.pth` (SHA `3d9a6b93b8634db08b303d220a6a5df48ef35beb84628d8456d8fddc6d903dfe`, 201,398,125 B)\

> **Provenance correction (E030 audit, 2026-08-23):** the historical dual-head
> trainer did not unpack the configured E004 `state_fp16` checkpoint and
> silently left E021 at seeded random initialization. The resulting checkpoint
> and all measured E022/E029 scores remain valid, but E021 was a full-data
> random-initialized training run rather than an E004-initialized transfer run.
> Strict packed-checkpoint initialization is enforced from E030 onward.

---

## 1. Summary

E021 scales the identity-initialized Dual-Head Mean/Fluctuation FNO architecture (`DualHeadFNO3d`) to all **81 valid released real trajectories** (`real_all_valid_v1` — 3,341 windows).

- **Training Controls**: 600 updates, Adam lr $3 \times 10^{-4}$ cosine annealing, batch size 4 (2,400 examples seen), per-window physical relative MSE loss on scored $u, v$ channels.
- **Initialization**: Actual seeded random initialization. The configured E004
  packed checkpoint was not unpacked by the historical trainer; see the
  provenance correction above.
- **Hardware & Runtime**: RTX 5050 Laptop GPU (`.venv-gpu`), completed in 106.75s.
- **Loss Progression**: Step 1: 1.948 $\to$ Step 300: 0.0100 $\to$ Step 600: 0.00666.
- **Packed Size**: 201.40 MB (well below 256 MB limit).
- **Validation Policy**: No local validation scored as unseen because all 81 valid real trajectories were used for training.
