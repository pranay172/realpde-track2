# Track 2 literature notes

Read and paper-text re-audited: **2026-08-20**. Sources are the referenced
`Find Relevant Research Papers` conversation, its local exports
`Research_Papers/Codex_Recs_Conv1.md` and `Codex_Recs_Conv2.md`, and every PDF
in this folder. Recommendations below were checked against the papers rather
than copied from the conversation. They are filtered through Track 2 facts we
already measured (E005/E008 TTT no-ops, residual autocorr 0.053, E013/E014
official-loss rejects, E016 official 75.497, and E017 online-SPS rejection).

**Do not treat Codex’s “LoRA TTT is very high payoff” as a next experiment.**
That family is already closed (D016). Papers are methods, not checkpoints.
External data/weights remain prohibited.

---

## How to read Codex vs this file

Codex ranked papers by *problem resemblance* (streaming adaptation, unseen Re,
TKE). That is a good search filter and a bad implementation ranking. Several
“Track 2 ★★★★★” papers are:

- training-time continual learning, not eval TTT;
- unsupervised PDE-residual adaptation that needs a known 2D equation;
- hybrid surrogate + CFD, which we do not have at inference.

Track 2 is **teacher-forced 20→20** with `prev_target` = previous GT, reset at
every trajectory, Time counting every `ttt_step`/`reset`. It is not a long
closed-loop AR rollout and not a PDE-residual problem.

Official leftover vs local E007: Rel-L2 −0.92, TKE −0.74, MVPE −3.43, SPS −8.22.
E016 recovered +0.52 MVPE by training all 81 files. Point-model Rel-L2 is not
the first-order hole.

---

## Verdict table (Track 2 only)

| Paper | Codex T2 pitch | What it actually is | Steal for T2? |
|---|---|---|---|
| CL-PDE-Surrogates (Hemati 2026) | Streaming TTT | **Train-time** continual learning + replay | Joint training supports E015. Pushforward is low relevance because evaluation never feeds predictions back. |
| Unsupervised Adaptation (Song 2026) | LoRA TTT | Offline PDE+BC fine-tune on a **new equation** | No. Known-PDE residual; NSLoRA ~+4% vs LoRA. |
| AdaLED (Kičić 2023) | Adapt-or-not controller | Hybrid **surrogate + CFD** + ensemble σ | E017 tested our residual-EMA SPS translation and rejected it. Nothing remains to implement directly. |
| GeoIncNO (Zhang 2026) | Mean/fluct decoder | New latent-increment operator; dual-decoder gated MFDR | Only the dual-head reconstruction is novel here; a split MSE alone is algebraically just reweighting. |
| Learning-Turbulent (Oommen 2025) | Spectral/GAN for TKE | adv-NO: same inference, better spatial E(k); field RMSE can rise | Diagnose spatial-spectrum bias first. Full published loss is illegal because it uses pretrained perceptual nets. |
| RECAST (Reza 2026) | Residual corrector | Coarse **solver-in-the-loop** + SHRED, 1D PDEs | No solver at eval. Residual-on-frozen-sim is Toma, not RECAST. |
| CFO (Hou 2025) | Long-horizon ODE | Flow-match ∂t, RK4 at inference | No RK4 (Time). Optional ∂t aux loss only. |
| PITI-DeepONet (Mandl 2025) | Tangent + OOD residual | Needs PDE residual / AD; RK4 | Reconstruction residual as OOD flag is legal but only useful if we adapt. |
| Temporal-NO (Diab 2025) | Multi-horizon training | New DeepONet + t-branch for AR rollout | Our fixed 20→20 call already bundles; prefix consistency is at most a low-priority train auxiliary. |
| PIANO (Nagda 2025) | Train on own rollouts | Autoregressive PINN with PDE-residual BPTT | No direct steal: the evaluator supplies the next measured input instead of consuming our output. |
| DyMixOp (Lai 2025) | Local-global mix | New architecture, convection PDEs | P3 architecture swap. Time unknown. |
| Augmented-PINN (Toma 2026) | Sim→real transfer | Stationary RANS PINN + PIV | Transfer principle only. Already our E004 path. |
| MF-DeepONet (Yang 2025) | Freeze + MergeNet | Multi-fidelity DeepONet | Frozen-sim + small real module: T1-ish point-model, not TTT. |

---

## 1. Continual learning for AR PDE surrogates — Hemati, Nguyen, Sandfeld 2026

`CL-PDE-Surrogates.pdf`

**Setup.** U-Net or FNO on 2D incompressible NS (Re 100–400) and Kolmogorov
flow, **16→16** windows. Regimes are presented **sequentially** during
training. Hidden test mixes in-dist, interpolation, and extrapolation Re.

**What works.** Their oracle is **joint training on all regimes at once**
(Table 1: U-Net joint Rel-L2 0.012 vs naive sequential 0.172). Sequential
fine-tuning forgets. Experience replay of ~10% old windows recovers most of
the joint score. **Replay-TS** stores the full history context and only a
sparse subset of rollout targets (they use 2 of 16). Pushforward training
(Brandstetter: train on the model’s own rolled-out states) is required or
replay hurts. A learned **experience-ID embedding** helps if you know which
regime you are in.

**Track 2 translation.**

- Eval is **not** a continual stream of changing Re. `reset_ttt_state` wipes
  everything at each trajectory. You cannot keep a replay buffer of earlier
  hidden files.
- We already do their oracle: E015 shuffles all 81 valid real files. That is
  joint training, not sequential CL.
- Experience-ID is unusable on the hidden set (Re/AoA unknown).
- The paper's pushforward trick is necessary because its deployed surrogate
  consumes its own prediction. Track 2 does not: every call receives a measured
  `x_t`. Self-rolled histories therefore create a train/eval mismatch rather
  than repairing one.
- Replay-TS is justified only when the temporal spectrum stays below the
  subsampled Nyquist limit. The paper's dispersive counterexample aliases and
  fails. Do not temporally slice a wake/TKE training signal without first
  measuring its temporal energy spectrum.
- A train-calibrated perturbation of measured input histories could still be a
  legal robustness augmentation, but that is a new noise hypothesis, not a
  result established by this paper.
- **Do not steal:** eval-time replay, sequential FT, or “this paper is TTT.”

Codex misread this as the closest T2 TTT paper. It is a **training curriculum**
paper.

---

## 2. Unsupervised adaptation of PDE foundation models — Song et al. 2026

`Unsupervised-Adaptation.pdf`

**Setup.** Pretrain a neighborhood-attention Transformer on **PDEBench**
families, then adapt to a **new governing equation** with **no interior
targets**: freeze backbone, insert LoRA, minimize PDE residual + BC. They
introduce NSLoRA (Newton–Schulz orthogonalization) because vanilla LoRA
collapses rank.

**Numbers that matter.** NSLoRA vs LoRA is a few points (Darcy 0.0078 vs
0.0081). Adaptation still needs the **PDE and BCs**. Limitation: single-step
only; AR untested. Pretrain is external PDEBench.

**Track 2 translation.**

- We already have the actual previous window `y_{t-1}`. A PDE residual on 2D
  NS is the *wrong* target (Toma: 3D PIV forced into 2D NS leaves a residual;
  RealPDEBench foil is a tapered 3D NACA, PIV at z=50 mm).
- E005 (full-model 1-step SGD) and E008 (last-layer 1-step) already tested
  delayed-target adaptation. Rel-L2/MVPE moved +0.00–0.03. Residual autocorr
  0.053. Time fell. D016 stops this family.
- LoRA vs last-layer is the same mechanism with a different mask. Not a new
  hypothesis until someone shows a *different* adapt signal (not pixel MSE on
  `y_{t-1}`).
- External PDEBench pretrain is illegal.

**Do not implement NSLoRA TTT.** Codex’s “do not adapt 50 M parameters” is
already what E008 tested.

---

## 3. AdaLED — Kičić et al. 2023

`AdaLED.pdf`

**Setup.** Convolutional autoencoder + ensemble of probabilistic RNNs in
latent space. A **controller** uses the surrogate only if predicted error and
uncertainty are below thresholds; otherwise it calls a **micro-solver (CFD)**
and may retrain the surrogate online. Demonstrated on 2D cylinder Re 400–1200
(laminar/shedding), not our Re 3750–26700 PIV.

**You cannot copy the last part.** There is no CFD in the Track 2 evaluator.

**Codex flip** (“adapt vs don’t adapt” instead of “model vs solver”) only
helps Time if adaptation helps quality. We measured the opposite: always-on
SGD TTT is a quality no-op and a Time tax. Gating a no-op saves Time we
already have (official Time 89.48, gate 86).

AdaLED's uncertainty is disagreement among probabilistic latent RNN ensemble
members, not a revealed-target residual EMA. The latter was our Track 2
translation:

```text
σ_t = λ σ_{t-1} + (1-λ) |ŷ_{t-1} − y_{t-1}|
bounds_t = ŷ_t ± (α|ŷ_t| + k σ_t)
```

E017 implemented this with trajectory reset and a 72-candidate train-only grid.
It selected `(α, λ, k) = (0.025, 0.9, 0.3)` but reached validation SPS 34.205,
below E007's 35.067 and E012's 35.938. Coverage rose to 69.7%, but extra width
was penalized. **The residual-EMA interval family is closed by D027; do not
retune it on validation or Codabench.**

Do not copy the AE+RNN. Latent dim 12 and 3 RNNs will not carry a 32×64
wake.

---

## 4. GeoIncNO — Zhang et al. 2026

`GeoIncNO.pdf`

**Setup.** Encode field → predict **latent increment** δz → Active-Band
Projection (FFT, energy-quantile bands, low-rank channel mix) → residual
state update → **Mean–Fluctuation Decoupled Reconstruction** (MFDR): fuse
mean and zero-mean parts separately; phase-correct **only the fluctuation**.
Trained with MSE + geometry regularizers; pushforward length 5. Simulated
PDEBench-style PDEs (Burgers, KS, 2D NS, SWE, 3D Euler, Maxwell). They also
cite RealPDEBench as related work; the main tables are not our foil PIV.

**Ablation that matters (2D NS, Table 2).** MFDR is the largest mean-drift
drop (1.32e-3 → 1.05e-4). Full-field phase correction *raises* mean drift
again. Restricting phase repair to the fluctuation is the point.

**Track 2 translation.**

- Official TKE = Rel-L2 of `0.5 (var_t u + var_t v)` (fluctuation energy).
  Official MVPE = Rel-L2 of **time-mean** wake probes. The decomposition
  `u = ū + u'` lines up with those two scores better than pixel MSE.
- We already tried *official* Rel-L2+TKE+MVPE mixes (E013/E014). Raw TKE
  error is ~3× Rel-L2, so the TKE term dominates and Rel-L2/MVPE gates
  failed (D023). MFDR is **not** that mix: it is a decoder split, not a
  0.5/0.3/0.2 weighted official functional.
- Competition windows are teacher-forced 20 frames, not 100-step AR. GeoIncNO
  wins are mostly rollout stability. Expected official delta is smaller than
  their tables.
- **Do not port ABP / new backbone.** Pack size, Time, and kit FNO already
  decided (D009).
- The paper's MFDR is not a mean/fluctuation target loss. It decodes two
  candidate fields (updated state and latent increment), learns separate gates
  for their temporal means and zero-mean fluctuations, and applies a centered
  phase correction only to the fluctuation.
- For one prediction error `e`, temporal orthogonality gives
  `Σ_t ||e_t||² = T||mean_t(e)||² + Σ_t ||e_t-mean_t(e)||²`. A plain
  mean/fluct split with equal weights is exactly the field MSE; unequal weights
  merely reweight it and collide conceptually with the E013/E014 lesson.
- **Next isolated test (supported by E019):** a genuinely dual-head, separately
  gated E004 decoder, without ABP or phase correction. E019 found centered
  fluctuation SSE shares of 0.666/0.670 and full-domain TKE-map amplitude
  ratios of 0.216/0.215 on the train calibration/audit partitions. It must pass
  the same Rel-L2/MVPE gates as E014 plus pack-size and Time gates. A post-split
  loss alone is not a paper-faithful or sufficiently distinct experiment.

---

## 5. Learning turbulent flows with generative models — Oommen et al. 2025

`Learning-Turbulent.pdf`

**Claim.** L2-trained operators oversmooth. Energy spectrum E(k) is
power-law; MSE is dominated by low-k. A model can look fine on Rel-L2 and
erase fine scales.

**Evidence.** Adv-NO (operator + discriminator **only in training**):

- Super-res: field nRMSE 0.054 → 0.066 (worse), spectrum error 0.160 → 0.011
  (15×). Same inference time.
- HIT forecast (160 frames, one trajectory): field 0.033 → 0.027, spectrum
  0.087 → 0.024. Same 0.082 s. Divergence penalty did **not** fix spectral
  bias.
- Diffusion is 100× slower. Do not ship a sampler.

**Track 2.** Persistence still leads local TKE (72.70 vs E007 70.50); official
TKE is 69.75. However, the paper measures a **spatial radial energy spectrum**,
while RealPDE scores the spatial field of **temporal** fluctuation energy
`0.5(var_t u + var_t v)`. Parseval links total energy, but better spatial
high-wavenumber fidelity does not guarantee better pointwise temporal variance.

Before reserving a run, measure on train/audit windows:

1. bandwise spatial-spectrum bias of E004, especially on `v` and the wake;
2. correlation between per-window log-spectrum error and official TKE error;
3. whether the deficit is consistently high-frequency rather than phase-only.

Only if that diagnostic supports the mechanism should a single train-scaled
binned log-spectrum auxiliary be tested with a pre-registered Rel-L2 floor.

**E019 result (2026-08-21): do not add that auxiliary from frozen-E004
evidence.** Both partitions have severe full-domain high-band underpower
(prediction/target 0.160/0.158), but the required association with the
organizer's per-window temporal-TKE error is opposite-sign (Spearman
−0.474/−0.337); audit low-minus-high also misses the locked 0.10 gap. Spatial
energy deficit is therefore not a useful per-window handle on this temporal
variance metric here.

Caveat they state and Table 1 confirms: recovering high-k can **raise**
pointwise field error (phase). Any such run needs a Rel-L2 floor, not “TKE
went up so we submit.”

The published adv-NO objective is not directly competition-legal: alongside L1
and adversarial terms it uses perceptual features from pretrained VGG-19 (2D)
or Med3D ResNet10 (3D). External pretrained weights are prohibited even if the
feature network would not ship. A legal adversarial variant must train every
component from scratch on released data and omit pretrained perceptual loss;
that is less directly supported and less reproducible than the diagnostic-gated
spectral auxiliary above.

Physics-informed div(u)=0 did not fix spectra. Do not add a 2D continuity
PINN on this PIV.

---

## 6. RECAST — Reza & Faraji 2026

`RECAST.pdf`

Learned correction **inside a coarse numerical integrator**, plus a second
net that super-resolves. 1D PDEs, SHRED/LSTM, 1000-step closed loop. The
coarse solver `S_c` runs at every step.

No coarse CFD in Track 2 eval. Persistence is not a coarse NS solver.
“Ŷ = F_frozen_sim + C_φ(X)” is Toma / MF-DeepONet, not RECAST. Its
solver-in-the-loop rollout solves the same self-fed deployment problem that is
absent from Track 2's measured-input stream.

Skip as an architecture.

---

## 7. CFO and PITI-DeepONet

`CFO.pdf`, `PITI-DeepONet.pdf`

Both learn **∂t** and integrate. CFO: spline + flow-matching, no ODE
backprop in train; RK4 at inference. PITI: DeepONet outputs reconstructed
state + tangent; optional PDE residual; reconstruction residual
`R = ||u^n − û^n||` as OOD flag (they **do not** change inference on R).

Track 2 already outputs 20 frames in one forward. RK4 is 4× forwards per
step → Time death. PDE residual needs a trustworthy 2D operator we do not
have.

Optional: auxiliary `||∂t Ŷ − Δ_t Y||` during train only. Low expected
official move on teacher-forced 20-frame windows. PITI’s reconstruction
residual as a TTT trigger is useless until TTT itself works.

---

## 8. Temporal-NO and PIANO

`Temporal-NO.pdf`, `PIANO.pdf`

TNO: DeepONet + temporal branch, **temporal bundling** (predict K future
states at once), then feed the last predicted bundle state into another AR
call. Our 20→20 call already bundles, but the evaluator does not perform that
second self-fed call. The paper also warns that predicting all available steps
at once can weaken learned temporal dynamics. A shared prefix-consistency head
could be explored eventually, but 5→5/10→10 training is not automatically
compatible with the fixed-channel kit FNO and is not current priority.

PIANO: AR PINN, backprop through its own pointwise rollout, supervised mainly by
PDE and boundary residuals. Its benefit specifically addresses a model that
consumes its previous predictions. Track 2 consumes new measured histories, so
neither the PINN nor its self-rollout curriculum directly matches deployment.

---

## 9. DyMixOp — Lai, Chen, Xu 2025

`DyMixOp.pdf`

Local-global mixing for convection-dominated PDEs. Paper-level architecture
swap. Kit FNO already has global Fourier modes; CNO already has local
filters and lost on Time (E003, 64 ms, Time 77). Do not start here.

---

## 10. Toma PINN and Yang MF-DeepONet (Track 1 papers, T2-relevant principle)

`Augmented-PINN.pdf`, DeepONet / MF notes in Codex.

Toma: NACA0012 PIV + RANS PINN. Transfer beats real-from-scratch. **Source
representation quality mattered more than matching Re.** 2D PINN fails where
the experiment is 3D. Do not copy the PINN. Do not force the net to memorize
every PIV artifact.

Yang: freeze branch/trunk, adapt MergeNet on high-fidelity; physics-guided
subsampling on |∂t u|. We should not throw away train_real. Reinterpret as
**loss weight** on high |∂t ω| / wake — related to E011 probe weighting,
which failed on v1 when mixed 0.5/0.5 with channel MSE. A *mild* vorticity
weight with a Rel-L2 drop cap is the only residual of this paper for T2.

Frozen-sim + small real residual head is the Toma/Yang architecture. That is
a **point-model** alternative to full E004 fine-tune (regularize toward sim
for hidden Re). Compare on `v1` against E007, not on E015 weights.

---

## What Codex got right

1. Track 2 is delayed supervision, not ImageNet TTA.
2. Mean/fluctuation lines up with TKE vs MVPE.
3. L2 oversmooths and TKE is a real hole, although spatial-spectrum error and
   RealPDE's temporal-variance TKE are not the same metric.
4. Do not ship diffusion / RK4 / 50 M TTT.
5. SPS is under-used relative to the official −8.22 vs local.
6. Papers are ideas, not Hugging Face weights.

## What Codex got wrong for *this* repo

1. **LoRA / last-layer / error-gated TTT as priority 4–5, “very high.”**
   E005+E008 already falsified delayed-target SGD on this stream. Gating
   does not create signal. Meta-TTT (priority 9) is the same family with
   more surface area.
2. **CL-PDE and Song as T2 TTT papers.** They are train-time CL and
   unsupervised PDE adaptation.
3. **AdaLED controller as a quality tool.** Without a solver, it is a Time
   switch on a no-op.
4. **Stacking TKE+mean+spectrum+vorticity+adv in one loss.** E013/E014
   showed unscaled official mixes fail Rel-L2/MVPE. One isolated term, train
   scales, pre-registered gates.
5. **Sim-backbone + residual adapter as if we had not full-FT.** E004/E015
   already full-FT packed FNO. The open question is whether a *smaller*
   real module generalizes better than full FT, not whether to invent FT.
6. **Pushforward as an obvious Track 2 augmentation.** The evaluator never
   feeds a prediction back as the next input, so AR self-rollout methods solve a
   deployment problem this protocol does not have.
7. **Adv-NO as a drop-in legal loss.** Its published objective depends on
   externally pretrained perceptual networks, and its spatial E(k) target is
   not the official temporal-variance TKE functional.

---

## T2-useful ideas that survive contact with our ledger

Not reserved. Isolated, legal, leakage-safe on `v1` unless noted.

| Rank | Idea | Why it is still open | Collision |
|---|---|---|---|
| 1 | **Spectrum/TKE diagnostic**, then at most one train-scaled binned log-spectrum auxiliary (Oommen) | TKE hole, but applicability must be established because spatial E(k) ≠ official temporal TKE | Need positive diagnostic and Rel-L2 floor; no pretrained perceptual net |
| 2 | **Dual-head mean/fluctuation decoder** (GeoIncNO MFDR-inspired) | Architecturally separates mean and fluctuation paths; unlike a split loss, this is not algebraically redundant | Related to D023; size/Time and Rel-L2/MVPE gates |
| 3 | **Frozen sim-FNO + small real residual** (Toma/Yang) | Hidden-regime regularizer vs full FT | Compare to E007 on v1, never score E015 |
| 4 | **Train-calibrated measured-history perturbation** | Could regularize PIV noise/domain shift, but is not established by the AR papers | New hypothesis; do not call it pushforward evidence |
| — | I011 residual *feedback* on the point | Still P2; autocorr 0.053 | After a better point model |
| — | Online residual-scale SPS | Closed by E017 | D027 |
| — | LoRA / SGD / gated TTT | Closed | D016 |
| — | Official Rel-L2+TKE+MVPE mixes on v1 | Closed | D023 |
| — | CNO/U-Net/DyMixOp/CFO-RK4 | Time or pack | D009, E003 |

---

## Reading order if you only re-open PDFs

1. Oommen §Results, Table 1–2, and Eq. 8 (spectral bias, metric mismatch,
   pretrained perceptual-network constraint).
2. GeoIncNO §4.5 and Table 2 (actual dual-decoder MFDR vs full-field phase).
3. Hemati Table 1 + Replay-TS (joint training is the oracle; temporal slicing
   fails once subsampling aliases high-frequency dynamics).
4. Toma transfer discussion (source quality > matched Re; 3D residual).
5. AdaLED only for provenance of the controller/ensemble uncertainty; E017
   already closes our residual-EMA translation.

Skip implementing Song, RECAST, PITI/CFO inference, DyMixOp, and PIANO’s PINN.
