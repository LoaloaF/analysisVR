# Pending Questions

Questions that are blocking specific slides or figures. Resolve before coding those slides.

---

## Units

| Feature | Current label | Known unit | Status |
|---|---|---|---|
| Track Position | `frame_position` | cm (−169 to 270) | CONFIRMED by supervisor |
| Head Angle | `head_angle` | cm (supervisor said unity coords = cm) | CONFIRMED — verify range in data |
| Forward Speed | `frame_raw_500msMedian` | ? (probably cm/s) | PENDING |
| Forward Acceleration | `frame_raw_abs_acc_500msMedian` | ? | PENDING |
| Rotational Velocity | `frame_YawPitch_abs_vel_sum_500msMedian` | ? (degrees/s? radians/s?) | PENDING |
| Rotational Acceleration | `frame_YawPitch_abs_acc_sum_500msMedian` | ? | PENDING |
| Head Angular Velocity | `head_angle_vel` | ? (same unit as head_angle / time) | PENDING |

**Blocks:** S06 (behavioral features trace), S25 (head angle scatter y-axis)

---

## Ensemble Selection

**Status:** RESOLVED.
E18 wins: preferred head angle std = 0.980 across 24 sessions (vs E08: 1.415 across 28 sessions).
E18 consistently prefers positive head angles (mean preferred = 0.896).
**Use E18 for S26 (stability slide).**

---

## Frequency Analysis Result

**Status:** RESOLVED.
TempConv-Pred shows meaningfully higher fast-component R² than MLP (0.26 vs −0.04 on best pair; 0.16 vs 0.02 on second). TempConv-Cont intermediate. Slow-component R² similar across all models.
**S28 message:** "TempConv captures fast neural fluctuations that MLP's point-in-time architecture cannot"
**S29 message:** "TempConv-Pred outperforms MLP on fast dynamics; models are otherwise comparable on grand mean R²"
**S31 impact:** Add final bullet: "10-step temporal window enables fast-dynamics coding that point-in-time MLP misses"

---

## Citations

**Question:** What is the lab's reference for the VR task setup / recording pipeline?
**Blocks:** S02 (task slide citation)

**Question:** What paper should be cited for neural ensemble extraction via ICA in this specific context?
**Blocks:** S05 (ensemble extraction citation)

**Question:** Is there a specific place-cell or spatial tuning reference the supervisor uses?
**Blocks:** S27 (position coding slide citation)

---

## CPV Image

**Question:** Where is the user's CPV explanation image stored?
**Blocks:** S17 (conditional PV slide — user said "add the image I made")
**Action:** User to provide file path or drop image.

---

## MLP Architecture Diagram

**Status:** RESOLVED — confirmed from training/train_mlp.py and utils/models.py.
Architecture:
  Input (12 features) → Linear(12→64) + ReLU → Linear(64→64) + ReLU → Linear(64→1)
  No dropout. No weight decay. Adam lr=1e-3. MSE loss. 100 epochs.
  Trained independently per (session × ensemble × seed).
TempConv:
  10-sample window (5 before + t + 4 after) → Conv1d(k=2,GELU) → 3×SkipBlock(Conv1d k=3,GELU,residual) → Conv1d(k=3) → embedding (8-dim)
  Contrastive variant: InfoNCE loss, k=20 nearest neighbors, batch_size=128, lr=3e-4, 20 epochs.
  Predictive variant: MSE loss, same conv stack without L2 normalisation.
Diagram to generate in same box/color style as S40.
**Blocks:** S11 — unblocked, ready to generate.
