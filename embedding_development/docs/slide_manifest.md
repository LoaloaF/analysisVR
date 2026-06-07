# Slide Manifest
# Neural Task Representation — Supervisor Meeting
# Current version: ultimate_presentation_v51.pptx  (94 slides)
# Last updated: 2026-06-03

Slide dimensions: 10" × 5.625" (widescreen 16:9)
Content area: ~9.5" × 4.2"
All slide titles BOLD.  No matplotlib suptitle/title in figures unless noted.
Font floor: tick ≥ 13pt (FONT.TICK), axis labels ≥ 15pt (FONT.LABEL),
            legend ≥ 13pt (FONT.LEGEND), panel labels ≥ 16pt (FONT.PANEL).
Figure native size = PPTX placement size (no rescaling).  DPI = 200.

Model colours:  Linear=#8B6552  MLP=#2CA02C  TC-Cont=#1F77B4  TC-Pred=#FF7F0E
Cue colours:    No cue=#AAAAAA  Cue 1=#FF7F0E  Cue 2=#D62728
Nature palette: PALETTE[0..9] in utils/figure_style.py

Feature short names (FEATURE_NAMES_SHORT from utils/figure_style.py):
  frame_raw_500msMedian            → Fwd Speed
  frame_raw_abs_acc_500msMedian    → Fwd Accel.
  frame_YawPitch_abs_vel_sum…      → Rot. Vel.
  frame_YawPitch_abs_acc_sum…      → Rot. Accel.
  head_angle_vel                   → Head Ang. Vel.
  head_angle                       → Head Angle
  frame_position                   → Track Pos.
  cue_visible                      → Cue Visible
  upcoming_choice                  → Up. Choice
  reward_window                    → Reward Win.
  lick_detected                    → Lick Det.

---

## NARRATIVE SPINE

**One-sentence framing:**
"We build and validate a scalable method for characterizing P(ensemble activity | state, actions)
 and show what it reveals about behavioral representations — and where naive statistics mislead."

**Seven sections:**
1. Intro & Setup           S01–S11
2. Linear Baseline         S12–S14
3. Nonlinear Models        S15–S36
4. Feature Attribution     S37–S47
5. Embedding Consistency   S48–S52
6. Case Studies            S53–S71
7. Summary + References    S72–S73
   Hidden: Methods cards   S74–S76
   Appendix                S77–S94

---

## SECTION 1 — Intro & Setup (S01–S11)

S01  Title slide — no figure
S02  Example trial — embedded MP4
S03  Paradigm & Setup (cont.) — no figure
S04  Data Distribution — no figure
S05  Input Data Distribution: P(State Feat., Action Feat.) — no figure
S06  Input Data Features: Continued — no figure
S07  Initial idea: Map from SIPEC to Ensemble Activation — embedded diagram
S08  Recover bin order, analyze relation to behavior — embedded diagram
S09  Approach Overview — no figure
S10  Dataset Details — no figure
S11  Predictive Encoding Metrics — no figure

---

## SECTION 2 — Linear Baseline (S12–S14)

S12  Linear Encoding Models — section header
S13  Linear Model: Is Neural Signature an Affine Transformation? — no figure
S14  Linear Baseline: Within-Session Ensemble R²
     Figure:   r2_bar_linear.png
     Script:   eval/generate_r2_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  4.5" × 4.2" (FIG.HALF)

---

## SECTION 3 — Nonlinear Models (S15–S36)

S15  Nonlinearity: ML Captures Tuning That Spearman ρ Misses
     Figure:   nonlinearity_advantage.png
     Script:   eval/generate_ml_vs_naive_figures.py
     Outdir:   outputs/mlps/ml_vs_naive/

S16  ML Captures Joint Effects: No Single Feature Explains ≥5% Variance
     Figure:   joint_effect_scatter.png
     Script:   eval/gen_new_analysis_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S17  Nonlinear Encoding Models — section header

S18  MLP Outperforms Linear Regression on Nonlinear Pairs
     Figure:   mlp_vs_linear_r2.png
     Script:   eval/generate_ml_vs_naive_figures.py
     Outdir:   outputs/mlps/ml_vs_naive/

S19  MLP Architecture — embedded diagram
S20  MLP Capability Demonstration — embedded images / training frames

S21  MLP Baseline: Within-Session Ensemble R²
     Figure:   r2_bar_mlp.png
     Script:   eval/generate_r2_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.5" × 4.2" (FIG.FULL)

S22  Ablation Proof: Single-Feature MLP Predicts Where GLM Sees Nothing
     Figure:   ablation_proof.png
     Script:   eval/generate_ml_vs_naive_figures.py
     Outdir:   outputs/mlps/ml_vs_naive/

S23  Temporal Context and Contrastive Learning — embedded diagram
S24  Can Contrastive Embeddings Reveal Behavioral Structure? — embedded images

S25  TempConv-Cont and TempConv-Pred R² Per Ensemble (Sorted by MLP R²)
     Figure:   r2_bar_tempconv_comparison.png
     Script:   eval/generate_r2_figures.py
     Outdir:   outputs/cebra_comparison/

S26  Grand Mean R² Across All Four Model Architectures
     Figure:   r2_comparison_grand_mean.png
     Script:   eval/eval_cebra_compare.py
     Outdir:   outputs/cebra_comparison/

S27  R² Distribution: Claim Scope and Valid Pair Thresholds
     Figure:   r2_distribution_cdf.png
     Script:   eval/gen_new_analysis_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S28  Valid (R²≥0.05) Pair Counts per Session: MLP vs Linear
     Figure:   two_thresholds_scatter.png
     Script:   eval/eval_cebra_compare.py
     Outdir:   outputs/cebra_comparison/

S29  TempConv vs MLP: Architecture-Independent Representations
     Figure:   tempconv_delta_r2.png
     Script:   eval/gen_new_analysis_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S30  Example Predictions — section header
S31  Example Plots – Ensemble Prediction (1 of 3) — embedded trace (not regenerable)
S32  Example Plots – Ensemble Prediction (2 of 3) — embedded trace (not regenerable)
S33  Example Plots – Ensemble Prediction (3 of 3) — embedded trace (not regenerable)
S34  Frequency Encoding — section header

S35  MLP Fails on Fast Neural Fluctuations; TempConv-Pred Captures Them
     Figure:   trend_noise_comparison.png
     Script:   eval/eval_trend_noise.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.5" × 4.2"

S36  Example Plots – Ensemble Prediction  [freq trace]
     Figure:   freq_trace_example.png
     Script:   eval/eval_freq_trace_example.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.5" × 4.2"
     Note:     Best (session, ensemble) pair selected automatically by max TC-Pred
               vs MLP noise-R² gap.  Currently S01 E03, gap=+0.362.

---

## SECTION 4 — Feature Attribution: GPV / CPV / IG (S37–S47)

S37  Encoding Input Importance — section header
S38  Permutation Variance — embedded diagram + method slide

S39  GPV Heatmap: Feature Attribution by Ensemble
     Figure:   gpv_task_ensembles.png
     Script:   eval/generate_gpv_task_ensembles.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.5" × 4.2"

S40  Conditional Permutation Variance — method slide

S41  Global PV vs Conditional PV — Points Below Diagonal Are Co-variation-Inflated
     Figure:   global_vs_cond_pv_scatter.png
     Script:   eval/generate_attribution_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S42  Integrated Gradients — embedded diagram

S43  Mean |IG| Attribution by Feature Group × Ensemble
     Figure:   ig_per_ensemble_heatmap.png
     Script:   eval/generate_attribution_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S44  Group-Level Attribution Profile: MLP vs TempConv-Cont (GPV and IG)
     Figure:   group_attribution_comparison.png
     Script:   eval/generate_group_attribution_comparison.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S45  Attribution Agreement: MLP vs TempConv-Cont (GPV ρ≈0.95, IG ρ≈0.94)
     Figure:   attribution_agreement_scatter.png
     Script:   eval/generate_attribution_agreement_figure.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S46  Attribution ≠ Naive Effect Size: MLP Captures Novel Multi-Feature Variance
     Figure:   ml_vs_naive_scatter.png
     Script:   eval/generate_ml_vs_naive_figures.py
     Outdir:   outputs/mlps/ml_vs_naive/

S47  Attribution evolves session-to-session: ensemble-specific fingerprints
     Figure:   evolution_lineplot_ig.png
     Script:   eval/eval_evolution_lineplots.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.51" × 4.65"
     Note:     Loads pre-computed IG checkpoints (ig_checkpoint_seed*.npz).
               Top 7 diverse pairs by mean IG + temporal CV. IQR = across 5 seeds.

---

## SECTION 5 — Embedding Consistency (S48–S52)

S48  Embedding Consistency — section header

S49  Attribution Profile Consistency: Cross-Architecture & Cross-Ensemble
     Figure:   attribution_consistency_merged.png
     Script:   eval/gen_attribution_consistency.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.5" × 4.2"
     Note:     Right panel shows cross-seed GPV for MLP, TC-Cont, TC-Pred
               (MLP=0.927, TC-Cont=0.891, TC-Pred=0.900).

S50  Embedding Geometric Consistency: Linear Map R² Across Variation Axes
     Figure:   embedding_linear_map.png
     Script:   eval/eval_embedding_linear_map.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Note:     CEBRA-analogous metric (ridge R² from embedding_A → embedding_B).

S51  Cross-Ensemble Prediction
     Figure:   cross_ensemble_prediction.png
     Script:   eval/eval_cross_ensemble_prediction.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S52  Cross-Ensemble Attribution Consistency Within Sessions
     Figure:   cross_ensemble_consistency.png
     Script:   eval/gen_new_analysis_figures.py
     Outdir:   outputs/mlps/ensembles_multiseed/

---

## SECTION 6 — Case Studies (S53–S71)

### Head Angle (S52–S58)

S52  Head Angle Analysis — title slide (no figure)

S53  Head Angle Is the Top-Attributed Feature Across All Metrics
     Figure:   head_angle_attribution_summary.png
     Script:   eval/gen_head_angle_attribution_summary.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.5" × 4.2"
     Note:     Uniform green (GPV) + orange (IG) bars; FancyBboxPatch around head_angle.

S54  Head Angle MLP Training Evolution — embedded PCA animation frames

S55  Head angle tuning: linear to non-monotonic
     Figure:   head_angle_tuning_variety.png
     Script:   eval/gen_head_angle_tuning_variety.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Figsize:  9.5" × 4.2"
     Note:     Row 1 (blue): highest |ρ| pairs (~0.35–0.39) — relatively linear.
               Row 2 (orange): highest IG pairs (|ρ|~0.03–0.10) — non-monotonic.

S56  Top Pairs: Head Angle × Head Angular Velocity (Colored by z-Scored Activity)
     Figure:   head_angle_scatter_2x2.png
     Script:   eval/eval_head_angle_scatter.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S57  E18 Tuning Curve Stability: Preferred Angle Consistent Across Sessions
     Figure:   head_angle_stability.png
     Script:   eval/eval_head_angle_stability.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S58  E18 Tuning Shape Variety Across Six Representative Sessions
     Figure:   head_angle_tuning_6panel.png
     Script:   eval/eval_head_angle_stability.py
     Outdir:   outputs/mlps/ensembles_multiseed/

### Position (S59–S63)

S59  Case Study: Position — section header

S60  Position Tuning Curves for Top 4 Pairs
     Figure:   position_tuning.png
     Script:   eval/eval_position_tuning.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S61  Position Is Captured via Co-varying Task Events
     Figure:   position_collinearity.png
     Script:   eval/eval_position_proxy.py
     Outdir:   outputs/mlps/ensembles_multiseed/

S62  Position Appears Primary by Correlation — Head Angle Outcompetes It in GPV
     Figure:   position_ablation.png  [row 1 only, honest attribution panels]
     Script:   eval/eval_position_ablation.py
     Outdir:   outputs/mlps/ensembles_multiseed/
     Note:     Full figure also includes ablation R² bars (row 2).

S63  Position-Only MLP Recovers Tuning Curve Shape: Position IS Predictive
     Figure:   position_ablation.png  [same figure, different framing in title]
     Script:   eval/eval_position_ablation.py
     Note:     Same PNG as S62; the slide is split in the build, each half
               corresponding to a row of the figure.

### Validation (S64–S67)

S64  Case Study: Validation — section header

S65  Neural signal: E07 × Cue Visible, E23 × Upcoming Choice
     Figure:   e07_e23_bucket_signal.png
     Script:   eval/e07_e23_bucket_analysis.py
     Outdir:   outputs/ablation_vs_attribution/
     Figsize:  9.5" × 4.2"

S66  Bucket 1: model attributes to their variable (GPV/R² ≥ 10%)
     Figure:   e07_e23_bucket_gpv.png
     Script:   eval/e07_e23_bucket_analysis.py
     Outdir:   outputs/ablation_vs_attribution/

S67  Bucket 2: a co-varying feature captures the signal
     Figure:   e07_e23_bucket_covariation.png
     Script:   eval/e07_e23_bucket_analysis.py
     Outdir:   outputs/ablation_vs_attribution/

### E07 Cue × Position Case Study (S68–S71)

S68  Case Study: Cue × Position — section header (no figure)

S69  E07 in the cue zone: Cue 1 vs Cue 2 position tuning
     Figure:   e07_cue_zone_tuning.png
     Script:   eval/eval_e07_cue_zone_tuning.py
     Outdir:   outputs/ablation_vs_attribution/
     Figsize:  9.5" × 4.2"
     Note:     Top 4 sessions by Cohen's d (Cue 1 vs Cue 2, cue zone only).
               cue_visible ∈ {1,2} timepoints; position bins within cue zone.

S70  Joint GPV in the cue zone: cue+position captures more than marginals
     Figure:   e07_cue_zone_joint_gpv.png
     Script:   eval/eval_e07_cue_zone_joint_gpv.py
     Outdir:   outputs/ablation_vs_attribution/
     Figsize:  9.5" × 4.2"
     Note:     GPV(cue)=0.002, GPV(pos)=0.001, GPV(joint)=0.004, interaction≈+0.001.
               Joint permutation (cue_visible + frame_position same row index) captures
               more than either marginal alone, but the total is still small.

S71  Joint GPV still fails: speed and head angle outcompete cue+position
     Figure:   e07_cue_zone_competition.png
     Script:   eval/eval_e07_cue_zone_competition.py
     Outdir:   outputs/ablation_vs_attribution/
     Figsize:  9.5" × 4.2"
     Note:     GPV in cue zone: Cue=0.002, Pos=0.001, Cue+Pos(joint)=0.003,
               Fwd Speed=0.011, Head Angle=0.011, Rot.Vel.=0.008.
               Purple dashed reference at GPV(joint); speed and head angle are 3× higher.

---

## SECTION 7 — Summary & References (S72–S73)

S72  Encoding Models / Summary — text bullets + Open Questions
S73  References — 7-entry bibliography

---

## HIDDEN / APPENDIX

S74  [Hidden] Methods Reference: Attribution Metrics
S75  [Hidden] Methods Reference: Effect Size Metrics
S76  [Hidden] Methods Reference: Models
S77–S94  [APPENDIX — HIDDEN] A01–A18 definition cards

---

## FIGURE → SCRIPT COMPLETE MAPPING

| PNG | Script | Output dir | Slide |
|---|---|---|---|
| r2_bar_linear.png | eval/generate_r2_figures.py | outputs/mlps/ensembles_multiseed/ | S14 |
| nonlinearity_advantage.png | eval/generate_ml_vs_naive_figures.py | outputs/mlps/ml_vs_naive/ | S15 |
| joint_effect_scatter.png | eval/gen_new_analysis_figures.py | outputs/mlps/ensembles_multiseed/ | S16 |
| mlp_vs_linear_r2.png | eval/generate_ml_vs_naive_figures.py | outputs/mlps/ml_vs_naive/ | S18 |
| r2_bar_mlp.png | eval/generate_r2_figures.py | outputs/mlps/ensembles_multiseed/ | S21 |
| ablation_proof.png | eval/generate_ml_vs_naive_figures.py | outputs/mlps/ml_vs_naive/ | S22 |
| r2_bar_tempconv_comparison.png | eval/generate_r2_figures.py | outputs/cebra_comparison/ | S25 |
| r2_comparison_grand_mean.png | eval/eval_cebra_compare.py | outputs/cebra_comparison/ | S26 |
| r2_distribution_cdf.png | eval/gen_new_analysis_figures.py | outputs/mlps/ensembles_multiseed/ | S27 |
| two_thresholds_scatter.png | eval/eval_cebra_compare.py | outputs/cebra_comparison/ | S28 |
| tempconv_delta_r2.png | eval/gen_new_analysis_figures.py | outputs/mlps/ensembles_multiseed/ | S29 |
| trend_noise_comparison.png | eval/eval_trend_noise.py | outputs/mlps/ensembles_multiseed/ | S35 |
| freq_trace_example.png | eval/eval_freq_trace_example.py | outputs/mlps/ensembles_multiseed/ | S36 |
| gpv_task_ensembles.png | eval/generate_gpv_task_ensembles.py | outputs/mlps/ensembles_multiseed/ | S39 |
| global_vs_cond_pv_scatter.png | eval/generate_attribution_figures.py | outputs/mlps/ensembles_multiseed/ | S41 |
| ig_per_ensemble_heatmap.png | eval/generate_attribution_figures.py | outputs/mlps/ensembles_multiseed/ | S43 |
| group_attribution_comparison.png | eval/generate_group_attribution_comparison.py | outputs/mlps/ensembles_multiseed/ | S44 |
| attribution_agreement_scatter.png | eval/generate_attribution_agreement_figure.py | outputs/mlps/ensembles_multiseed/ | S45 |
| evolution_lineplot_ig.png | eval/eval_evolution_lineplots.py | outputs/mlps/ensembles_multiseed/ | S47 |
| ml_vs_naive_scatter.png | eval/generate_ml_vs_naive_figures.py | outputs/mlps/ml_vs_naive/ | S46 |
| attribution_consistency_merged.png | eval/gen_attribution_consistency.py | outputs/mlps/ensembles_multiseed/ | S49 |
| embedding_linear_map.png | eval/eval_embedding_linear_map.py | outputs/mlps/ensembles_multiseed/ | S50 |
| cross_ensemble_prediction.png | eval/eval_cross_ensemble_prediction.py | outputs/mlps/ensembles_multiseed/ | S51 |
| cross_ensemble_consistency.png | eval/gen_new_analysis_figures.py | outputs/mlps/ensembles_multiseed/ | S52 |
| head_angle_attribution_summary.png | eval/gen_head_angle_attribution_summary.py | outputs/mlps/ensembles_multiseed/ | S53 |
| head_angle_tuning_variety.png | eval/gen_head_angle_tuning_variety.py | outputs/mlps/ensembles_multiseed/ | S55 |
| head_angle_scatter_2x2.png | eval/eval_head_angle_scatter.py | outputs/mlps/ensembles_multiseed/ | S56 |
| head_angle_stability.png | eval/eval_head_angle_stability.py | outputs/mlps/ensembles_multiseed/ | S57 |
| head_angle_tuning_6panel.png | eval/eval_head_angle_stability.py | outputs/mlps/ensembles_multiseed/ | S58 |
| position_tuning.png | eval/eval_position_tuning.py | outputs/mlps/ensembles_multiseed/ | S60 |
| position_collinearity.png | eval/eval_position_proxy.py | outputs/mlps/ensembles_multiseed/ | S61 |
| position_ablation.png | eval/eval_position_ablation.py | outputs/mlps/ensembles_multiseed/ | S62,S63 |
| e07_e23_bucket_signal.png | eval/e07_e23_bucket_analysis.py | outputs/ablation_vs_attribution/ | S65 |
| e07_e23_bucket_gpv.png | eval/e07_e23_bucket_analysis.py | outputs/ablation_vs_attribution/ | S66 |
| e07_e23_bucket_covariation.png | eval/e07_e23_bucket_analysis.py | outputs/ablation_vs_attribution/ | S67 |
| e07_cue_zone_tuning.png | eval/eval_e07_cue_zone_tuning.py | outputs/ablation_vs_attribution/ | S69 |
| e07_cue_zone_joint_gpv.png | eval/eval_e07_cue_zone_joint_gpv.py | outputs/ablation_vs_attribution/ | S70 |
| e07_cue_zone_competition.png | eval/eval_e07_cue_zone_competition.py | outputs/ablation_vs_attribution/ | S71 |

---

## ORPHANED FIGURES (on disk, not in v45 deck)

| PNG | Script | Reason removed |
|---|---|---|
| honest_attribution_example.png | eval/honest_attribution_example.py | Was S46 in earlier versions; current S46 shows ml_vs_naive_scatter |
| e07_joint_tuning.png | eval/eval_e07_joint_tuning.py | Replaced by e07_cue_zone_tuning.png in v43 |
| e07_joint_gpv.png | eval/eval_e07_joint_gpv.py | All-timepoints version; replaced by cue-zone version |
| e07_joint_captured.png | eval/eval_e07_joint_captured.py | Replaced by cue-zone attribution/competition figures |
| e07_position_cue_tuning.png | eval/eval_e07_position_cue_tuning.py | Was S70 in v41; flat/empty; replaced |
| e07_cue_zone_attribution.png | eval/eval_e07_cue_zone_attribution.py | Replaced by competition figure in v45 |
| e07_cue_zone_features.png | eval/eval_e07_cue_zone_features.py | Replaced by competition figure in v45 |
| e07_cue_zone_separation.png | eval/eval_e07_cue_zone_separation.py | Replaced by joint GPV in v45 |
| e07_assembly_heatmap.png | eval/eval_e07_assembly_heatmap.py | Trial×position heatmap; replaced by cue-zone analysis |
| embedding_consistency_comparison.png | eval/eval_cross_model_consistency.py | Probe-prediction Pearson r; wrong metric; removed in v20 |
| gpv_group_ensemble_heatmap.png | eval/generate_attribution_figures.py | Superseded by gpv_task_ensembles.png |
| embedding_consistency_violin.png | eval/eval_cebra_consistency.py | Wrong consistency metric; replaced by embedding_linear_map |
| attribution_consistency_merged.png (old) | eval/build_v16.py (inline) | Now generated by gen_attribution_consistency.py |

---

## KNOWN ANOMALIES

1. **S31–S33 not regenerable**: trial trace slides copied from earlier PPTX;
   eval/eval_spikes_and_traces.py produces similar traces but at different path/format.

2. **S62/S63 share one PNG**: position_ablation.png has two rows; each slide
   shows one row, with the build script cutting the image at the slide boundary.

3. **Appendix A06 orphaned metric**: describes probe-prediction Pearson r consistency
   (removed from main deck in v20); consider relabelling to match embedding_linear_map.

4. **attribution_consistency_merged anomaly**: was generated inline in build_v16.py;
   now has a dedicated script (gen_attribution_consistency.py) — use that for reruns.
