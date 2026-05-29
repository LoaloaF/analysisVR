# Slide Plan v2
*Based on thesis context: P(x | s, a) characterization as measurement tool for evolving representations*

---

## Narrative spine

The thesis asks how neural representations evolve under learning. To measure evolution,
you need a reliable, scalable way to characterize what is represented at any given
point in time. This project provides that measurement tool and validates it.

One-sentence framing for the talk:
> "We build and validate a scalable method for characterizing the encoding function
> P(ensemble activity | state, actions) — and show what it reveals about the
> structure of behavioral representations in this dataset."

This preempts the "did you discover anything new" objection by being explicit that
the contribution is methodological + characterization, not a novel biological discovery.

---

## Section-by-section plan (48 slides target)

---

### 1. INTRO  (≈10 slides)
**Goal**: situate the project within the thesis, establish the encoding framing,
describe the dataset.

| # | Type | Content | Notes |
|---|------|---------|-------|
| 1 | OLD | Title | |
| 2 | OLD | Example trial (MP4) | Shows what the task looks like concretely |
| 3 | OLD | Paradigm & Setup | VR corridor, cue zones, reward zones |
| 4 | OLD | Data distribution | Sessions, ensembles, trial counts |
| 5 | OLD | P(a), P(s), P(x) framework | Supervisor's framework — sets up encoding view |
| 6 | OLD | Input Data Features: P(s,a) continued | The 11 feature groups |
| 7 | OLD | SIPEC → ensemble pipeline | How ensembles are extracted |
| 8 | OLD | Recover bin order | |
| 9 | OLD | Approach overview | |
| 10 | OLD | Dataset details | 29 sessions, 23 ensembles, trial counts |

*Cut candidate*: slides 7-8 (pipeline detail) may be too deep for a thesis presentation —
consider collapsing into one "Methods overview" slide.

---

### 2. MODELS & RAW R² RESULTS  (≈8 slides)
**Goal**: introduce the three model tiers (Linear → MLP → TempConv) alongside
their R² performance. The models are not competing — they're tools at different
levels of complexity.

**Key framing change**: Report R² as a *distribution* (fraction of pairs at each
threshold), not grand mean. The 5% grand mean is pulled down by hundreds of
non-responsive pairs; the scientifically relevant question is how many pairs
show reliable encoding at R²≥0.05, R²≥0.10, R²≥0.20.

| # | Type | Content | Notes |
|---|------|---------|-------|
| 11 | OLD | Predictive encoding metrics | Defines R², train/test split, 5-seed protocol |
| 12 | OLD | Linear model explanation | Affine baseline |
| 13 | FIG | r2_bar_linear.png | Per-ensemble linear R² |
| 14 | FIG | Part 2 — nonlinearity_advantage.png | **Moved here** — motivates why linear isn't enough before introducing MLP |
| 15 | FIG | Part 3 — mlp_vs_linear_r2.png | MLP beats linear for nonlinear pairs — justifies the upgrade |
| 16 | FIG | Part 4 — ablation_proof.png | Single-feature MLP finds signal GLM misses — ML-exclusive claim |
| 17 | OLD | MLP explanation | |
| 18 | OLD | MLP capability demo (GIFs) | |
| 19 | FIG | r2_bar_mlp.png | Per-ensemble MLP R² |
| 20 | OLD | TempConv explanation | |
| 21 | OLD | CEBRA structure / contrastive learning | |
| 22 | FIG | r2_grand_mean_bars.png | **Reframed**: same representations, different complexity — not a competition |
| 23 | FIG | two_thresholds_scatter.png | Pair counts at R²≥0.01 and R²≥0.05 |

*TODO*: Add R² distribution panel (histogram or CDF of mean R² across all valid
pairs) to slide 22 or 23 to show the "meaningful subset" framing concretely.

---

### 3. EXAMPLE PREDICTIONS  (≈3 slides)
**Goal**: qualitative demonstration that the models are doing something real.
These land better after R² is established.

| # | Type | Content |
|---|------|---------|
| 24 | OLD | Example Plots — Ensemble Prediction 1 |
| 25 | OLD | Example Plots — Ensemble Prediction 2 |
| 26 | OLD | Example Plots — Ensemble Prediction 3 |

---

### 4. FEATURE ATTRIBUTION  (≈10 slides)
**Goal**: characterize the structure of P(x | s, a) — which behavioral dimensions
drive which ensembles. This is the core scientific contribution.

**Framing**: attribution is not a discovery claim — it's a characterization of
the encoding function. The validation chain (consistency, agreement with naive
methods) shows the characterization is reliable.

| # | Type | Content | Notes |
|---|------|---------|-------|
| 27 | OLD | Permutation variance explanation | |
| 28 | FIG | gpv_group_ensemble_heatmap.png | Which features drive which ensembles |
| 29 | OLD | Conditional PV explanation | |
| 30 | FIG | global_vs_cond_pv_scatter.png | **Reframed**: evidence collinearity exists — NOT a causal identification tool. Be honest: CPV identifies that some attribution is shared, not which feature is causal. |
| 31 | OLD | Integrated Gradients explanation | |
| 32 | FIG | ig_per_ensemble_heatmap.png | IG attribution — convergent evidence |
| 33 | FIG | group_attribution_comparison.png | MLP vs TempConv-Cont attribution profile |
| 34 | FIG | attribution_agreement_scatter.png | ρ≈0.95 across architectures |
| 35 | FIG | Part 1 — ml_vs_naive_scatter.png | Attribution agrees with naive effect sizes — validates the tool |

*TODO (optional)*: Add one slide showing interaction-effect pairs — ensembles where
MLP R² is high but ALL individual univariate correlations are low. This is the
strictly ML-exclusive claim. Need to query the data first.

---

### 5. CONSISTENCY  (≈2 slides)
**Goal**: this is the anchor for the thesis. If representations can be reliably
characterized across seeds and architectures, they can be tracked over learning.

**Reframed as**: "The measurement tool is stable — a prerequisite for using it
to track representation evolution."

| # | Type | Content | Notes |
|---|------|---------|-------|
| 36 | FIG | r2_consistency_bar.png | Cross-seed R² consistency |
| 37 | FIG | embedding_consistency_comparison.png | Cross-model consistency (MLP, TempConv-Cont, TempConv-Pred) |

---

### 6. FREQUENCY / TEMPORAL STRUCTURE  (≈2 slides)
**Goal**: TempConv story. Reframe — TempConv is NOT "better than MLP in general."
It converges on the same representations (attribution ρ≈0.95) but additionally
captures a subset of ensembles with fast temporal dynamics that point-in-time
models miss.

**Reframed claim**: "Behavioral representations are architecture-independent
(MLP ≈ TempConv attribution), but TempConv-Pred's temporal context reveals
a minority of ensembles encoding sub-second dynamics beyond instantaneous state."

*TODO*: Quantify — how many pairs does TempConv-Pred beat MLP by ΔR²>0.05?
Are those pairs identifiable by their neural signal spectral content?
This determines how strongly to sell the frequency story.

| # | Type | Content |
|---|------|---------|
| 38 | FIG | trend_noise_comparison.png |

*Consider*: one additional slide showing ΔR² distribution (TempConv-Pred minus MLP)
and the fraction of pairs where temporal context matters, if the numbers support it.

---

### 7. CASE STUDIES  (≈9 slides)
**Goal**: concrete examples of what the encoding characterization reveals.
Head angle is the strongest story — the full chain from attribution → tuning
curves → ablation → non-monotonic detection is defensible step-by-step.
Position is the second story — shows the method handles collinear features
honestly.

| # | Type | Content | Notes |
|---|------|---------|-------|
| 39 | OLD | Head Angle Analysis | |
| 40 | OLD | Head Angle PCA animation (GIFs) | |
| 41 | FIG | head_angle_scatter_2x2.png | Top pairs colored by activation |
| 42 | FIG | head_angle_stability.png | E18 preferred angle consistent across sessions |
| 43 | FIG | head_angle_tuning_6panel.png | Non-monotonic tuning shapes |
| 44 | FIG | position_tuning.png | Tuning curves exist, but attribution is low |
| 45 | FIG | position_collinearity.png | Task events are position-triggered → proxy features capture it |
| 46 | FIG | position_ablation.png | Position-only MLP recovers curve shape → position IS predictive, just collinear |
| 47 | OLD | Case Study E07/E23 intro | |
| 48 | FIG | case_studies_e07_e23.png | |

---

## Slides cut from v6

- **Behavioral trace** (feature trace figure): cut by user in v3 — not needed
- **Recover bin order**: may cut in future — pipeline detail
- **SIPEC slide**: may cut — pipeline detail

---

## Experimental results (resolved)

### 1. Interaction-effect pairs ✓
43% of R²≥0.05 pairs (87/202) have max univariate |ρ|<0.15 across all features.
28% of R²≥0.10 pairs (21/74) also fall in this zone.
→ ML-exclusive claim is quantitatively substantial, not anecdotal.
→ ADD: one scatter figure (max |ρ| vs MLP R², highlight ML-exclusive quadrant).
   Filename: `joint_effect_scatter.png`
   Placement: after attribution heatmaps, before Part 1 validation.

### 2. TempConv ≈ MLP ✓
TempConv-Pred beats MLP by >0.05 ΔR² in only 3.4% of pairs (15/442).
Mean ΔR² = −0.012 (MLP slightly better overall). Median ΔR² ≈ 0.
→ REFRAME: architecture-independent representations, not "TempConv is better."
→ ADD: ΔR² distribution histogram showing concentration near zero.
   Filename: `tempconv_delta_r2.png`
   Placement: replaces or supplements the grand mean bars slide.

### 3. R² distribution ✓
45.7% of valid pairs ≥ R²=0.05 (202/442). Only 16.7% ≥ R²=0.10 (74/442).
→ REPLACE grand mean framing with distribution CDF.
   Filename: `r2_distribution_cdf.png`
   Placement: alongside or replacing r2_grand_mean_bars.

### 4. Cross-ensemble attribution consistency ✓
Within-session (different ensembles, same session): median ρ = 0.81, 90% > 0.5.
Across-session (same ensemble, different sessions): median ρ = 0.73, 81% > 0.5.
→ Behavioural representation is session-level, not ensemble-specific.
→ Note: not surprising given LSTM cross-ensemble generalisation result
  (section cancelled), but new to this audience. Include as Option A.
→ ADD: violin plot comparing within- vs across-session distributions.
   Filename: `cross_ensemble_consistency.png`
   Placement: consistency section, after embedding_consistency_comparison.

## Remaining TODOs

1. **CPV slide reframing**: Change narrative text from "collinearity filtering"
   to "evidence that attribution is shared — causal identification requires
   intervention."

2. **TempConv slide reframing**: Change title/narrative to emphasise
   architecture-independence. Drop claim that TempConv is generally better.

---

## Slides that need narrative text edits (no new figures)

- Slide 5 (P(a),P(s),P(x) framework): add one sentence — "This project
  characterizes the encoding direction P(x|s,a) at a point in time."
- Slide 22 (grand mean R²): retitle from "comparison" framing to
  "same representations at different complexity levels"
- Slide 30 (CPV scatter): add honest caveat about model choice vs causality
- Slide 33 (group attribution comparison): emphasize convergence,
  not difference
- Slide 38 (frequency): explicitly state "architecture-independent representations,
  with TempConv additionally sensitive to temporal dynamics"
