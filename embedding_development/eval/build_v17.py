#!/usr/bin/env python3
"""
build_v17.py  —  Add hidden appendix to v16 (67 slides).

Appends 17 text-only hidden reference slides covering:
  Metrics    : R², GPV, CPV, IG, Spearman ρ, Pearson r, Trend/Noise R², η²/Cohen's d
  Models     : Summary table, Linear, MLP, TempConv-Cont, TempConv-Pred
  Experiments: Evaluation setup, input features, single-feature ablation,
               embedding geometry, attribution consistency, frequency decomposition,
               case studies

Net: 67 + 17 = 84 slides
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')

SRC  = os.path.join(root, 'outputs', 'ultimate_presentation_v16.pptx')
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v17.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v17.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)

C_BG    = RGBColor(0xFF, 0xFF, 0xFF)   # white background
C_TITLE = RGBColor(0x1A, 0x23, 0x3A)   # dark navy
C_HEAD  = RGBColor(0x2C, 0x3E, 0x6B)   # section header
C_BODY  = RGBColor(0x22, 0x22, 0x22)   # body text
C_TAG   = RGBColor(0x88, 0x88, 0x88)   # grey tag label
C_RULE  = RGBColor(0xCC, 0xCC, 0xCC)   # horizontal rule colour


def add_appendix_slide(prs, title, sections):
    """
    Add a text-only hidden slide.

    sections: list of (header, lines) tuples.
      header : str  — e.g. "Definition", "Computation", "Appears on"
      lines  : list[str]  — bullet points or paragraphs
    """
    layout = prs.slide_layouts[6]      # blank
    slide  = prs.slides.add_slide(layout)

    # Background
    fill = slide.background.fill
    fill.solid(); fill.fore_color.rgb = C_BG

    # "[APPENDIX]" tag
    tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.05), Inches(2.5), Inches(0.28))
    p  = tb.text_frame.paragraphs[0]
    r  = p.add_run(); r.text = '[APPENDIX — HIDDEN]'
    r.font.size = Pt(7); r.font.color.rgb = C_TAG; r.font.bold = False

    # Title
    tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.28), Inches(9.4), Inches(0.52))
    tf = tb.text_frame; tf.word_wrap = True
    p  = tf.paragraphs[0]
    r  = p.add_run(); r.text = title
    r.font.size = Pt(18); r.font.bold = True; r.font.color.rgb = C_TITLE

    # Horizontal rule (thin rectangle)
    slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        int(Inches(0.3)), int(Inches(0.83)), int(Inches(9.4)), int(Inches(0.012))
    ).fill.solid(); slide.shapes[-1].fill.fore_color.rgb = C_RULE
    slide.shapes[-1].line.fill.background()

    # Body — iterate sections
    n_sections = len(sections)
    col_w      = Inches(9.4) / max(n_sections, 1)
    top        = Inches(0.92)
    avail_h    = Inches(4.55)

    for col_idx, (header, lines) in enumerate(sections):
        x = Inches(0.3) + int(col_idx * col_w)
        w = int(col_w - Inches(0.15))

        # Section header
        tb = slide.shapes.add_textbox(x, int(top), w, int(Inches(0.28)))
        p  = tb.text_frame.paragraphs[0]
        r  = p.add_run(); r.text = header.upper()
        r.font.size = Pt(8); r.font.bold = True; r.font.color.rgb = C_HEAD

        # Section body
        body_top = int(top + Inches(0.30))
        tb2 = slide.shapes.add_textbox(x, body_top, w, int(avail_h - Inches(0.30)))
        tf2 = tb2.text_frame; tf2.word_wrap = True

        first = True
        for line in lines:
            if first:
                p2 = tf2.paragraphs[0]; first = False
            else:
                p2 = tf2.add_paragraph()
            p2.space_before = Pt(2)
            if line.startswith('•'):
                p2.level = 1
                r2 = p2.add_run(); r2.text = line[1:].strip()
                r2.font.size = Pt(9); r2.font.color.rgb = C_BODY
            elif line.startswith('  –'):
                p2.level = 2
                r2 = p2.add_run(); r2.text = line[3:].strip()
                r2.font.size = Pt(8.5); r2.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            elif line == '':
                r2 = p2.add_run(); r2.text = ''
                r2.font.size = Pt(4)
            else:
                r2 = p2.add_run(); r2.text = line
                r2.font.size = Pt(9.5); r2.font.bold = True; r2.font.color.rgb = C_BODY

    return len(prs.slides) - 1


# ══════════════════════════════════════════════════════════════════════════════
# Appendix slide definitions
# ══════════════════════════════════════════════════════════════════════════════

APPENDIX_SLIDES = [

    # ── METRICS ──────────────────────────────────────────────────────────────

    ("A01 · Metric: R² (Coefficient of Determination)", [
        ("Definition", [
            "Fraction of variance in neural activity explained by the model's predictions.",
            "• R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²",
            "• R² = 1 → perfect prediction",
            "• R² = 0 → predicts mean only",
            "• R² < 0 → worse than constant mean",
        ]),
        ("Computation", [
            "Per (session, ensemble) pair:",
            "• Load test-set behavioral features X",
            "• Run forward pass → ŷ",
            "• sklearn r2_score(y_test, ŷ)",
            "• Averaged across 5 seeds for figures",
            "• Threshold R²≥0.1 used for most analyses",
        ]),
        ("Appears on", [
            "S14 Linear baseline",
            "S21 MLP baseline",
            "S25–26 Cross-model summary",
            "S27 CDF threshold plot",
            "S35 Frequency decomp.",
            "S47 Consistency heatmap",
            "S49 Embedding geometry",
        ]),
    ]),

    ("A02 · Metric: GPV — Global Permutation Variance", [
        ("Definition", [
            "Drop in R² when a semantic feature group is globally permuted (shuffled across all samples).",
            "• GPV(g) = R²_base − R²_permuted(g)",
            "• Large GPV → feature group g matters",
            "• Zero GPV → group redundant with others",
            "• Can be negative (permuting one correlated feature can slightly improve R²)",
        ]),
        ("Computation", [
            "For each of 11 semantic groups g:",
            "• Permute all columns in g with same row index",
            "• Recompute R² on permuted data",
            "• Repeat 5 times; take median",
            "• Averaged across 5 random seeds",
            "Groups: 7 continuous singletons + 4 categorical (one-hot jointly permuted)",
        ]),
        ("Appears on", [
            "S38 GPV heatmap (11×23)",
            "S40 GPV vs CPV scatter",
            "S43 Cross-model profile",
            "S44 Attribution agreement",
            "S48 Consistency merged",
            "S63 E07 / E23 case study",
        ]),
    ]),

    ("A03 · Metric: CPV — Conditional Permutation Variance", [
        ("Definition", [
            "GPV conditioned on other features via k-NN swapping, reducing collinearity inflation.",
            "• CPV(g) = R²_base − R²_knn_swap(g)",
            "• If GPV >> CPV: feature group g is correlated with other predictors",
            "• If GPV ≈ CPV: feature group g has unique predictive power",
        ]),
        ("Computation", [
            "For each semantic group g:",
            "• Build k-NN index on all features EXCEPT g (k=10)",
            "• For each sample, swap its g-values with a random k-NN neighbor's g-values",
            "• Recompute R²; repeat 5 times; median",
            "• Points below diagonal in GPV-vs-CPV scatter = collinearity inflated",
        ]),
        ("Appears on", [
            "S40 Global vs Conditional PV scatter",
            "S66 Methods Reference",
        ]),
    ]),

    ("A04 · Metric: IG — Integrated Gradients", [
        ("Definition", [
            "Gradient-based attribution: integral of gradients along straight-line path from baseline (0) to input.",
            "• IG_i(x) = (x_i − 0) × ∫₀¹ ∂F(α·x)/∂x_i dα",
            "• Satisfies completeness: Σ IG_i = F(x) − F(0)",
            "• |IG|: importance magnitude",
            "• Signed IG: direction of effect per categorical level",
        ]),
        ("Computation", [
            "50 interpolation steps α ∈ [0, 1]:",
            "• X_interp = α × X_test (baseline = 0)",
            "• Accumulate gradients ∂F/∂x at each α",
            "• IG = (X − 0) × mean(grads)",
            "• Summed to semantic groups",
            "Per-level: masked to trials where level is active (≥20 trials required)",
        ]),
        ("Appears on", [
            "S42 IG heatmap (11×23)",
            "S43 Cross-model profile",
            "S44 Attribution agreement ρ",
            "S45 IG vs η² scatter",
            "S63 E07 / E23 case study",
        ]),
    ]),

    ("A05 · Metric: Spearman ρ of Attribution Profiles", [
        ("Definition", [
            "Rank correlation between two models' (or ensembles') GPV or IG profiles over 11 semantic groups.",
            "• ρ = 1 → identical feature importance ranking",
            "• ρ = 0 → no agreement",
            "• ρ < 0 → disagreement (unlikely)",
            "Computed per (session, ensemble) pair where both models have R²≥0.1",
        ]),
        ("Computation", [
            "For each valid (s, e) pair:",
            "• Extract 11-element GPV vector from model A and model B",
            "• Compute scipy.stats.spearmanr(gpv_A, gpv_B)",
            "• Aggregate across all valid pairs → violin plot",
            "Three pair types: MLP×TC-Cont, MLP×TC-Pred, TC-Cont×TC-Pred",
            "Also computed: cross-ensemble (same model, different e_i vs e_j within session)",
        ]),
        ("Appears on", [
            "S44 Attribution agreement scatter",
            "S48 Merged consistency violin",
        ]),
    ]),

    ("A06 · Metric: Pearson r (Cross-Seed Prediction Consistency)", [
        ("Definition", [
            "Mean pairwise Pearson correlation of test-set predictions across random seeds.",
            "• r = corr(ŷ_seed_i, ŷ_seed_j) on overlapping test trials",
            "• High r → model converges to same function regardless of random initialisation",
            "• Averaged over C(5,2)=10 seed pairs per (session, ensemble)",
        ]),
        ("Computation", [
            "For each (session, ensemble, model_type):",
            "• Find trials in both seeds' test sets (intersection)",
            "• Compute predictions with each seed's model on shared trials",
            "• Pairwise Pearson r across 10 seed pairs",
            "• Mean r stored per (session, ensemble)",
            "Displayed as heatmap: sessions × ensembles",
        ]),
        ("Appears on", [
            "S47 Embedding consistency heatmap",
        ]),
    ]),

    ("A07 · Metric: Linear Map R² (Embedding Geometry)", [
        ("Definition", [
            "Mean R² of ridge regression predicting embedding B from embedding A.",
            "• Linear map R² = mean_d R²(H_A → H_B[:,d])  over 64 dimensions",
            "• High R² → embeddings are linearly related (same geometric structure)",
            "• Low R² → embeddings span different subspaces",
            "Distinct from Pearson r: tests GEOMETRIC EQUIVALENCE not just correlation",
        ]),
        ("Computation", [
            "For each embedding pair (H_A, H_B) with shape (T, 64):",
            "• For each of 64 output dims d:",
            "  – 5-fold CV Ridge(α=1.0) regressing H_A → H_B[:,d]",
            "  – R² clipped to [0,1], mean across folds",
            "• Mean R² across 64 dims",
            "Four variation axes: cross-seed, cross-session,",
            "cross-ensemble, cross-architecture",
        ]),
        ("Appears on", [
            "S49 Embedding geometric consistency violin",
        ]),
    ]),

    ("A08 · Metric: Trend / Noise R² (Frequency Decomposition)", [
        ("Definition", [
            "Decompose neural activity into slow (trend) and fast (noise) components; measure R² on each.",
            "• Slow (trend): 500ms moving average of y_true and y_pred",
            "• Fast (noise): residual after removing trend",
            "• R²_trend: how well model captures slow dynamics",
            "• R²_noise: how well model captures fast oscillations",
        ]),
        ("Computation", [
            "window = 12 bins @ 40ms/bin = 480ms:",
            "• y_slow = uniform_filter1d(y_true, size=12)",
            "• y_fast = y_true − y_slow",
            "• ŷ_slow = uniform_filter1d(ŷ, size=12)",
            "• ŷ_fast = ŷ − ŷ_slow",
            "• R²_trend = r2_score(y_slow, ŷ_slow)",
            "• R²_noise = r2_score(y_fast, ŷ_fast)",
        ]),
        ("Appears on", [
            "S35 Slow/fast bar chart",
            "MLP, TC-Cont, TC-Pred compared",
            "Top-5 (session, ensemble) pairs by MLP R²",
        ]),
    ]),

    ("A09 · Metric: η² and Cohen's d (Univariate Effect Sizes)", [
        ("η² — Eta-Squared (Categorical)", [
            "Fraction of neural variance explained by a SINGLE feature:",
            "• η² = SS_between / SS_total",
            "• Computed per feature, per (session, ensemble)",
            "• Measures univariate explanatory power (no joint effects)",
            "• Used as naive baseline vs GPV and IG",
            "  – Low η² but high GPV → feature is jointly predictive",
        ]),
        ("Cohen's d (Continuous)", [
            "Standardised mean difference between two groups:",
            "• d = (μ₁ − μ₂) / σ_pooled",
            "• |d| > 0.2: small, > 0.5: medium, > 0.8: large",
            "• Used to compare ensemble activity between stimulus conditions",
            "  – e.g., cue_visible=0 vs cue_visible=1",
            "• Appears in case study slides (E07 / E23 validation)",
        ]),
        ("Appears on", [
            "S16 Joint effects scatter (η²)",
            "S22 Single-feature ablation (η²)",
            "S45 IG vs η² scatter",
            "S66 Methods Reference",
            "Case study slides S62–S63",
        ]),
    ]),

    # ── MODELS ───────────────────────────────────────────────────────────────

    ("A10 · Models: Summary Table", [
        ("Architecture", [
            "Linear   Input(17) → Ridge(α=1) → 1",
            "MLP      Input(17) → FC(64)+ReLU → FC(64)+ReLU → 1",
            "TC-Cont  Input(17, ±5 frames) → TempConv → Embed(64)",
            "TC-Pred  Input(17, ±5 frames) → TempConv → Embed(64)",
        ]),
        ("Training", [
            "Loss  | Temporal | Seeds | Typical R²",
            "MSE L2-reg  | No  | 5 | 0.03",
            "MSE         | No  | 5 | 0.08",
            "InfoNCE     | Yes | 5 | 0.10",
            "Forward MSE | Yes | 5 | 0.09",
            "",
            "All models: per-session, per-ensemble, 40ms bins",
            "29 sessions × 23 ensembles × 5 seeds = 3,335 models per type",
        ]),
        ("Embedding", [
            "Linear:  no embedding (direct regression)",
            "MLP:     64-dim hidden layer (last before output)",
            "TC-Cont: 64-dim CEBRA embedding (L2-normalised)",
            "TC-Pred: 64-dim CEBRA embedding (not normalised)",
            "",
            "All embeddings: same dimensionality for fair comparison",
            "Cross-architecture consistency tested in S49",
        ]),
    ]),

    ("A11 · Model: Linear Baseline (Ridge Regression)", [
        ("Architecture", [
            "Input (17 features) → Ridge(α=1.0) → scalar output",
            "",
            "Input features (z-scored within session):",
            "• 7 continuous: speed, accel., rot.vel., rot.accel., head angular vel., head angle, position",
            "• 10 one-hot: cue(1), choice(-1,0,1), reward(1), lick(1)",
            "",
            "No nonlinearity. No temporal context. No hidden representation.",
        ]),
        ("Training", [
            "Per session × ensemble × seed:",
            "• sklearn Ridge(alpha=1.0)",
            "• 80/20 train/test split (stratified by seed)",
            "• No early stopping, no dropout",
            "• Baseline for nonlinearity tests (S15, S22)",
        ]),
        ("Interpretation", [
            "Captures linear additive effects only.",
            "Low R²≈0.03 reflects that neural tuning is:",
            "• Nonlinear (threshold, saturation)",
            "• Joint multi-feature (interaction effects)",
            "• Temporally structured (history-dependent)",
            "",
            "Used as lower bound in all comparisons.",
        ]),
    ]),

    ("A12 · Model: MLP (Multilayer Perceptron)", [
        ("Architecture", [
            "Input(17) → Linear(17→64) + ReLU",
            "          → Linear(64→64) + ReLU",
            "          → Linear(64→1)",
            "",
            "Embedding = last hidden layer output (64-dim)",
            "No dropout. No weight decay. No batch norm.",
            "Activation: ReLU throughout.",
        ]),
        ("Training", [
            "Loss:      MSE (Mean Squared Error)",
            "Optimiser: Adam (lr=1e-3)",
            "Epochs:    100",
            "Batch:     full session (no mini-batching)",
            "Seeds:     [42, 43, 44, 45, 46]",
            "Per model: ~1 session × 1 ensemble × 1 seed",
        ]),
        ("Properties", [
            "• Captures nonlinear, instantaneous tuning",
            "• No temporal context (single 40ms bin)",
            "• Fast to train (~seconds per model)",
            "• Attribution-friendly (GPV, CPV, IG all supported)",
            "• Embedding highly reproducible across seeds (Pearson r>0.9)",
            "• 3,335 total trained models",
        ]),
    ]),

    ("A13 · Model: TempConv-Cont and TempConv-Pred (CEBRA)", [
        ("Architecture", [
            "TempConv encoder via CEBRA framework:",
            "• Input window: ±5 frames @ 40ms = 440ms context",
            "• Conv1d(k=2, GELU) → 3×SkipBlock(Conv1d k=3,GELU,residual) → Conv1d(k=3)",
            "• Output: 64-dim embedding",
            "• TC-Cont: L2-normalised (unit sphere); TC-Pred: unnormalised",
        ]),
        ("Loss Functions", [
            "TC-Cont (Contrastive):",
            "• InfoNCE loss: pulls behaviourally similar timepoints together",
            "• Positive pairs: same session timepoints that are temporally close",
            "• k=20 negatives per positive, batch=128, lr=3e-4, 20 epochs",
            "",
            "TC-Pred (Predictive):",
            "• MSE loss: predicts future behavioral state",
            "• No explicit contrastive objective",
            "• Same conv architecture, same hyperparameters",
        ]),
        ("Properties vs MLP", [
            "• Temporal context (440ms window) enables fast dynamics capture (S35)",
            "• TC-Cont embedding rotational freedom → lower cross-seed linear map R²",
            "• TC-Pred embedding matches MLP geometry (cross-arch R²≈0.68)",
            "• Both temporal models outperform MLP on fast/noise component (R²_noise)",
            "• Models trained per-session, per-ensemble: same granularity as MLP",
        ]),
    ]),

    # ── EXPERIMENTS ──────────────────────────────────────────────────────────

    ("A14 · Experiment: Evaluation Setup", [
        ("Dataset", [
            "29 sessions (1 rat, 1 recording site)",
            "• Sessions 0–6: uncued (rat stops at any reward zone)",
            "• Sessions 7–28: cued (rat must stop at cued zone)",
            "23 ensembles per session (global ICA on 77 consistent neurons)",
            "40ms bins; all sessions share same 17 behavioral features",
            "Train/test: 80/20 random split, stratified by seed",
        ]),
        ("Thresholds & Hyperparameters", [
            "R² inclusion thresholds:",
            "• R²≥0.01: broad validity (CDF plots)",
            "• R²≥0.05: medium validity (counts)",
            "• R²≥0.10: strict validity (attribution, embedding)",
            "",
            "Other parameters:",
            "• Seeds: [42, 43, 44, 45, 46]",
            "• Attribution permutation repeats: 5",
            "• IG interpolation steps: 50",
            "• k-NN neighbors (CPV): k=10",
            "• Ridge α (all probes): 1.0",
        ]),
        ("Valid Pair Counts", [
            "At R²≥0.10 threshold:",
            "• Linear: ~50 valid (session, ensemble) pairs",
            "• MLP:    ~250 valid pairs",
            "• TC-Cont: ~300 valid pairs",
            "• TC-Pred: ~320 valid pairs",
            "",
            "29 sessions × 23 ensembles = 667 possible pairs",
            "~35–45% pass R²≥0.10 for nonlinear models",
        ]),
    ]),

    ("A15 · Experiment: Input Features and Semantic Groups", [
        ("17 Input Features (z-scored)", [
            "7 continuous features (single columns):",
            "• Forward Speed (cm/s)          [fwd_speed]",
            "• Forward Acceleration (cm/s²)  [fwd_accel]",
            "• Rotational Velocity (°/s)      [rot_vel]",
            "• Rotational Acceleration (°/s²) [rot_accel]",
            "• Head Angular Velocity (°/s)    [head_ang_vel]",
            "• Head Angle (°)                 [head_angle]",
            "• Track Position (cm)            [position]",
            "10 one-hot columns (4 categorical variables):",
            "• Cue Visible {0,1}",
            "• Upcoming Choice {−1, 0, 1}",
            "• Reward Window {0, 1}",
            "• Lick Detected {0, 1}",
        ]),
        ("11 Semantic Groups (GPV/CPV/IG)", [
            "Each continuous feature → 1 group (singleton)",
            "Each categorical variable → 1 group (all one-hot cols jointly permuted)",
            "",
            "Groups 1–7: Fwd Speed, Fwd Accel, Rot Vel, Rot Accel,",
            "            Head Ang Vel, Head Angle, Position",
            "Groups 8–11: Cue Visible, Upcoming Choice, Reward Win, Lick",
            "",
            "Grouping rationale: permuting all levels of a categorical together",
            "preserves marginal distributions within groups",
        ]),
        ("Session Structure", [
            "Sessions 0–6 (UNCUED):",
            "• Rat stops at any reward zone",
            "• No spatial cue presented",
            "• Less structured behaviour",
            "",
            "Sessions 7–28 (CUED):",
            "• Rat must stop at the cued reward zone",
            "• Spatial cue appears mid-trial",
            "• More structured behavioural traces",
        ]),
    ]),

    ("A16 · Experiment: Single-Feature Ablation & Nonlinearity", [
        ("Single-Feature Ablation (S22)", [
            "Tests whether one feature alone is sufficient for MLP to predict activity.",
            "",
            "Method:",
            "• Train MLP(1→64→64→1) on each feature separately",
            "• Compare single-feature MLP R² vs full MLP R² vs GLM R²",
            "",
            "Finding: Single-feature MLP can predict where GLM sees R²≈0.",
            "Interpretation: MLP captures threshold/nonlinear tuning that",
            "linear regression cannot.",
        ]),
        ("Nonlinearity Proof (S15, S16)", [
            "Two complementary tests:",
            "",
            "1. Spearman ρ vs MLP R²:",
            "• Many pairs: high MLP R² but low Pearson r (feature-activity)",
            "• Nonlinear tuning exists beyond what linear correlation captures",
            "",
            "2. η² << 5% for all single features (S16):",
            "• No single feature explains >5% variance",
            "• But MLP (multi-feature, nonlinear) achieves R²>0.1 on same pairs",
            "• Joint multi-feature nonlinear interaction is necessary",
        ]),
        ("Position Ablation (S60)", [
            "Position-only MLP trained on track position alone:",
            "• MLP(1→64→64→1) with only frame_position as input",
            "• Recovers tuning curve SHAPE (R²>0.2 for top pairs)",
            "• Demonstrates position IS predictive despite GPV showing task events dominate",
            "",
            "Interpretation: position and task events are collinear (position-triggered events).",
            "GPV shows task events > position because they carry unique information beyond",
            "position alone.",
        ]),
    ]),

    ("A17 · Experiment: Embedding Geometry and Consistency", [
        ("Cross-Seed Linear Map (S49)", [
            "Same model, same session, same ensemble — different random seed.",
            "Result: R²≈0.84 (MLP) / 0.71 (TC-Cont) / 0.84 (TC-Pred)",
            "",
            "Interpretation:",
            "• MLP and TC-Pred: converge to same embedding geometry across inits",
            "• TC-Cont: contrastive loss has rotational symmetry → different local optima",
            "  (low R² does NOT mean rotation — that would give R²≈1.0 under linear map)",
        ]),
        ("Cross-Session Linear Map (S49)", [
            "Same model, same ensemble, seed=42 — different training session.",
            "Cross-session ≈ cross-seed R² (0.83 vs 0.84 for MLP)",
            "",
            "Interpretation:",
            "• The embedding geometry is determined by BEHAVIORAL structure,",
            "  not by which neural population the model was trained on.",
            "• Models trained on different recording sessions converge to the",
            "  same representation of locomotor behavior.",
        ]),
        ("Cross-Ensemble Prediction (S50)", [
            "Source model trained on ensemble e_i; test on ensemble e_j activity.",
            "Self R²≈0.11; Cross R²≈0.017 (ratio ≈ 0.16)",
            "",
            "Interpretation:",
            "• Embedding geometry is universal (cross-ensemble linear map R²≈0.85)",
            "• BUT specific neural readout is target-specific:",
            "  the embedding encodes behavior, not a universal neural predictor.",
            "• Each ensemble requires its own linear readout head.",
        ]),
    ]),

]


# ══════════════════════════════════════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════════════════════════════════════
prs = Presentation(SRC)
print(f'Opened v16: {len(prs.slides)} slides')

for title, sections in APPENDIX_SLIDES:
    add_appendix_slide(prs, title, sections)

print(f'Final slide count: {len(prs.slides)}')
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
