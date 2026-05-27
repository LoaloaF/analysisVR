#!/usr/bin/env python3
"""
build_manifest_slides.py

Generate a supplementary PPTX (manifest_slides.pptx) containing one slide per
manifest figure.  Each figure is placed at its exact native size, centred on the
10" × 5.625" slide canvas.

Usage:
    python eval/build_manifest_slides.py

Output:
    outputs/manifest_slides.pptx
    /mnt/c/Users/amits/Desktop/manifest_slides.pptx
"""
import os, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import FIG, DPI

# ─── Slide dimensions ─────────────────────────────────────────────────────────
SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)
TITLE_H = Inches(0.55)       # height of title band at top

TITLE_FONT_SIZE = Pt(18)
CAPTION_FONT_SIZE = Pt(10)

# ─── MANIFEST: (section, slide title, filename, section heading?) ──────────────
MANIFEST = [
    # ─── Section 2 — Data ──────────────────────────────────────────────────────
    ('Data',
     'Behavioral Features: 7 Continuous + 4 Categorical at 40 ms Resolution',
     'behavioral_trace.png'),

    # ─── Section 3 — Models ────────────────────────────────────────────────────
    ('Models',
     'Linear Baseline: Within-Session Ensemble R² (n=29 sessions × 23 ensembles)',
     'r2_bar_linear.png'),

    ('Models',
     'MLP Baseline: Within-Session Ensemble R²',
     'r2_bar_mlp.png'),

    ('Models',
     'MLP Cross-Seed Consistency (Pearson r on Held-out Trials)',
     'r2_consistency_bar.png'),

    ('Models',
     'Grand Mean R² Across All Four Model Architectures',
     'r2_grand_mean_bars.png'),

    ('Models',
     'Valid (R² ≥ 0.01) Pair Counts at Two Thresholds: MLP vs Linear',
     'two_thresholds_scatter.png'),

    # ─── Section 5 — Attribution ───────────────────────────────────────────────
    ('Attribution',
     'Global Permutation Variance by Feature Group × Ensemble (Sorted by Mean R²)',
     'gpv_group_ensemble_heatmap.png'),

    ('Attribution',
     'Mean |IG| Attribution by Feature Group × Ensemble',
     'ig_per_ensemble_heatmap.png'),

    ('Attribution',
     'Global PV vs Conditional PV — Points Below Diagonal Are Collinearity-Inflated',
     'global_vs_cond_pv_scatter.png'),

    ('Attribution',
     'Case Studies: E07 (Speed-Dominated) vs E23 (Head-Angle Dominated)',
     'case_studies_e07_e23.png'),

    # ─── Section 6 — Head Angle ────────────────────────────────────────────────
    ('Head Angle',
     'Top Pairs: Head Angle × Head Angular Velocity (Colored by z-Scored Ensemble Activity)',
     'head_angle_scatter_2x2.png'),

    ('Head Angle',
     'E18 Tuning Curve Stability — Preferred Angle Consistent Across Sessions',
     'head_angle_stability.png'),

    ('Head Angle',
     'E18 Tuning Shape Variety Across Six Representative Sessions',
     'head_angle_tuning_6panel.png'),

    ('Head Angle',
     'Position Tuning Curves for Top 4 Pairs (MLP Cannot Decode Position via Attribution)',
     'position_tuning.png'),

    # ─── Section 7 — Frequency Analysis ───────────────────────────────────────
    ('Frequency',
     'MLP Fails on Fast Neural Fluctuations; TempConv-Pred Captures Them',
     'trend_noise_comparison.png'),

    # ─── Section 8 — TempConv / CEBRA ─────────────────────────────────────────
    ('TempConv / CEBRA',
     'Attribution Agreement: MLP vs TempConv-Cont (GPV ρ=0.955, IG ρ=0.936)',
     'attribution_agreement_scatter.png'),

    ('TempConv / CEBRA',
     'Group-Level Attribution Profile: MLP vs TempConv-Cont (GPV and IG)',
     'group_attribution_comparison.png'),

    # ─── Section 9 — ML vs Naive Analysis ────────────────────────────────────
    ('ML vs Naive',
     'Part 1 — Validation: ML Attribution Tracks Naive Data-Effect Size',
     'ml_vs_naive_scatter.png'),

    ('ML vs Naive',
     'Part 2 — Nonlinearity: ML Captures Tuning That Spearman ρ Misses',
     'nonlinearity_advantage.png'),

    ('ML vs Naive',
     'Part 3 — ML vs GLM: MLP Outperforms Linear, Especially for Attribution-Flagged Pairs',
     'mlp_vs_linear_r2.png'),

    ('ML vs Naive',
     'Part 4 — Ablation Proof: Single-Feature MLP Predicts Where GLM Sees Nothing',
     'ablation_proof.png'),
]

# ─── Search dirs (same as validate_figures.py) ────────────────────────────────
SEARCH_DIRS = [
    './outputs/cebra_comparison',
    './outputs/mlps/ensembles_multiseed',
    './outputs/mlps/ml_vs_naive',
    './outputs/cebra_eval/ensembles',
    './outputs/temporal_advantage',
]

# ─── Build figure-path index ──────────────────────────────────────────────────
found = {}
for d in SEARCH_DIRS:
    if not os.path.isdir(d):
        continue
    for fname in os.listdir(d):
        if fname.endswith('.png'):
            found[fname] = os.path.join(d, fname)

# ─── Create PPTX ──────────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]   # blank layout

current_section = None
sections_seen = set()

for section, title_text, filename in MANIFEST:
    if filename not in found:
        print(f'  [MISSING] {filename} — skipping')
        continue

    img_path = found[filename]
    img      = Image.open(img_path)
    px_w, px_h = img.size
    fig_w_in = px_w / DPI
    fig_h_in = px_h / DPI

    # ── add section divider slide if section changes ──────────────────────────
    if section not in sections_seen:
        sections_seen.add(section)
        div_slide = prs.slides.add_slide(blank_layout)
        # dark background
        bg = div_slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(0x1A, 0x23, 0x3A)

        txBox = div_slide.shapes.add_textbox(
            Inches(1), Inches(2.0), Inches(8), Inches(1.5))
        tf = txBox.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = section
        run.font.size = Pt(36)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # ── add figure slide ──────────────────────────────────────────────────────
    slide = prs.slides.add_slide(blank_layout)

    # Title bar
    title_box = slide.shapes.add_textbox(
        Inches(0.15), Inches(0.05), SLIDE_W - Inches(0.3), TITLE_H)
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title_text
    run.font.size = TITLE_FONT_SIZE
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1A, 0x23, 0x3A)

    # Content area: centre figure below title bar
    content_top = TITLE_H + Inches(0.05)
    content_h_in = (SLIDE_H - content_top - Inches(0.08)) / 914400.0  # EMU to inches
    content_w_in = (SLIDE_W - Inches(0.10)) / 914400.0

    # Scale to fit content area while preserving aspect ratio
    scale = min(content_w_in / fig_w_in, content_h_in / fig_h_in)
    placed_w = Inches(fig_w_in * scale)
    placed_h = Inches(fig_h_in * scale)

    left = (SLIDE_W - placed_w) // 2
    top  = content_top + Emu(int((content_h_in - fig_h_in * scale) / 2 * 914400))

    slide.shapes.add_picture(img_path, left, top, placed_w, placed_h)

    # Caption: filename in lower-right corner
    cap = slide.shapes.add_textbox(
        SLIDE_W - Inches(3.5), SLIDE_H - Inches(0.22),
        Inches(3.4), Inches(0.20))
    tf_c = cap.text_frame
    p_c  = tf_c.paragraphs[0]
    p_c.alignment = PP_ALIGN.RIGHT
    run_c = p_c.add_run()
    run_c.text = filename
    run_c.font.size = CAPTION_FONT_SIZE
    run_c.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    print(f'  {section}  /  {filename}  [{fig_w_in:.2f}" × {fig_h_in:.2f}"]')

# ─── Save ─────────────────────────────────────────────────────────────────────
out_paths = [
    './outputs/manifest_slides.pptx',
    '/mnt/c/Users/amits/Desktop/manifest_slides.pptx',
]
for p in out_paths:
    os.makedirs(os.path.dirname(p) if os.path.dirname(p) else '.', exist_ok=True)
    prs.save(p)
    n_slides = len(prs.slides)
    print(f'\nSaved {n_slides}-slide PPTX → {p}')
