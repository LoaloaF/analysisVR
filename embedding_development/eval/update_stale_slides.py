#!/usr/bin/env python3
"""
update_stale_slides.py

Replace images in stale slides of Neural_Task_Representations-updated.pptx:
  S10 (idx 9):  variance_vs_mse.png [0], variance_vs_r2.png [1]
  S30 (idx 29): v3_cond_pv_comparison.png [0]
  S37 (idx 36): cross_seed_consistency.png [0]
  S38 (idx 37): trend_noise_decomposition.png [0]
  S64-S66: comparison slides added by fix_story_and_cebra.py
           (found by title text, replace single image)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

# Standard R² heatmap layout: left text box + right image
R2_TXT_L, R2_TXT_T, R2_TXT_W, R2_TXT_H = Inches(0.25), Inches(1.15), Inches(3.0),  Inches(4.35)
R2_IMG_L, R2_IMG_T, R2_IMG_W, R2_IMG_H = Inches(3.35), Inches(1.15), Inches(6.15), Inches(4.35)


def standardise_r2_slide(slide, img_path):
    """Enforce the standard R² heatmap layout on a slide.
    - Keeps existing content text box (repositions it); adds empty one if absent.
    - Removes all pictures and inserts the new image at the standard slot.
    - Never touches title or page-number placeholders.
    """
    # Identify content placeholder (not title idx=0, not page-number)
    def _is_content_ph(s):
        try:
            return (s.shape_type == 14
                    and s.placeholder_format is not None
                    and s.placeholder_format.idx not in (0, 12))
        except ValueError:
            return False

    content_ph = next((s for s in slide.shapes if _is_content_ph(s)), None)

    # Remove all existing pictures
    for pic in [s for s in slide.shapes if s.shape_type == 13]:
        pic._element.getparent().remove(pic._element)

    # Reposition or create text box
    if content_ph is not None:
        content_ph.left   = R2_TXT_L
        content_ph.top    = R2_TXT_T
        content_ph.width  = R2_TXT_W
        content_ph.height = R2_TXT_H
    else:
        txBox = slide.shapes.add_textbox(R2_TXT_L, R2_TXT_T, R2_TXT_W, R2_TXT_H)
        txBox.text_frame.word_wrap = True

    # Add image at standard position
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, R2_IMG_L, R2_IMG_T,
                                 width=R2_IMG_W, height=R2_IMG_H)
        print(f"  standardised → {os.path.basename(img_path)}")
    else:
        print(f"  SKIP (missing): {img_path}")

PPTX_PATH = "/mnt/c/Users/amits/Downloads/Neural_Task_Representations-updated.pptx"
base      = os.path.dirname(os.path.abspath(__file__))
root      = os.path.join(base, "..")

DESKTOP   = "/mnt/c/Users/amits/Desktop/presentation_images"

prs    = Presentation(PPTX_PATH)
slides = prs.slides


def pics(slide):
    return [s for s in slide.shapes if s.shape_type == 13]


def replace_images_on_slide(slide, replacements):
    """replacements: list of (shape_idx, img_path) sorted by shape_idx ascending.
    Collects all geometries before any removal so indices don't shift."""
    pic_list = pics(slide)
    geoms = {}
    to_remove = []
    for shape_idx, img_path in replacements:
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {img_path}")
            continue
        if shape_idx >= len(pic_list):
            print(f"  WARN: shape_idx={shape_idx} but only {len(pic_list)} images")
            continue
        s = pic_list[shape_idx]
        geoms[shape_idx] = (s.left, s.top, s.width, s.height, img_path, s._element)
        to_remove.append(s._element)
    for elem in to_remove:
        elem.getparent().remove(elem)
    for shape_idx, (l, t, w, h, img_path, _) in sorted(geoms.items()):
        slide.shapes.add_picture(img_path, l, t, width=w, height=h)
        print(f"  Replaced [{shape_idx}] → {os.path.basename(img_path)}")


# ── Fixed-index replacements ──────────────────────────────────────────────────
spk_mlp = os.path.join(root, "outputs", "mlps",   "spikes_multiseed")
ens_mlp = os.path.join(root, "outputs", "mlps",   "ensembles_multiseed")
spk_lin = os.path.join(root, "outputs", "linear", "spikes_multiseed")
ens_lin = os.path.join(root, "outputs", "linear", "ensembles_multiseed")
traces  = os.path.join(root, "outputs", "trial_traces_comparison")

fixed_updates = [
    # S10: variance scatter (z-scored FR)
    (9,  0, os.path.join(spk_mlp, "variance_vs_mse.png")),
    (9,  1, os.path.join(spk_mlp, "variance_vs_r2.png")),
    # S14: linear ensemble R² bar
    (13, 0, os.path.join(ens_lin, "r2_bar.png")),
    # S18: MLP spikes R² bar
    (17, 0, os.path.join(spk_mlp, "r2_bar.png")),
    # S19: trial traces (top) + overlay (bottom)
    (18, 0, os.path.join(traces,  "traces_S01_E03.png")),
    (18, 1, os.path.join(traces,  "overlay_S01_E03.png")),
    # S21: spikes GPV evolution heatmap
    (20, 0, os.path.join(spk_mlp, "evolution_heatmap_gpv.png")),
    # S22: spikes GPV evolution lineplot
    (21, 0, os.path.join(spk_mlp, "evolution_lineplot_gpv.png")),
    # S26: MLP ensemble R² bar
    (25, 0, os.path.join(ens_mlp, "r2_bar.png")),
    # S27: ensemble global-PV SEM heatmap
    (26, 0, os.path.join(ens_mlp, "global_pv_sem_heatmap.png")),
    # S28: ensemble GPV evolution heatmap
    (27, 0, os.path.join(ens_mlp, "evolution_heatmap_gpv.png")),
    # S29: ensemble GPV evolution lineplot
    (28, 0, os.path.join(ens_mlp, "evolution_lineplot_gpv.png")),
    # S30: cond PV comparison
    (29, 0, os.path.join(root, "outputs", "residual_choice", "v3_cond_pv_comparison.png")),
    # S31: cond-PV vs global scatter
    (30, 0, os.path.join(ens_mlp, "global_vs_cond_pv_scatter.png")),
    # S33: ensemble cond-PV evolution heatmap
    (32, 0, os.path.join(ens_mlp, "evolution_heatmap_cond_pv.png")),
    # S35: IG evolution heatmap
    (34, 0, os.path.join(ens_mlp, "evolution_heatmap_ig.png")),
    # S36: IG evolution lineplot
    (35, 0, os.path.join(ens_mlp, "evolution_lineplot_ig.png")),
    # S37: cross-seed consistency
    (36, 0, os.path.join(ens_mlp, "cross_seed_consistency.png")),
    # S38: trend-noise decomposition
    (37, 0, os.path.join(ens_mlp, "trend_noise_decomposition.png")),
]

# Group by slide index so multi-image slides are batched correctly
from collections import defaultdict
by_slide = defaultdict(list)
for slide_idx, shape_idx, img_path in fixed_updates:
    by_slide[slide_idx].append((shape_idx, img_path))

# S21: reformat to match S28 layout (remove text placeholder, use S28 image geometry)
_s21 = slides[20]
_s21_text_ph = next(
    (s for s in _s21.shapes
     if s.shape_type == 14 and s.top > Inches(1) and s.height > Inches(1.5)),
    None)
if _s21_text_ph is not None:
    _s21_text_ph._element.getparent().remove(_s21_text_ph._element)
    print("S21: removed text placeholder")
# Replace image using S28 geometry instead of the original oversized slot
_s21_img_path = os.path.join(spk_mlp, "evolution_heatmap_gpv.png")
if os.path.exists(_s21_img_path):
    for pic in [s for s in _s21.shapes if s.shape_type == 13]:
        pic._element.getparent().remove(pic._element)
    _s21.shapes.add_picture(_s21_img_path,
        Inches(0.09), Inches(1.17), width=Inches(9.78), height=Inches(3.94))
    print(f"S21: replaced image with S28 geometry → {os.path.basename(_s21_img_path)}")
by_slide.pop(20, None)   # handled above, skip in main loop

for slide_idx in sorted(by_slide):
    print(f"S{slide_idx+1}:")
    replace_images_on_slide(slides[slide_idx], by_slide[slide_idx])

# ── Standardised R² heatmap slides (text box left + image right) ─────────────
r2_heatmap_slides = [
    (12, os.path.join(ens_lin, "r2_heatmap.png"),  "S13 linear ensemble"),
    (16, os.path.join(spk_mlp, "r2_heatmap.png"),  "S17 MLP spikes"),
    (24, os.path.join(ens_mlp, "r2_heatmap.png"),  "S25 MLP ensemble"),
]
for slide_idx, img_path, label in r2_heatmap_slides:
    print(f"{label}:")
    standardise_r2_slide(slides[slide_idx], img_path)

# ── Title-based replacements for TempConv comparison slides ──────────────────────
# These were added by fix_story_and_cebra.py; find by title text
comparison_dir   = os.path.join(root, "outputs", "cebra_comparison")
desktop_comp_dir = os.path.join(DESKTOP, "comparison")

title_to_img = {
    "TempConv vs MLP: Delta R²":               os.path.join(comparison_dir, "delta_r2.png"),
    "Model Comparison: Good Ensembles per Session":
        os.path.join(comparison_dir, "per_session_good_ensembles.png"),
    "MLP vs TempConv-Pred Ensemble R²":        os.path.join(comparison_dir, "scatter_mlp_vs_tempconv_pred.png"),
    "Model Comparison at Two Thresholds":   os.path.join(comparison_dir, "two_thresholds_scatter.png"),
}

# Also copy cebra r² bars/heatmaps if updated
cebra_dir      = os.path.join(root, "outputs", "cebra_eval", "ensembles")
cebra_pred_dir = os.path.join(root, "outputs", "cebra_pred_eval", "ensembles")

# TempConv r² bars go through the normal title-based path
title_to_img.update({
    "TempConv Predictive — Ensemble R²":
        os.path.join(cebra_pred_dir, "r2_bar.png"),
})

# TempConv r² heatmaps get the standardised layout
r2_heatmap_titles = {
    "TempConv Contrastive — Session×Ensemble R²":
        os.path.join(cebra_dir,      "r2_heatmap.png"),
    "TempConv Predictive — Session×Ensemble R²":
        os.path.join(cebra_pred_dir, "r2_heatmap.png"),
}

for slide in slides:
    title_text = next((s.text.strip() for s in slide.shapes
                       if s.has_text_frame and s.text.strip() in r2_heatmap_titles), None)
    if title_text:
        print(f"'{title_text}':")
        standardise_r2_slide(slide, r2_heatmap_titles[title_text])

for slide in slides:
    if slide._element.get('show', '1') == '0':
        continue
    def _is_title_ph(s):
        try:
            return s.has_text_frame and s.placeholder_format is not None and s.placeholder_format.idx == 0
        except ValueError:
            return False
    title_shape = next((s for s in slide.shapes if _is_title_ph(s)), None)
    if title_shape is None:
        # fall back to any text box whose text is in our map
        title_shape = next((s for s in slide.shapes
                            if s.has_text_frame
                            and s.text.strip() in title_to_img), None)
    if title_shape is None:
        continue
    title = title_shape.text.strip()
    if title in title_to_img:
        img_path = title_to_img[title]
        if not os.path.exists(img_path):
            print(f"S'{title}': SKIP (missing) {img_path}")
            continue
        pic_list = pics(slide)
        if not pic_list:
            slide.shapes.add_picture(img_path,
                Inches(0.25), Inches(1.25), width=Inches(9.5), height=Inches(4.5))
            print(f"  '{title}': added image")
        else:
            replace_images_on_slide(slide, [(0, img_path)])
            print(f"  '{title}': updated")

prs.save(PPTX_PATH)
print(f"\nSaved → {PPTX_PATH}")
