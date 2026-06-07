#!/usr/bin/env python3
"""
update_cleanup_slides.py

Replace images in Neural_Task_Representation_cleanup.pptx (47 slides after
update_example_slides.py has been run — run that script first to insert the
3 example slides S18-20).

SKIP S02 (idx=1) per user instruction.

Slide → image mapping (indices assume 47-slide deck):
  S10  (idx=9):  variance_vs_mse [0], variance_vs_r2 [1]
  S12  (idx=11): linear ens r2_heatmap  (standardise_r2_slide)
  S13  (idx=12): linear ens r2_bar
  S16  (idx=15): spikes mlp r2_heatmap  (standardise_r2_slide)
  S17  (idx=16): spikes mlp r2_bar
  S18  (idx=17): example overlay — Session 2024-11-14  E03  Trial 22
  S19  (idx=18): example overlay — Session 2024-11-28  E04  Trial 2
  S20  (idx=19): example overlay — Session 2025-01-24  E05  Trial 31
  S23  (idx=22): spikes gpv evolution lineplot (S27-style geometry)
  S24  (idx=23): ens mlp r2_heatmap  (standardise_r2_slide)
  S25  (idx=24): ens mlp r2_bar
  S26  (idx=25): ens mlp global_pv_sem_heatmap
  S27  (idx=26): ens mlp evolution_heatmap_gpv
  S28  (idx=27): ens mlp evolution_lineplot_gpv
  S29  (idx=28): intentionally blank
  S31  (idx=30): global_vs_cond_pv_scatter (Cond-PV Plotted Against Global)
  S32  (idx=31): ens mlp evolution_heatmap_cond_pv
  S34  (idx=33): ens mlp evolution_heatmap_ig
  S35  (idx=34): ens mlp evolution_lineplot_ig
  S36  (idx=35): cross_seed_consistency
  S37  (idx=36): trend_noise_decomposition
  S39  (idx=38): TempConv-Cont r2_heatmap (standardise_r2_slide)
  S40  (idx=39): TempConv-Cont r2_bar  (fix broken geometry)
  S41  (idx=40): TempConv-Pred r2_heatmap  (standardise_r2_slide)
  S42  (idx=41): TempConv-Pred r2_bar
  S43  (idx=42): comparison delta_r2
  S44  (idx=43): comparison per_session_good_ensembles
  S45  (idx=44): comparison scatter_mlp_vs_tempconv_pred
  S46  (idx=45): comparison two_thresholds_scatter
"""
import os
from collections import defaultdict
from pptx import Presentation
from pptx.util import Inches

# ── Standard R² heatmap layout constants ─────────────────────────────────────
R2_TXT_L, R2_TXT_T, R2_TXT_W, R2_TXT_H = Inches(0.25), Inches(1.15), Inches(3.0),  Inches(4.35)
R2_IMG_L, R2_IMG_T, R2_IMG_W, R2_IMG_H = Inches(3.35), Inches(1.15), Inches(6.15), Inches(4.35)

# ── Evolution heatmap geometry (S25 style) ────────────────────────────────────
EVO_L, EVO_T, EVO_W, EVO_H = Inches(0.09), Inches(1.17), Inches(9.78), Inches(3.94)

# ── Bar chart geometry — matches MLP bar slides (S13/S17/S23) ────────────────
BAR_L, BAR_T, BAR_W, BAR_H = Inches(0.0),  Inches(1.4),  Inches(10.0), Inches(3.1)

PPTX_PATH = "/mnt/c/Users/amits/Downloads/Neural_Task_Representation_cleanup.pptx"
base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, "..")

spk_mlp  = os.path.join(root, "outputs", "mlps",   "spikes_multiseed")
ens_mlp  = os.path.join(root, "outputs", "mlps",   "ensembles_multiseed")
ens_lin  = os.path.join(root, "outputs", "linear", "ensembles_multiseed")
traces   = os.path.join(root, "outputs", "trial_traces_comparison")
res_ch   = os.path.join(root, "outputs", "residual_choice")
cebra_c  = os.path.join(root, "outputs", "cebra_eval",      "ensembles")
cebra_p  = os.path.join(root, "outputs", "cebra_pred_eval", "ensembles")
cebra_cmp= os.path.join(root, "outputs", "cebra_comparison")

prs    = Presentation(PPTX_PATH)
slides = prs.slides

if len(slides) < 47:
    print("WARNING: expected 47 slides (after update_example_slides.py) "
          f"but found {len(slides)}. Run update_example_slides.py first.")



def pics(slide):
    return [s for s in slide.shapes if s.shape_type == 13]


def replace_images_on_slide(slide, replacements):
    """replacements: list of (shape_idx, img_path).
    Collects all geometries before any removal to avoid index shifting."""
    pic_list = pics(slide)
    geoms    = {}
    to_remove = []
    for shape_idx, img_path in replacements:
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {img_path}")
            continue
        if shape_idx >= len(pic_list):
            print(f"  WARN: shape_idx={shape_idx} but only {len(pic_list)} images on slide")
            continue
        s = pic_list[shape_idx]
        geoms[shape_idx] = (s.left, s.top, s.width, s.height, img_path, s._element)
        to_remove.append(s._element)
    for elem in to_remove:
        elem.getparent().remove(elem)
    for shape_idx, (l, t, w, h, img_path, _) in sorted(geoms.items()):
        slide.shapes.add_picture(img_path, l, t, width=w, height=h)
        print(f"  Replaced [{shape_idx}] → {os.path.basename(img_path)}")


def standardise_r2_slide(slide, img_path):
    """Enforce standard R² heatmap layout: text box left, image right.
    Keeps or creates content text box; never touches title or page-number."""
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

    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, R2_IMG_L, R2_IMG_T,
                                 width=R2_IMG_W, height=R2_IMG_H)
        print(f"  standardised → {os.path.basename(img_path)}")
    else:
        print(f"  SKIP (missing): {img_path}")


def replace_full_slot(slide, img_path, l, t, w, h):
    """Remove all pictures on slide and add one at given geometry."""
    for pic in [s for s in slide.shapes if s.shape_type == 13]:
        pic._element.getparent().remove(pic._element)
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, l, t, width=w, height=h)
        print(f"  placed → {os.path.basename(img_path)}")
    else:
        print(f"  SKIP (missing): {img_path}")


# ── Fixed-index replacements (batch by slide) ─────────────────────────────────
# NOTE: all indices ≥ 18 are shifted +2 relative to the original 45-slide deck
#       because update_example_slides.py inserted slides 19 and 20.
fixed_updates = [
    # S10: variance scatter plots
    (9,  0, os.path.join(spk_mlp, "variance_vs_mse.png")),
    (9,  1, os.path.join(spk_mlp, "variance_vs_r2.png")),
    # S13: linear ensemble R² bar
    (12, 0, os.path.join(ens_lin, "r2_bar.png")),
    # S17: MLP spikes R² bar
    (16, 0, os.path.join(spk_mlp, "r2_bar.png")),
    # S25: MLP ensemble R² bar  (was S23, +2)
    (24, 0, os.path.join(ens_mlp, "r2_bar.png")),
    # S26: global PV semantic heatmap  (was S24, +2)
    (25, 0, os.path.join(ens_mlp, "global_pv_sem_heatmap.png")),
    # S27: ensemble GPV evolution heatmap  (was S25, +2)
    (26, 0, os.path.join(ens_mlp, "evolution_heatmap_gpv.png")),
    # S28: ensemble GPV evolution lineplot  (was S26, +2)
    (27, 0, os.path.join(ens_mlp, "evolution_lineplot_gpv.png")),
    # S29: intentionally blank (cond-PV analysis pending session selection fix)
    # S31: Cond-PV Plotted Against Global → scatter  (was S29, +2)
    (30, 0, os.path.join(ens_mlp, "global_vs_cond_pv_scatter.png")),
    # S32: cond-PV evolution heatmap  (was S30, +2)
    (31, 0, os.path.join(ens_mlp, "evolution_heatmap_cond_pv.png")),
    # S34: IG evolution heatmap  (was S32, +2)
    (33, 0, os.path.join(ens_mlp, "evolution_heatmap_ig.png")),
    # S35: IG evolution lineplot  (was S33, +2)
    (34, 0, os.path.join(ens_mlp, "evolution_lineplot_ig.png")),
    # S36: cross-seed consistency  (was S34, +2)
    (35, 0, os.path.join(ens_mlp, "cross_seed_consistency.png")),
    # S37: trend-noise decomposition  (was S35, +2)
    (36, 0, os.path.join(ens_mlp, "trend_noise_decomposition.png")),
    # S43–S46: comparison slides  (was S42-S45, +2)
    (42, 0, os.path.join(cebra_cmp, "delta_r2.png")),
    (43, 0, os.path.join(cebra_cmp, "per_session_good_ensembles.png")),
    (44, 0, os.path.join(cebra_cmp, "scatter_mlp_vs_tempconv_pred.png")),
    (45, 0, os.path.join(cebra_cmp, "two_thresholds_scatter.png")),
]

by_slide = defaultdict(list)
for slide_idx, shape_idx, img_path in fixed_updates:
    by_slide[slide_idx].append((shape_idx, img_path))

# ── S18-20 (idx=17-19): example overlay slides — handled by update_example_slides.py
#    Re-stamp images here so update_cleanup_slides.py stays authoritative.
_example_overlays = [
    (17, "overlay_S01_E03.png"),
    (18, "overlay_S07_E04.png"),
    (19, "overlay_S25_E05.png"),
]
_IMG_L, _IMG_W, _IMG_H = 0, 9_144_000, round(9_144_000 / 3.5)
_IMG_T = (1_053_478 + 5_143_500) // 2 - _IMG_H // 2
from pptx.util import Emu as _Emu
for _idx, _fname in _example_overlays:
    _img_path = os.path.join(traces, _fname)
    print(f"S{_idx+1} (example overlay):")
    replace_full_slot(slides[_idx], _img_path,
                      _Emu(_IMG_L), _Emu(_IMG_T), _Emu(_IMG_W), _Emu(_IMG_H))
by_slide.pop(17, None)
by_slide.pop(18, None)
by_slide.pop(19, None)

# ── S23 (idx=22): spikes GPV evolution — use EVO geometry  (was S21, +2) ──────
print("S23 (spikes evolution):")
replace_full_slot(slides[22],
                  os.path.join(spk_mlp, "evolution_lineplot_gpv.png"),
                  EVO_L, EVO_T, EVO_W, EVO_H)
by_slide.pop(22, None)  # handled above

# ── S40 (idx=39) and S42 (idx=41): TempConv bar slides — enforce MLP geometry ────
#    (was S38/idx=37 and S40/idx=39, +2)
print("S40 (TempConv-Cont bar):")
replace_full_slot(slides[39],
                  os.path.join(cebra_c, "r2_bar.png"),
                  BAR_L, BAR_T, BAR_W, BAR_H)
by_slide.pop(39, None)

print("S42 (TempConv-Pred bar):")
replace_full_slot(slides[41],
                  os.path.join(cebra_p, "r2_bar.png"),
                  BAR_L, BAR_T, BAR_W, BAR_H)
by_slide.pop(41, None)

# ── Run fixed-index replacements ─────────────────────────────────────────────
for slide_idx in sorted(by_slide):
    print(f"S{slide_idx+1}:")
    replace_images_on_slide(slides[slide_idx], by_slide[slide_idx])

# ── Standardised R² heatmap slides ───────────────────────────────────────────
# Indices ≥ 18 shifted +2 vs. the original 45-slide deck.
r2_heatmap_slides = [
    (11, os.path.join(ens_lin, "r2_heatmap.png"),  "S12 linear ens"),
    (15, os.path.join(spk_mlp, "r2_heatmap.png"),  "S16 MLP spikes"),
    (23, os.path.join(ens_mlp, "r2_heatmap.png"),  "S24 MLP ens"),        # was 21
    (38, os.path.join(cebra_c, "r2_heatmap.png"),  "S39 TempConv-Cont"), # was 36
    (40, os.path.join(cebra_p, "r2_heatmap.png"),  "S41 TempConv-Pred"),     # was 38
]
for slide_idx, img_path, label in r2_heatmap_slides:
    print(f"{label}:")
    standardise_r2_slide(slides[slide_idx], img_path)

prs.save(PPTX_PATH)
print(f"\nSaved → {PPTX_PATH}")
