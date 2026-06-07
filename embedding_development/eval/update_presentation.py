#!/usr/bin/env python3
"""
update_presentation.py
Replace images in the PPTX presentation with updated figures.
For single-image slides: mirrors existing image position exactly.
For multi-image slides: replaces the correct index.
For empty slides (S21, S31): places image in content area below title.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Emu

PPTX_IN  = "/mnt/c/Users/amits/Downloads/Neural_Task_Representations-19-05-2026.pptx"
PPTX_OUT = "/mnt/c/Users/amits/Downloads/Neural_Task_Representations-updated.pptx"
IMG_ROOT = "/mnt/c/Users/amits/Desktop/presentation_images"

prs    = Presentation(PPTX_IN)
slides = prs.slides

def pics(slide):
    return [s for s in slide.shapes if s.shape_type == 13]

def remove_and_add(slide, shape, img_path):
    """Remove a picture shape and add a new image at the same geometry."""
    l, t, w, h = shape.left, shape.top, shape.width, shape.height
    shape._element.getparent().remove(shape._element)
    slide.shapes.add_picture(img_path, l, t, width=w, height=h)

def add_content_area(slide, img_path, left_in, top_in, width_in, height_in):
    slide.shapes.add_picture(img_path,
        Inches(left_in), Inches(top_in),
        width=Inches(width_in), height=Inches(height_in))

# ── replacement map ───────────────────────────────────────────────────────────
# Each entry: slide_0idx → list of (rel_img_path, shape_idx_or_None)
# shape_idx=None → place in content area (slide has no existing image to mirror)
# shape_idx=N    → replace picture at that index
updates = [
    # S13: linear ensemble heatmap
    (12, [("linear/r2_heatmap.png", 0)]),
    # S14: linear ensemble bar
    (13, [("linear/r2_bar.png", 0)]),
    # S17: MLP spikes heatmap
    (16, [("mlp/r2_heatmap_spikes.png", 0)]),
    # S18: MLP spikes bar
    (17, [("mlp/r2_bar_spikes.png", 0)]),
    # S19: trial traces (top) + overlay (bottom) — already stacked full-width
    (18, [("trial_traces/traces_S01_E03.png",  0),
          ("trial_traces/overlay_S01_E03.png", 1)]),
    # S21: neuron PV evolution heatmap — no existing image, use content area
    (20, [("fix_broken_slides/S21_evolution_heatmap_gpv_neurons.png", None)]),
    # S22: neuron PV lineplot (U labels)
    (21, [("fix_broken_slides/S22_evolution_lineplot_gpv.png", 0)]),
    # S25: MLP ensemble heatmap
    (24, [("mlp/r2_heatmap.png", 0)]),
    # S26: MLP ensemble bar
    (25, [("mlp/r2_bar.png", 0)]),
    # S27: ensemble global-PV SEM heatmap
    (26, [("fix_broken_slides/S27_global_pv_sem_heatmap.png", 0)]),
    # S28: ensemble global-PV evolution heatmap
    (27, [("mlp/evolution_heatmap_gpv.png", 0)]),
    # S29: ensemble global-PV lineplot
    (28, [("fix_broken_slides/S29_evolution_lineplot_gpv.png", 0)]),
    # S31: cond-PV scatter — no existing image, use content area
    (30, [("fix_broken_slides/S31_global_vs_cond_pv_scatter.png", None)]),
    # S33: ensemble cond-PV evolution heatmap
    (32, [("fix_broken_slides/S33_evolution_heatmap_cond_pv.png", 0)]),
    # S34: IG — replace the large left image (index 2, 6.2" wide)
    (33, [("fix_broken_slides/S34_ig_sem_heatmap.png", 2)]),
    # S35: IG evolution heatmap
    (34, [("fix_broken_slides/S35_evolution_heatmap_ig.png", 0)]),
    # S36: IG evolution lineplot
    (35, [("fix_broken_slides/S36_evolution_lineplot_ig.png", 0)]),
]

for slide_idx, img_list in updates:
    slide   = slides[slide_idx]
    pic_list = pics(slide)
    for rel_path, shape_idx in img_list:
        img_path = os.path.join(IMG_ROOT, rel_path)
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {rel_path}")
            continue

        if shape_idx is None:
            # No existing image — place in body area below title
            # Use width of slide minus small margins; start below title (~1.25")
            add_content_area(slide, img_path,
                             left_in=0.25, top_in=1.25,
                             width_in=9.5, height_in=4.5)
            print(f"  S{slide_idx+1:02d}: added  {os.path.basename(img_path)}")
        else:
            # Refresh pic list (previous replace may have changed order)
            pic_list = pics(slide)
            if shape_idx >= len(pic_list):
                print(f"  S{slide_idx+1:02d}: WARN  shape_idx={shape_idx} but {len(pic_list)} images — skipping")
                continue
            remove_and_add(slide, pic_list[shape_idx], img_path)
            print(f"  S{slide_idx+1:02d}: replaced [{shape_idx}] → {os.path.basename(img_path)}")

prs.save(PPTX_OUT)
print(f"\nSaved → {PPTX_OUT}")
