#!/usr/bin/env python3
"""
fix_story_and_cebra.py

1. Hide S23-S24 (neuron embedding consistency — wrong focus for ensemble story)
2. Unhide S57-S59 (TempConv slides)
3. Replace S59's old GoF heatmap with tempconv_cont_r2_bar.png
4. Add new slides after S59 for remaining TempConv images + comparison section
"""
import os, copy
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

PPTX_IN  = "/mnt/c/Users/amits/Downloads/Neural_Task_Representations-updated.pptx"
PPTX_OUT = "/mnt/c/Users/amits/Downloads/Neural_Task_Representations-updated.pptx"
IMG_ROOT = "/mnt/c/Users/amits/Desktop/presentation_images"

prs    = Presentation(PPTX_IN)
slides = prs.slides

# ── helpers ───────────────────────────────────────────────────────────────────
def set_hidden(slide, hidden: bool):
    el = slide._element
    if hidden:
        el.set('show', '0')
    else:
        if 'show' in el.attrib:
            del el.attrib['show']

def pics(slide):
    return [s for s in slide.shapes if s.shape_type == 13]

def replace_image(slide, shape_idx, img_path):
    pic_list = pics(slide)
    if shape_idx >= len(pic_list):
        print(f"  WARNING: shape_idx={shape_idx} but only {len(pic_list)} images")
        return
    s = pic_list[shape_idx]
    l, t, w, h = s.left, s.top, s.width, s.height
    s._element.getparent().remove(s._element)
    slide.shapes.add_picture(img_path, l, t, width=w, height=h)

def _copy_slide_xml(prs_obj, source_slide):
    """Duplicate a slide's XML into a new slide at the end of the deck."""
    template_layout = source_slide.slide_layout
    new_slide = prs_obj.slides.add_slide(template_layout)
    # Replace the new slide's spTree with a deep copy of the source's spTree
    src_sp_tree = source_slide.shapes._spTree
    tgt_sp_tree = new_slide.shapes._spTree
    # Clear target spTree children (except the first two: cNvGrpSpPr, grpSpPr)
    for child in list(tgt_sp_tree)[2:]:
        tgt_sp_tree.remove(child)
    for child in list(src_sp_tree)[2:]:
        tgt_sp_tree.append(copy.deepcopy(child))
    return new_slide

def add_image_slide(prs_obj, template_slide_idx, title_text, img_path,
                    left=0.25, top=1.25, width=9.5, height=4.5):
    """Add a new slide (copy of template) with title + image."""
    src   = prs_obj.slides[template_slide_idx]
    slide = _copy_slide_xml(prs_obj, src)
    # Remove all existing pictures from the copy
    for p in list(pics(slide)):
        p._element.getparent().remove(p._element)
    # Set title placeholder if present
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 0:
            tf = shape.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = title_text
            break
    # Add the image
    slide.shapes.add_picture(img_path,
        Inches(left), Inches(top), width=Inches(width), height=Inches(height))
    return slide

# ── 1. Hide S23 and S24 (0-indexed: 22, 23) ──────────────────────────────────
for idx in [22, 23]:
    set_hidden(slides[idx], True)
    print(f"  Hidden S{idx+1:02d}")

# ── 2. Unhide S57, S58, S59 (0-indexed: 56, 57, 58) ─────────────────────────
for idx in [56, 57, 58]:
    set_hidden(slides[idx], False)
    print(f"  Unhidden S{idx+1:02d}")

# ── 3. Replace S59's image with TempConv contrastive R² bar ─────────────────────
s59_img = os.path.join(IMG_ROOT, "cebra/tempconv_cont_r2_bar.png")
replace_image(slides[58], 0, s59_img)
print(f"  S59: replaced image → tempconv_cont_r2_bar.png")

# ── 4. Add new TempConv + comparison slides after S59 ───────────────────────────
# Use S35 (index 34) as template — it's a clean title + single image slide
TEMPLATE = 34

new_slides = [
    ("TempConv Contrastive — Session×Ensemble R²",
     "cebra/cebra_contrastive_r2_heatmap.png"),
    ("TempConv Predictive — Ensemble R²",
     "cebra/cebra_predictive_r2_bar.png"),
    ("TempConv Predictive — Session×Ensemble R²",
     "cebra/cebra_predictive_r2_heatmap.png"),
    ("TempConv vs MLP: Delta R²",
     "cebra/cebra_delta_r2.png"),
    ("Model Comparison: Good Ensembles per Session",
     "comparison/per_session_good_ensembles.png"),
    ("MLP vs TempConv-Pred Ensemble R²",
     "comparison/scatter_mlp_vs_tempconv_pred.png"),
    ("Model Comparison at Two Thresholds",
     "comparison/two_thresholds_scatter.png"),
]

for title, rel_path in new_slides:
    img_path = os.path.join(IMG_ROOT, rel_path)
    if not os.path.exists(img_path):
        print(f"  SKIP (missing): {rel_path}")
        continue
    add_image_slide(prs, TEMPLATE, title, img_path)
    print(f"  Added slide: '{title}'")

prs.save(PPTX_OUT)
print(f"\nSaved → {PPTX_OUT}")
