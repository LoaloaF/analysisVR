#!/usr/bin/env python3
"""
build_v7.py  —  Insert 4 new analysis figure slides into v6 → v7.

v6 has 48 slides (0-based indices):
  21: Grand Mean R² Across All Four Model Architectures
  33: Attribution Agreement: MLP vs TempConv-Cont (GPV ρ≈0.95...)
  36: Embedding Consistency Across Models

Insertions:
  After slide 21 → r2_distribution_cdf.png, tempconv_delta_r2.png  (indices shift +2)
  After slide 35 (was 33, shifted +2) → joint_effect_scatter.png   (indices shift +3)
  After slide 40 (was 36, shifted +3) → cross_ensemble_consistency.png
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

SRC  = '/mnt/c/Users/amits/Desktop/ultimate_presentation_v6.pptx'
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v7.pptx',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'outputs', 'ultimate_presentation_v7.pptx'),
]

base    = os.path.dirname(os.path.abspath(__file__))
root    = os.path.join(base, '..')
IMG_DIR = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)
TITLE_H = Inches(0.55)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x23, 0x3A)
DPI     = 200


def add_figure_slide(prs, img_path, title_text):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)
    fill   = slide.background.fill
    fill.solid(); fill.fore_color.rgb = WHITE

    tb = slide.shapes.add_textbox(Inches(0.15), Inches(0.05),
                                   SLIDE_W - Inches(0.3), TITLE_H)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title_text
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = DARK

    img    = Image.open(img_path)
    fw, fh = img.width / DPI, img.height / DPI
    ct     = TITLE_H + Inches(0.05)
    ch     = (SLIDE_H - ct - Inches(0.08)) / 914400
    cw     = (SLIDE_W - Inches(0.10)) / 914400
    sc     = min(cw / fw, ch / fh)
    pw, ph = Inches(fw * sc), Inches(fh * sc)
    left   = (SLIDE_W - pw) // 2
    top    = ct + Emu(int((ch - fh * sc) / 2 * 914400))
    slide.shapes.add_picture(img_path, left, top, pw, ph)

    cap = slide.shapes.add_textbox(SLIDE_W - Inches(4.0), SLIDE_H - Inches(0.22),
                                    Inches(3.9), Inches(0.20))
    p2  = cap.text_frame.paragraphs[0]
    p2.alignment = PP_ALIGN.RIGHT
    r2  = p2.add_run()
    r2.text = os.path.basename(img_path)
    r2.font.size = Pt(9); r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return len(prs.slides) - 1


def move_slide(prs, from_idx, to_idx):
    """Move a slide from from_idx to to_idx in the deck (0-based)."""
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el = children.pop(from_idx)
    children.insert(to_idx, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)


prs = Presentation(SRC)
n0  = len(prs.slides)
print(f'Opened v6: {n0} slides')

# ── Insert in reverse order so earlier insertions don't shift later targets ──
# We insert from LAST to FIRST so indices remain stable.

# 4. cross_ensemble_consistency  → after slide 39 (embedding consistency, shifted +3)
idx4 = add_figure_slide(prs, os.path.join(IMG_DIR, 'cross_ensemble_consistency.png'),
                        'Cross-Ensemble Attribution Consistency Within Sessions')
move_slide(prs, idx4, 40)   # insert after slide 39 (0-based), so position 40
print(f'  Inserted cross_ensemble_consistency at position 40')

# 3. joint_effect_scatter → after slide 35 (attribution agreement, shifted +2)
idx3 = add_figure_slide(prs, os.path.join(IMG_DIR, 'joint_effect_scatter.png'),
                        'ML Captures Joint Effects — No Single Feature Predicts')
move_slide(prs, idx3, 36)
print(f'  Inserted joint_effect_scatter at position 36')

# 2. tempconv_delta_r2 → after slide 22 (shifted after r2_cdf insertion, +1)
idx2 = add_figure_slide(prs, os.path.join(IMG_DIR, 'tempconv_delta_r2.png'),
                        'TempConv vs MLP: Architecture-Independent Representations')
move_slide(prs, idx2, 23)
print(f'  Inserted tempconv_delta_r2 at position 23')

# 1. r2_distribution_cdf → after slide 21 (Grand Mean R²)
idx1 = add_figure_slide(prs, os.path.join(IMG_DIR, 'r2_distribution_cdf.png'),
                        'R² Distribution: Claim Scope and Valid Pair Thresholds')
move_slide(prs, idx1, 22)
print(f'  Inserted r2_distribution_cdf at position 22')

print(f'\nFinal slide count: {len(prs.slides)}')

for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved → {out}')
print('Done.')
