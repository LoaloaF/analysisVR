#!/usr/bin/env python3
"""
build_v26.py  —  Replace single E07/E23 validation slide with 3 focused slides.

v25 (84) → v26 (86 slides, +2):
  Replace S62 (single covariate validation figure) with:
    S62  "Slide 1: The Supervisor's Case — Cue and Choice Do Show a Signal"
         Figure: e07_e23_supervisors_signal.png
    S63  "Slide 2: Our Model Detects It — And Attributes More to Our Primary Variable"
         Figure: e07_e23_gpv_comparison.png
    S64  "Slide 3: Our Variable Co-varies With Theirs (E07) / Captures Independent Tuning (E23)"
         Figure: e07_e23_covariate_explanation.png
  Old single S62 (2×2 e07_e23_covariate_validation.png) removed.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
adir = os.path.join(root, 'outputs', 'ablation_vs_attribution')

SRC  = os.path.join(root, 'outputs', 'ultimate_presentation_v25.pptx')
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v26.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v26.pptx'),
]

C_BG    = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE = RGBColor(0x1A, 0x23, 0x3A)


def move_slide(prs, from_idx, to_idx):
    xml = prs.slides._sldIdLst
    els = list(xml); el = els.pop(from_idx); els.insert(to_idx, el)
    for c in list(xml): xml.remove(c)
    for c in els:        xml.append(c)


def remove_slide(prs, idx):
    slide = prs.slides[idx]; slide_part = slide.part; prs_part = prs.slides.part
    rId = None
    for rel_id, rel in prs_part.rels.items():
        try:
            if rel.target_part is slide_part: rId = rel_id; break
        except Exception: pass
    if rId: prs_part.rels.pop(rId)
    xml = prs.slides._sldIdLst; xml.remove(list(xml)[idx])


def find_idx(prs, fragment):
    for i, sl in enumerate(prs.slides):
        for sh in sl.shapes:
            if sh.has_text_frame and fragment.lower() in sh.text_frame.text.lower():
                return i
    return None


def add_figure_slide(prs, title_text, img_path,
                     fig_top=0.62, fig_h=4.20):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill  = slide.background.fill; fill.solid(); fill.fore_color.rgb = C_BG
    tb = slide.shapes.add_textbox(Inches(0.20), Inches(0.06), Inches(9.60), Inches(0.52))
    p  = tb.text_frame.paragraphs[0]
    r  = p.add_run(); r.text = title_text
    r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = C_TITLE
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path,
            int(Inches(0.25)), int(Inches(fig_top)),
            int(Inches(9.50)), int(Inches(fig_h)))
    else:
        print(f'  ⚠  Missing: {img_path}')
    return len(prs.slides) - 1


prs = Presentation(SRC)
print(f'Opened v25: {len(prs.slides)} slides')

# Step 1: add the three new slides at the end
idx_s1 = add_figure_slide(
    prs,
    'Slide 1: The Supervisor\'s Variables Do Show a Neural Signal in Some Sessions',
    os.path.join(adir, 'e07_e23_supervisors_signal.png'))
print(f'  Added supervisors_signal at idx {idx_s1}')

idx_s2 = add_figure_slide(
    prs,
    'Slide 2: Our Model Detects It — and Attributes Far More to Our Primary Variable',
    os.path.join(adir, 'e07_e23_gpv_comparison.png'))
print(f'  Added gpv_comparison at idx {idx_s2}')

idx_s3 = add_figure_slide(
    prs,
    'Slide 3: Our Variable Co-varies With Theirs (E07) — It Captures the Same Signal',
    os.path.join(adir, 'e07_e23_covariate_explanation.png'))
print(f'  Added covariate_explanation at idx {idx_s3}')
# → 87 slides

# Step 2: find and remove the old single S62 slide
old_idx = find_idx(prs, 'When Cue/Choice Discriminates')
if old_idx is None:
    old_idx = find_idx(prs, 'Outpredict Cue and Choice Labels')
if old_idx is not None:
    remove_slide(prs, old_idx)
    print(f'  Removed old single-figure validation at idx {old_idx}')
    # adjust new slide indices after removal
    if idx_s1 > old_idx: idx_s1 -= 1
    if idx_s2 > old_idx: idx_s2 -= 1
    if idx_s3 > old_idx: idx_s3 -= 1
# → 86 slides

# Step 3: move the three new slides to immediately after the E07/E23 section header
header_idx = find_idx(prs, 'Case Studies')
if header_idx is None:
    header_idx = find_idx(prs, 'E07')
target = header_idx + 1

# Move S1 first (then S2 and S3 follow)
move_slide(prs, idx_s1, target)
print(f'  Moved supervisors_signal to idx {target}')
# S2 and S3 are now at idx_s2+1 and idx_s3+1 if they were after target
idx_s2 = find_idx(prs, 'Our Model Detects It')
idx_s3 = find_idx(prs, 'Co-varies With Theirs')

move_slide(prs, idx_s2, target + 1)
print(f'  Moved gpv_comparison to idx {target+1}')

idx_s3 = find_idx(prs, 'Co-varies With Theirs')
move_slide(prs, idx_s3, target + 2)
print(f'  Moved covariate_explanation to idx {target+2}')

# ── Verify ────────────────────────────────────────────────────────────────────
print(f'\nFinal slide count: {len(prs.slides)}')
print('\nE07/E23 section:')
header_idx = find_idx(prs, 'Case Studies')
for i in range(header_idx, min(header_idx + 6, len(prs.slides))):
    sl = prs.slides[i]
    texts = [sh.text_frame.text.strip()[:70] for sh in sl.shapes if sh.has_text_frame]
    title = next((t for t in texts if t), '(empty)')
    print(f'  S{i+1:02d}: {title}')

for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'\nSaved → {out}')
