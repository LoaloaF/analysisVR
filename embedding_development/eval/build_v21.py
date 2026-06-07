#!/usr/bin/env python3
"""
build_v21.py  —  Insert honest_attribution_example.png into v20.

Changes from v20 (84 slides → 85 slides):
  INSERT S46 after current S45 (idx 44):
    "Attribution Is More Honest Than Correlation: S21 E08 Case Study"
    Figure: honest_attribution_example.png (9.5" × 5.67")
    4-panel: raw |r| bar → position #1 | GPV bar → head_angle #1 |
             head_angle tuning curve (U-shape, explains r≈0) |
             position tuning curve (noisy, no shape)

Narrative role: concrete single-pair justification for why the model is more
honest than univariate correlation. Only 19% of R²≥0.1 pairs have top-2
agreement between raw |r| and GPV. For S21 E08, correlation says
"position-tuned" (r=0.14, ranks #1); GPV + IG both say "head_angle-tuned"
(r=0.03, ranks #1 by both methods) because head_angle has U-shaped tuning
that Spearman r cannot detect.

Placement: after Attribution ≠ Naive (S45), before Embedding Consistency
section header (old S46 → new S47).

Net: 84 + 1 = 85 slides.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')

SRC  = os.path.join(root, 'outputs', 'ultimate_presentation_v20.pptx')
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v21.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v21.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)

IMG     = os.path.join(mdir, 'honest_attribution_example.png')
INSERT_AFTER_IDX = 44   # after S45 (idx 44)

C_BG    = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE = RGBColor(0x1A, 0x23, 0x3A)
C_NOTE  = RGBColor(0x55, 0x55, 0x55)


def move_slide(prs, from_idx, to_idx):
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el         = children.pop(from_idx)
    children.insert(to_idx, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)


def add_honest_attribution_slide(prs):
    layout = prs.slide_layouts[6]   # blank
    slide  = prs.slides.add_slide(layout)

    # White background
    fill = slide.background.fill
    fill.solid(); fill.fore_color.rgb = C_BG

    # Title bar
    title_h = Inches(0.52)
    tb = slide.shapes.add_textbox(
        Inches(0.20), Inches(0.06), Inches(9.60), title_h)
    tf = tb.text_frame
    p  = tf.paragraphs[0]
    r  = p.add_run()
    r.text = ('Attribution Is More Honest Than Correlation: '
              'S21 E08 Case Study')
    r.font.size  = Pt(18)
    r.font.bold  = True
    r.font.color.rgb = C_TITLE

    # Figure — fits the full content area below title
    fig_top  = Inches(0.60)
    fig_h    = SLIDE_H - fig_top - Inches(0.05)
    fig_w    = SLIDE_W - Inches(0.40)
    slide.shapes.add_picture(
        IMG,
        int(Inches(0.20)), int(fig_top),
        int(fig_w), int(fig_h),
    )

    return len(prs.slides) - 1


prs = Presentation(SRC)
print(f'Opened v20: {len(prs.slides)} slides')

# Add slide at end (must add before any removals per python-pptx naming rules)
new_idx = add_honest_attribution_slide(prs)
print(f'  Added honest_attribution slide at idx {new_idx} (S{new_idx+1})')

# Move it to immediately after S45 (idx 44)
target_idx = INSERT_AFTER_IDX + 1   # = 45
move_slide(prs, new_idx, target_idx)
print(f'  Moved to idx {target_idx} (S{target_idx+1}) — after S45 / before old S46')

print(f'Final slide count: {len(prs.slides)}')

# Verify neighbours
for i in range(43, 50):
    sl = prs.slides[i]
    texts = [sh.text_frame.text.strip() for sh in sl.shapes if sh.has_text_frame]
    title = next((t for t in texts if t), '(no title)')
    print(f'  S{i+1:02d} (idx {i}): {title[:75]}')

for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
