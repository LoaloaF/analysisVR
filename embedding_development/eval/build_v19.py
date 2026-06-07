#!/usr/bin/env python3
"""
build_v19.py  —  Add reference markers to slides + rebuild bibliography.

Changes from v18:
  1. Resolve [7] VR pipeline as SIPEC [6] (same paper); renumber [8]→[7]
  2. Rebuild bibliography slide (S65) with updated 7-entry list
  3. Add [N] reference markers (bottom-right, superscript style) to:
       [1] CEBRA        → S17, S23, S24, S25
       [2] IG           → S41, S42, S43
       [3] Breiman      → S37, S38
       [4] Strobl       → S39, S40
       [5] ICA          → S08
       [6] SIPEC        → S02, S03, S05, S07

Net: 85 slides (no count change — replacing bibliography + adding text boxes)
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')

SRC  = os.path.join(root, 'outputs', 'ultimate_presentation_v17.pptx')  # rebuild from v17
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v19.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v19.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)

C_BG    = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE = RGBColor(0x1A, 0x23, 0x3A)
C_NUM   = RGBColor(0x2C, 0x6E, 0xB5)
C_BODY  = RGBColor(0x22, 0x22, 0x22)
C_PEND  = RGBColor(0x99, 0x99, 0x99)
C_RULE  = RGBColor(0xCC, 0xCC, 0xCC)
C_TAG   = RGBColor(0x88, 0x88, 0x88)


# ── Updated reference list ─────────────────────────────────────────────────
REFERENCES = [
    (1, "CEBRA",
     "Schneider S, Lee JH, Mathis MW (2023). Learnable latent embeddings for joint behavioural and "
     "neural analysis. Nature 617, 360–368. https://doi.org/10.1038/s41586-023-06031-6",
     False),

    (2, "Integrated Gradients",
     "Sundararajan M, Taly A, Yan Q (2017). Axiomatic attribution for deep networks. "
     "Proc. 34th ICML, PMLR 70, 3319–3328. https://proceedings.mlr.press/v70/sundararajan17a.html",
     False),

    (3, "Permutation Feature Importance",
     "Breiman L (2001). Random forests. Machine Learning 45(1), 5–32. "
     "https://doi.org/10.1023/A:1010933404324",
     False),

    (4, "Conditional Permutation Importance",
     "Strobl C, Boulesteix A-L, Kneib T, Augustin T, Zeileis A (2008). Conditional variable "
     "importance for random forests. BMC Bioinformatics 9, 307. https://doi.org/10.1186/1471-2105-9-307",
     False),

    (5, "ICA",
     "Hyvärinen A, Oja E (2000). Independent component analysis: algorithms and applications. "
     "Neural Networks 13(4–5), 457–472. https://doi.org/10.1016/S0893-6080(00)00026-5",
     False),

    (6, "SIPEC  (behavioural pipeline + VR task)",
     "Marks M et al. (2022). SIPEC: the deep-learning Swiss knife for behavioural data analysis. "
     "Nature Methods 19, 432–443. https://doi.org/10.1038/s41592-022-01397-x",
     False),

    (7, "Neural ensemble extraction",
     "[ fill in — lab or collaborator paper for ICA-based ensemble extraction in this context ]",
     True),
]

# ── Slide → reference(s) mapping (0-indexed) ──────────────────────────────
# Only confirmed refs (no pending [7])
SLIDE_REFS = {
    1:  [6],        # S02 Example trial / task
    2:  [6],        # S03 Paradigm & Setup
    4:  [6],        # S05 Input Data Distribution (behavioral features)
    6:  [6],        # S07 Map from SIPEC (title mentions SIPEC)
    7:  [5],        # S08 Neural ensemble extraction — ICA
    16: [1],        # S17 Nonlinear Encoding Models (TC intro)
    22: [1],        # S23 Temporal Context and Contrastive Learning
    23: [1],        # S24 Contrastive Embeddings
    24: [1],        # S25 TempConv R² per ensemble
    36: [3],        # S37 Permutation Variance method
    37: [3],        # S38 GPV heatmap
    38: [4],        # S39 Conditional Permutation Variance
    39: [4],        # S40 GPV vs CPV scatter
    40: [2],        # S41 Integrated Gradients method
    41: [2],        # S42 IG heatmap
    42: [2],        # S43 Group attribution profile
}


def move_slide(prs, from_idx, to_idx):
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el         = children.pop(from_idx)
    children.insert(to_idx, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)


def add_ref_marker(slide, ref_numbers):
    """Add a small [N,M,...] text box at the bottom-right of a slide."""
    label = ''.join(f'[{n}]' for n in ref_numbers)
    tb = slide.shapes.add_textbox(
        int(SLIDE_W - Inches(0.80)), int(SLIDE_H - Inches(0.28)),
        int(Inches(0.75)), int(Inches(0.24))
    )
    tf = tb.text_frame
    p  = tf.paragraphs[0]
    from pptx.enum.text import PP_ALIGN
    p.alignment = PP_ALIGN.RIGHT
    r  = p.add_run()
    r.text = label
    r.font.size  = Pt(8)
    r.font.bold  = False
    r.font.color.rgb = C_NUM


def build_bibliography_slide(prs):
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)
    fill   = slide.background.fill
    fill.solid(); fill.fore_color.rgb = C_BG

    # Tag
    tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.05), Inches(2.5), Inches(0.28))
    r  = tb.text_frame.paragraphs[0].add_run()
    r.text = ''; r.font.size = Pt(7)

    # Title
    tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.10), Inches(9.4), Inches(0.50))
    p  = tb.text_frame.paragraphs[0]
    r  = p.add_run(); r.text = 'References'
    r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = C_TITLE

    # Rule
    rect = slide.shapes.add_shape(1,
        int(Inches(0.3)), int(Inches(0.63)), int(Inches(9.4)), int(Inches(0.012)))
    rect.fill.solid(); rect.fill.fore_color.rgb = C_RULE
    rect.line.fill.background()

    n     = len(REFERENCES)
    left  = n // 2 + n % 2
    col_w = Inches(4.55)
    top   = Inches(0.70)
    row_h = Inches(0.475)

    for col, refs in enumerate([REFERENCES[:left], REFERENCES[left:]]):
        x = Inches(0.25) + col * Inches(5.0)
        for row, (num, tag, text, pending) in enumerate(refs):
            y = top + row * row_h

            nb = slide.shapes.add_textbox(int(x), int(y), int(Inches(0.28)), int(row_h))
            r  = nb.text_frame.paragraphs[0].add_run()
            r.text = f'[{num}]'; r.font.size = Pt(8.5); r.font.bold = True
            r.font.color.rgb = C_PEND if pending else C_NUM

            cb = slide.shapes.add_textbox(
                int(x + Inches(0.30)), int(y),
                int(col_w - Inches(0.30)), int(row_h - Inches(0.04)))
            tf = cb.text_frame; tf.word_wrap = True

            p1 = tf.paragraphs[0]
            r1 = p1.add_run(); r1.text = tag
            r1.font.size = Pt(8); r1.font.bold = True
            r1.font.color.rgb = C_PEND if pending else C_TITLE

            p2 = tf.add_paragraph()
            r2 = p2.add_run(); r2.text = text
            r2.font.size = Pt(7.2)
            r2.font.color.rgb = C_PEND if pending else C_BODY
            if pending: r2.font.italic = True

    return len(prs.slides) - 1


# ── Build ──────────────────────────────────────────────────────────────────
prs = Presentation(SRC)
print(f'Opened v17: {len(prs.slides)} slides')

# 1. Add reference markers to main slides
for slide_idx, ref_nums in sorted(SLIDE_REFS.items()):
    add_ref_marker(prs.slides[slide_idx], ref_nums)
    print(f'  S{slide_idx+1:02d}: added {["["+str(n)+"]" for n in ref_nums]}')

# 2. Add bibliography slide at end, move to S65 (just before hidden slides)
new_idx = build_bibliography_slide(prs)
move_slide(prs, new_idx, 64)
print(f'  Bibliography rebuilt at S65')

print(f'Final slide count: {len(prs.slides)}')
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
