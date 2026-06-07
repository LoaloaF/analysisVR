#!/usr/bin/env python3
"""
build_v18.py  —  Add visible bibliography slide to v17 (84 slides).

Inserts one reference slide at S65 (just before the hidden methods slides),
pushing existing S65-S84 to S66-S85.

Net: 84 + 1 = 85 slides
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')

SRC  = os.path.join(root, 'outputs', 'ultimate_presentation_v17.pptx')
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v18.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v18.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)

C_BG    = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE = RGBColor(0x1A, 0x23, 0x3A)
C_NUM   = RGBColor(0x2C, 0x6E, 0xB5)   # reference number blue
C_BODY  = RGBColor(0x22, 0x22, 0x22)
C_PEND  = RGBColor(0x99, 0x99, 0x99)   # pending / fill-in
C_RULE  = RGBColor(0xCC, 0xCC, 0xCC)


# References — (number, tag, citation_text, pending)
# pending=True renders in grey with "[ fill in ]"
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

    (6, "SIPEC",
     "Marks M et al. (2022). SIPEC: the deep-learning Swiss knife for behavioural data analysis. "
     "Nature Methods 19, 432–443. https://doi.org/10.1038/s41592-022-01397-x",
     False),

    (7, "VR task / recording pipeline",
     "[ fill in — lab paper describing the linear-track VR setup and electrophysiology pipeline ]",
     True),

    (8, "Neural ensemble extraction",
     "[ fill in — lab or collaborator paper for ICA-based ensemble extraction in this context ]",
     True),
]


def move_slide(prs, from_idx, to_idx):
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el         = children.pop(from_idx)
    children.insert(to_idx, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)


def build_bibliography_slide(prs):
    layout = prs.slide_layouts[6]   # blank
    slide  = prs.slides.add_slide(layout)

    # Background
    fill = slide.background.fill
    fill.solid(); fill.fore_color.rgb = C_BG

    # Title
    tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.10), Inches(9.4), Inches(0.50))
    p  = tb.text_frame.paragraphs[0]
    r  = p.add_run(); r.text = 'References'
    r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = C_TITLE

    # Thin rule
    rect = slide.shapes.add_shape(1,
        int(Inches(0.3)), int(Inches(0.63)), int(Inches(9.4)), int(Inches(0.012)))
    rect.fill.solid(); rect.fill.fore_color.rgb = C_RULE
    rect.line.fill.background()

    # Two-column layout
    n      = len(REFERENCES)
    left   = n // 2 + n % 2   # first column gets the extra ref if odd
    right  = n - left
    col_w  = Inches(4.55)
    top    = Inches(0.70)
    row_h  = Inches(0.475)

    for col, refs in enumerate([ REFERENCES[:left], REFERENCES[left:] ]):
        x = Inches(0.25) + col * Inches(5.0)
        for row, (num, tag, text, pending) in enumerate(refs):
            y = top + row * row_h

            # Number bubble
            nb = slide.shapes.add_textbox(int(x), int(y), int(Inches(0.28)), int(row_h))
            p  = nb.text_frame.paragraphs[0]
            r  = p.add_run(); r.text = f'[{num}]'
            r.font.size = Pt(8.5); r.font.bold = True
            r.font.color.rgb = C_PEND if pending else C_NUM

            # Citation text
            cb = slide.shapes.add_textbox(
                int(x + Inches(0.30)), int(y),
                int(col_w - Inches(0.30)), int(row_h - Inches(0.04)))
            tf = cb.text_frame; tf.word_wrap = True

            # Tag line
            p1 = tf.paragraphs[0]
            r1 = p1.add_run(); r1.text = tag
            r1.font.size = Pt(8); r1.font.bold = True
            r1.font.color.rgb = C_PEND if pending else C_TITLE

            # Citation line
            p2 = tf.add_paragraph()
            r2 = p2.add_run(); r2.text = text
            r2.font.size = Pt(7.2)
            r2.font.color.rgb = C_PEND if pending else C_BODY
            if pending:
                r2.font.italic = True

    return len(prs.slides) - 1


prs = Presentation(SRC)
print(f'Opened v17: {len(prs.slides)} slides')

# Add bibliography at end, then move to just before first hidden slide (idx 64)
new_idx = build_bibliography_slide(prs)
move_slide(prs, new_idx, 64)   # slot in right before the hidden methods slides
print(f'  Bibliography added at S65 (idx 64)')

print(f'Final slide count: {len(prs.slides)}')
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
