#!/usr/bin/env python3
"""
build_v9.py  —  Insert large-centered section title slides into v8 → v9.

Section dividers are inserted (in reverse order to keep indices stable)
before each major section in the v8 deck.

Divider style: white background, large bold centered section name,
smaller centered subtitle below, no figures.
"""
import os
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

SRC  = '/mnt/c/Users/amits/Desktop/ultimate_presentation_v8.pptx'
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v9.pptx',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'outputs', 'ultimate_presentation_v9.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x23, 0x3A)
ACCENT  = RGBColor(0x2C, 0x5F, 0x9E)   # muted blue rule line


def add_section_slide(prs, title, subtitle=''):
    """Add a large-centered section divider at the end of the deck."""
    layout = prs.slide_layouts[6]   # blank
    slide  = prs.slides.add_slide(layout)

    # White background
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE

    mid_y = SLIDE_H / 2   # Emu

    # ── Thin horizontal rule above title ─────────────────────────────────────
    rule_w = Inches(4.0)
    rule_h = Emu(4 * 12700)   # ~0.5 pt thick
    rule_l = (SLIDE_W - rule_w) // 2
    rule_t = mid_y - Inches(1.05)
    connector = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE = 1
        rule_l, rule_t, rule_w, rule_h
    )
    connector.fill.solid()
    connector.fill.fore_color.rgb = ACCENT
    connector.line.fill.background()   # no border

    # ── Section title ─────────────────────────────────────────────────────────
    title_h = Inches(0.90)
    title_t = mid_y - title_h // 2 - Inches(0.30)
    if subtitle:
        title_t = mid_y - Inches(0.75)
    tb_title = slide.shapes.add_textbox(
        Inches(0.5), title_t, SLIDE_W - Inches(1.0), title_h
    )
    tf = tb_title.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = title
    r.font.size = Pt(38)
    r.font.bold = True
    r.font.color.rgb = DARK

    # ── Subtitle ─────────────────────────────────────────────────────────────
    if subtitle:
        sub_t = mid_y + Inches(0.12)
        tb_sub = slide.shapes.add_textbox(
            Inches(0.5), sub_t, SLIDE_W - Inches(1.0), Inches(0.55)
        )
        tf2 = tb_sub.text_frame
        tf2.word_wrap = False
        p2 = tf2.paragraphs[0]
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(18)
        r2.font.bold = False
        r2.font.color.rgb = ACCENT

    return len(prs.slides) - 1


def move_slide(prs, from_idx, to_idx):
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el = children.pop(from_idx)
    children.insert(to_idx, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)


# ── Section definitions: (insert BEFORE this v8 0-based idx, title, subtitle)
# Listed from LAST to FIRST so each insertion doesn't shift earlier targets.
SECTIONS = [
    # idx  title                           subtitle
    (51, 'Case Studies',                  'E07 and E23 — attribution in depth'),
    (48, 'Case Study: Position',          'Position coding via collinear features'),
    (43, 'Case Study: Head Angle',        'Angular tuning across sessions'),
    (42, 'Frequency Encoding',            'When temporal structure matters'),
    (39, 'Embedding Consistency',         'Cross-seed · Cross-model · Cross-ensemble'),
    (29, 'Feature Attribution',           'GPV · CPV · Integrated Gradients'),
    (26, 'Example Predictions',           'MLP trajectory reconstructions'),
    (11, 'Encoding Models',               'Linear · MLP · TempConv'),
]

prs = Presentation(SRC)
print(f'Opened v8: {len(prs.slides)} slides')

# Insert from last to first
for target_idx, title, subtitle in SECTIONS:
    new_idx = add_section_slide(prs, title, subtitle)
    move_slide(prs, new_idx, target_idx)
    print(f'  Inserted "{title}" before (new) S{target_idx+1:02d}')

print(f'\nFinal slide count: {len(prs.slides)}')
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved → {out}')
print('Done.')
