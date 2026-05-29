#!/usr/bin/env python3
"""
build_v11.py  —  Four remaining changes applied to the user's edited v10.

Source: user's manually-edited ultimate_presentation_v10.pptx (65 slides)

Changes:
  1. S47  embedding_consistency_comparison.png
          → per-model ensemble ordering, all 23 ensembles, R²≥0.1 each

  2. New slide after S47
          cross_model_attribution_consistency.png
          (Spearman ρ between GPV profiles across architectures)

  3. Conclusion slide (S62 in current v10)
          Right column: "Open Questions" → "Limitations"

  4. All blue accent colors (ACCENT blue) on section divider slides and the
     conclusion slide replaced with dark grey (#333333)
"""
import os, zipfile, shutil
import xml.etree.ElementTree as ET
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE

SRC  = '/mnt/c/Users/amits/Desktop/ultimate_presentation_v10.pptx'
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v11.pptx',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'outputs', 'ultimate_presentation_v11.pptx'),
]

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir = os.path.join(root, 'outputs', 'cebra_comparison')

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)
TITLE_H = Inches(0.55)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
DARK       = RGBColor(0x1A, 0x23, 0x3A)
DARK_GREY  = RGBColor(0x33, 0x33, 0x33)
ACCENT_OLD = RGBColor(0x2C, 0x5F, 0x9E)   # muted blue to replace
DPI = 200


def _img_size(img_path):
    img = Image.open(img_path)
    return img.width / DPI, img.height / DPI


def _fit_in_content_area(fw, fh):
    ct = TITLE_H + Inches(0.05)
    ch = (SLIDE_H - ct - Inches(0.08)) / 914400
    cw = (SLIDE_W - Inches(0.10)) / 914400
    sc = min(cw / fw, ch / fh)
    pw = Inches(fw * sc)
    ph = Inches(fh * sc)
    left = (SLIDE_W - pw) // 2
    top  = ct + Emu(int((ch - fh * sc) / 2 * 914400))
    return left, top, pw, ph


def replace_figure(slide, new_img_path):
    for shape in list(slide.shapes):
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            shape._element.getparent().remove(shape._element)
    fw, fh = _img_size(new_img_path)
    left, top, pw, ph = _fit_in_content_area(fw, fh)
    slide.shapes.add_picture(new_img_path, left, top, pw, ph)


def add_figure_slide(prs, img_path, title_text, after_idx):
    """Insert a new figure slide after after_idx."""
    layout = prs.slide_layouts[6]   # blank
    slide  = prs.slides.add_slide(layout)

    fill = slide.background.fill
    fill.solid(); fill.fore_color.rgb = WHITE

    tb = slide.shapes.add_textbox(Inches(0.15), Inches(0.05),
                                   SLIDE_W - Inches(0.3), TITLE_H)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title_text
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = DARK

    fw, fh = _img_size(img_path)
    left, top, pw, ph = _fit_in_content_area(fw, fh)
    slide.shapes.add_picture(img_path, left, top, pw, ph)

    # Move from end to after_idx+1
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el = children.pop(len(children) - 1)
    children.insert(after_idx + 1, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)
    return after_idx + 1


def _same_color(rgb1, rgb2):
    """Compare two RGBColor values. RGBColor is an int subclass; use int equality."""
    try:
        return int(rgb1) == int(rgb2)
    except Exception:
        return False


def deblue(prs):
    """
    Replace ACCENT_OLD blue (#2C5F9E) with DARK_GREY (#333333) across the deck.

    Uses lxml directly — python-pptx's color API fails when a shape has both
    a spPr solidFill and a p:style schemeClr (the API resolves to the scheme
    and can't read or write the explicit srgbClr).
    """
    from lxml import etree
    A_NS  = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    OLD_VAL = '2C5F9E'
    NEW_VAL = '333333'
    changed = 0

    for slide in prs.slides:
        # Walk every srgbClr element in the slide XML
        for el in slide._element.iter(f'{{{A_NS}}}srgbClr'):
            if el.get('val', '').upper() == OLD_VAL:
                el.set('val', NEW_VAL)
                changed += 1
    return changed


def _clean_orphaned_parts(pptx_path):
    """Strip any unreferenced slide XML from the zip."""
    REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
    SLIDE_RT = ('http://schemas.openxmlformats.org/officeDocument/2006/'
                'relationships/slide')

    with zipfile.ZipFile(pptx_path, 'r') as z:
        rels_data = z.read('ppt/_rels/presentation.xml.rels')
        root_el = ET.fromstring(rels_data)
        referenced = set()
        for rel in root_el.findall(f'{{{REL_NS}}}Relationship'):
            if rel.get('Type') == SLIDE_RT:
                t = rel.get('Target', '')
                referenced.add('ppt/' + t.lstrip('/'))

        slide_entries = [n for n in z.namelist()
                         if 'ppt/slides/slide' in n
                         and 'slideLayout' not in n
                         and 'slideMaster' not in n]
        orphans = set()
        for name in slide_entries:
            bare = name.replace('.rels', '').replace('/slides/_rels/', '/slides/')
            if bare not in referenced:
                orphans.add(name)
                orphans.add(name.replace('/slides/', '/slides/_rels/').replace('.xml', '.xml.rels'))

        seen = set(); skip = set()
        for info in z.infolist():
            if info.filename in seen:
                skip.add(info.filename)
            seen.add(info.filename)
        to_remove = orphans | skip

        if not to_remove:
            print('  Zip is clean.')
            return

        print(f'  Removing {len(to_remove)} orphaned/duplicate parts.')
        tmp = pptx_path + '.clean.tmp'
        with zipfile.ZipFile(pptx_path, 'r') as zin, \
             zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
            written = set()
            for item in zin.infolist():
                if item.filename in to_remove or item.filename in written:
                    continue
                zout.writestr(item, zin.read(item.filename))
                written.add(item.filename)
        shutil.move(tmp, pptx_path)


# ── OPEN ──────────────────────────────────────────────────────────────────────
prs = Presentation(SRC)
n_orig = len(prs.slides)
print(f'Opened v10: {n_orig} slides')

# Verify key slides before touching anything
for i, slide in enumerate(prs.slides):
    for shape in slide.shapes:
        if shape.has_text_frame:
            t = shape.text_frame.text.strip()
            if 'embedding_consistency_comparison' in t.lower():
                print(f'  S47 figure slide found at idx {i} (S{i+1})')
            if 'Summary & Open Questions' in t:
                print(f'  Conclusion slide found at idx {i} (S{i+1})')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Replace S47 (embedding_consistency_comparison.png) in-place
# ══════════════════════════════════════════════════════════════════════════════

CONS_IDX = 46   # S47 — "Embedding Consistency Across Models"
cons_img = os.path.join(cdir, 'embedding_consistency_comparison.png')

replace_figure(prs.slides[CONS_IDX], cons_img)
print(f'  Replaced S{CONS_IDX+1:02d} with updated embedding_consistency_comparison')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Add cross-model attribution consistency slide after S47
#           (add BEFORE removing/modifying anything else to get correct slide#)
# ══════════════════════════════════════════════════════════════════════════════

cross_img = os.path.join(mdir, 'cross_model_attribution_consistency.png')
new_idx = add_figure_slide(
    prs, cross_img,
    'Cross-Model Attribution Consistency: GPV Profile Agreement (Spearman ρ)',
    after_idx=CONS_IDX,
)
print(f'  Inserted cross-model consistency at S{new_idx+1:02d}')

# After insertion, the conclusion slide shifted by 1
# (was idx 61, now idx 62 in the 66-slide deck)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Conclusion slide: replace "Open Questions" with "Limitations"
# ══════════════════════════════════════════════════════════════════════════════

LIMITATIONS = [
    'Correlative, not causal: cannot distinguish feedforward from feedback '
    'contributions; intervention experiments (optogenetics, pharmacology) required.',
    'Ensemble activations collapse spatial organization; single-unit resolution '
    'is lost and cross-animal generalization is untested.',
    'Behavioral feature set is hand-engineered and incomplete; latent variables '
    '(e.g. attention, arousal) may drive attribution without being represented.',
    'Non-monotonic head-angle tuning interpretation has not been validated '
    'histologically or against single-unit receptive field mapping.',
    'R² evaluated per session; session-to-session variability in data quality '
    'means aggregate statistics may be dominated by a few high-yield sessions.',
]

# Find conclusion slide (S62 before insertion, S63 after; search by title text)
conc_idx = None
for i, slide in enumerate(prs.slides):
    for shape in slide.shapes:
        if shape.has_text_frame and 'Summary & Open Questions' in shape.text_frame.text:
            conc_idx = i
            break
    if conc_idx is not None:
        break

if conc_idx is None:
    print('  WARNING: conclusion slide not found — skipping limitations update')
else:
    slide = prs.slides[conc_idx]
    # Find the right-column textbox (the "Open Questions" one)
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text
        if 'Open Questions' in txt or '•' in txt:
            tf = shape.text_frame
            tf.clear()   # wipe all paragraphs

            # Header
            p0 = tf.paragraphs[0]
            r0 = p0.add_run()
            r0.text = 'Limitations'
            r0.font.size = Pt(11); r0.font.bold = True
            r0.font.color.rgb = DARK

            # Bullets
            for lim in LIMITATIONS:
                p = tf.add_paragraph()
                p.space_before = Pt(5)
                r = p.add_run()
                r.text = '• ' + lim
                r.font.size = Pt(8.5); r.font.color.rgb = DARK

            # Also fix slide title text box
            for s2 in slide.shapes:
                if s2.has_text_frame and 'Summary & Open Questions' in s2.text_frame.text:
                    for para in s2.text_frame.paragraphs:
                        for run in para.runs:
                            if 'Summary & Open Questions' in run.text:
                                run.text = 'Summary & Limitations'
            break

    print(f'  Updated conclusion slide S{conc_idx+1:02d}: Open Questions → Limitations')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Replace all ACCENT blue with dark grey
# ══════════════════════════════════════════════════════════════════════════════

n_changed = deblue(prs)
print(f'  Replaced {n_changed} blue accent elements with dark grey')

# ── SAVE ──────────────────────────────────────────────────────────────────────
print(f'\nFinal slide count: {len(prs.slides)}  (was {n_orig})')
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
    _clean_orphaned_parts(out)
    print(f'Cleaned → {out}')
print('Done.')
