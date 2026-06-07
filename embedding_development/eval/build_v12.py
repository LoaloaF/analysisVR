#!/usr/bin/env python3
"""
build_v12.py  —  Apply 64d-model evaluation results to v11 (66 slides).

Changes from v11:
  1. S25 (idx 24): Replace R² per-ensemble bar with 64d version
  2. S26 (idx 25): Replace grand-mean R² bar with 64d version
  3. S47 (idx 46): Replace embedding consistency heatmaps with 64d matrices
  4. NEW S49     : Insert representation_consistency.png (decodability profiles)
                   after old S48 (cross-model attribution consistency)

Net: 66 + 1 = 67 slides
"""
import os, shutil, zipfile, xml.etree.ElementTree as ET
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')
mdir  = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir  = os.path.join(root, 'outputs', 'cebra_comparison')

SRC  = '/mnt/c/Users/amits/Desktop/ultimate_presentation_v11.pptx'
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v12.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v12.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)
DPI     = 200


def _img_size(img_path):
    img = Image.open(img_path)
    return img.width / DPI, img.height / DPI


def _fit_in_content_area(fw, fh,
                          max_w=Inches(9.5), max_h=Inches(4.8),
                          top=Inches(0.75)):
    scale = min(max_w / Inches(fw), max_h / Inches(fh))
    w = Inches(fw) * scale
    h = Inches(fh) * scale
    l = (SLIDE_W - w) // 2
    t = top + (max_h - h) // 2
    return l, t, w, h


def replace_figure(slide, img_path):
    """Replace the largest picture shape on a slide with img_path."""
    pics = [s for s in slide.shapes if s.shape_type == 13]
    if not pics:
        return
    target = max(pics, key=lambda s: s.width * s.height)
    fw, fh = _img_size(img_path)
    l, t, w, h = _fit_in_content_area(fw, fh)
    target.left, target.top, target.width, target.height = int(l), int(t), int(w), int(h)
    # Swap the image blob
    from pptx.util import Emu as _Emu
    pic_part = target.part
    rId = list(pic_part.rels.keys())[0]
    image_part = pic_part.rels[rId].target_part
    with open(img_path, 'rb') as f:
        image_part._blob = f.read()


def add_figure_slide(prs, img_path, title_text, subtitle_text=None):
    """Add a blank slide with a title bar and centred figure."""
    layout = prs.slide_layouts[6]
    slide  = prs.slides.add_slide(layout)
    fill   = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = __import__('pptx').dml.color.RGBColor(0xFF, 0xFF, 0xFF)

    DARK   = __import__('pptx').dml.color.RGBColor(0x1A, 0x23, 0x3A)
    GREY   = __import__('pptx').dml.color.RGBColor(0x33, 0x33, 0x33)

    # Title
    tb = slide.shapes.add_textbox(Inches(0.3), Inches(0.08),
                                   SLIDE_W - Inches(0.6), Inches(0.52))
    p  = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r  = p.add_run()
    r.text = title_text
    r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = DARK

    if subtitle_text:
        tb2 = slide.shapes.add_textbox(Inches(0.3), Inches(0.60),
                                        SLIDE_W - Inches(0.6), Inches(0.30))
        p2  = tb2.text_frame.paragraphs[0]
        r2  = p2.add_run()
        r2.text = subtitle_text
        r2.font.size = Pt(10); r2.font.color.rgb = GREY

    # Rule
    rule_w = Inches(9.4)
    conn = slide.shapes.add_shape(
        1, (SLIDE_W - rule_w) // 2, Inches(0.70), rule_w, Emu(4 * 12700))
    conn.fill.solid(); conn.fill.fore_color.rgb = GREY
    conn.line.fill.background()

    # Figure
    fw, fh = _img_size(img_path)
    l, t, w, h = _fit_in_content_area(fw, fh, top=Inches(0.80))
    slide.shapes.add_picture(img_path, l, t, w, h)

    return len(prs.slides) - 1


def move_slide(prs, from_idx, to_idx):
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el         = children.pop(from_idx)
    children.insert(to_idx, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)


def _clean_orphaned_parts(pptx_path):
    REL_NS     = 'http://schemas.openxmlformats.org/package/2006/relationships'
    SLIDE_TYPE = ('http://schemas.openxmlformats.org/officeDocument/2006/'
                  'relationships/slide')
    with zipfile.ZipFile(pptx_path, 'r') as z:
        rels_data = z.read('ppt/_rels/presentation.xml.rels')
        root_el   = ET.fromstring(rels_data)
        referenced = {
            'ppt/' + rel.get('Target', '').lstrip('/')
            for rel in root_el.findall(f'{{{REL_NS}}}Relationship')
            if rel.get('Type') == SLIDE_TYPE
        }
        slide_entries = [n for n in z.namelist()
                         if 'ppt/slides/slide' in n
                         and 'slideLayout' not in n
                         and 'slideMaster' not in n]
        orphans = set()
        for name in slide_entries:
            bare = name.replace('/slides/_rels/', '/slides/').replace('.rels', '')
            if bare not in referenced:
                orphans.add(name)
                orphans.add(name.replace('/slides/', '/slides/_rels/').replace('.xml', '.xml.rels'))
        seen = set(); skip = set()
        for info in z.infolist():
            if info.filename in seen: skip.add(info.filename)
            seen.add(info.filename)
        to_remove = orphans | skip
        if not to_remove:
            return
        print(f'  Cleaning {len(to_remove)} orphaned parts')
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


# ── Open ──────────────────────────────────────────────────────────────────────
prs = Presentation(SRC)
n_orig = len(prs.slides)
print(f'Opened v11: {n_orig} slides')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Add new representation consistency slide FIRST (before any moves)
#           Adds as idx 66, becomes slide67.xml (above existing max=66.xml)
# ══════════════════════════════════════════════════════════════════════════════
repr_img = os.path.join(mdir, 'representation_consistency.png')
if os.path.exists(repr_img):
    new_idx = add_figure_slide(
        prs, repr_img,
        'Representation Consistency: Decodability Profile Agreement',
        'Spearman ρ between behavioral decodability profiles '
        '(5-fold CV ridge probe, 64-dim embedding → 11 feature groups)'
    )
    print(f'  Added representation_consistency slide at idx {new_idx}')
else:
    print(f'  SKIP: {repr_img} not found')
    new_idx = None

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Replace R² comparison figures (in-place, no index shift)
# ══════════════════════════════════════════════════════════════════════════════
replacements = {
    24: (os.path.join(cdir, 'r2_bar_tempconv_comparison.png'),
         'S25 TempConv R² bars (64d)'),
    25: (os.path.join(cdir, 'r2_comparison_grand_mean.png'),
         'S26 grand mean R² (64d)'),
    46: (os.path.join(cdir, 'embedding_consistency_comparison.png'),
         'S47 embedding consistency heatmaps (64d)'),
}
for idx, (img_path, desc) in replacements.items():
    if not os.path.exists(img_path):
        print(f'  SKIP {desc}: {img_path} not found')
        continue
    replace_figure(prs.slides[idx], img_path)
    print(f'  Replaced {desc}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Move new representation slide to just after S48 (idx 47→48)
#           After adding in step 1, it's at idx 66.
#           S48 = "Cross-Model Attribution Consistency" at idx 47.
#           Insert representation as new S49 → move to idx 48.
# ══════════════════════════════════════════════════════════════════════════════
if new_idx is not None:
    target_idx = 48   # after S48 cross-model attribution
    move_slide(prs, new_idx, target_idx)
    print(f'  Moved representation_consistency to S{target_idx+1} (after cross-model attribution)')

# ── Save ──────────────────────────────────────────────────────────────────────
print(f'\nFinal slide count: {len(prs.slides)}  (was {n_orig})')
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
    _clean_orphaned_parts(out)
    print(f'Cleaned → {out}')

print('Done.')
