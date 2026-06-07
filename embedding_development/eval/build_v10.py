#!/usr/bin/env python3
"""
build_v10.py  —  Apply all v9-review feedback → v10  (64 slides).

Confirmed v9 slide map (from PPTX inspection):
  S34 (idx 33)  GPV heatmap
  S38 (idx 37)  IG heatmap
  S41 (idx 40)  joint_effect_scatter (η² vs MLP R²)
  S42 (idx 41)  'ML Attribution Tracks Naive Data-Effect Size'  ← flip
  S44 (idx 43)  'MLP Cross-Seed Consistency (R² bar)' ← MLP-only, remove
  S50 (idx 49)  'Head Angle Tuning Curves' (4×3 grid, too small)
  S52 (idx 51)  head_angle_scatter_2x2
  S53 (idx 52)  head_angle_stability.png
  S54 (idx 53)  head_angle_tuning_6panel.png (now 2×2)
  S62–S64       hidden reference slides

Changes applied:
  1. Figure replacements (no index shifts):
       S34  GPV heatmap  → add "MLP" architecture label
       S38  IG heatmap   → add "MLP" label, fix cbar pad
       S50  tuning curves → replace with landscape hypothesis_summary.png
       S52  scatter      → y-axis label left-column only (overlap fix)
       S53  stability    → regenerated version
       S54  6-panel      → redesigned 2×2 with larger fonts
  2. Title flip:   S42 → 'Attribution ≠ Naive Effect Size'
  3. Slide move:   S41 → idx 27 (last slide of Encoding Models section)
  4. Slide remove: S44 (MLP-only consistency)
  5. New slide:    conclusion slide before hidden slides

Net: 64 – 1 (remove) + 0 (move doesn't change count) + 1 (conclusion) = 64 slides
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE

SRC  = '/mnt/c/Users/amits/Desktop/ultimate_presentation_v9.pptx'
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v10.pptx',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'outputs', 'ultimate_presentation_v10.pptx'),
]

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)
TITLE_H = Inches(0.55)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x23, 0x3A)
ACCENT  = RGBColor(0x2C, 0x5F, 0x9E)
DPI     = 200


# ── helpers ───────────────────────────────────────────────────────────────────
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


def move_slide(prs, from_idx, to_idx):
    xml_slides = prs.slides._sldIdLst
    children   = list(xml_slides)
    el = children.pop(from_idx)
    children.insert(to_idx, el)
    for c in list(xml_slides):
        xml_slides.remove(c)
    for c in children:
        xml_slides.append(c)


def remove_slide(prs, idx):
    """
    Fully remove a slide: drop it from _sldIdLst (ordering) AND pop its
    relationship from presentation.xml.rels (so the backing slide XML part
    is not written to the saved zip).

    IMPORTANT: always call add_slide BEFORE remove_slide.  python-pptx names
    new slide parts slide{N+1}.xml where N = len(_sldIdLst).  If we remove
    first, N drops by 1 and the new slide gets the same number as an existing
    part → duplicate name in the zip.  Adding first (N still includes the to-
    be-removed slide) gives the new slide a truly new number.
    """
    slide      = prs.slides[idx]
    slide_part = slide.part
    prs_part   = prs.slides.part   # the PresentationPart

    # 1. Remove the relationship so the slide part won't be written to the zip
    rId = None
    for rel_id, rel in prs_part.rels.items():
        try:
            if rel.target_part is slide_part:
                rId = rel_id
                break
        except Exception:
            pass
    if rId is not None:
        prs_part.rels.pop(rId)

    # 2. Remove from _sldIdLst (slide ordering)
    xml_slides = prs.slides._sldIdLst
    el = list(xml_slides)[idx]
    xml_slides.remove(el)


def flip_title_text(slide, new_text):
    """Replace the first non-empty run in any textbox with new_text."""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                if run.text.strip():
                    run.text = new_text
                    return True
    return False


def add_conclusion_slide(prs):
    layout = prs.slide_layouts[6]   # blank
    slide  = prs.slides.add_slide(layout)
    fill   = slide.background.fill
    fill.solid(); fill.fore_color.rgb = WHITE

    # Title bar
    tb_t = slide.shapes.add_textbox(Inches(0.5), Inches(0.12),
                                     SLIDE_W - Inches(1.0), Inches(0.52))
    p = tb_t.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = 'Summary & Open Questions'
    r.font.size = Pt(24); r.font.bold = True; r.font.color.rgb = DARK

    # Thin rule
    rule_w = Inches(8.5)
    rule_t = Inches(0.70)
    conn = slide.shapes.add_shape(1, (SLIDE_W - rule_w)//2, rule_t,
                                   rule_w, Emu(4 * 12700))
    conn.fill.solid(); conn.fill.fore_color.rgb = ACCENT
    conn.line.fill.background()

    FINDINGS = [
        ('Encoding Models',
         'Nonlinear MLP outperforms linear regression; joint multi-feature '
         'effects—invisible to any single η² test—explain the advantage.'),
        ('Feature Attribution',
         'Head angle and angular velocity dominate across ensembles. '
         'Attribution is reproducible: cross-seed ρ ≈ 0.95; GPV ≈ IG.'),
        ('Embedding Consistency',
         'Attribution profiles are stable across seeds and architectures '
         '(MLP vs TempConv), reflecting neural structure not model artefacts.'),
        ('Case Study — Head Angle',
         'Apparent IG over-attribution is explained by non-monotonic tuning '
         'that Spearman ρ misses; η² aligns better with IG than ρ does.'),
        ('Case Study — Position',
         'Position-sensitive ensembles are explained by correlated task events '
         '(upcoming choice, reward window) consistent with prospective coding.'),
    ]
    QUESTIONS = [
        'What drives session-to-session variance in R² and ensemble yield?',
        'Is head-angle tuning vestibular (self-motion) or visual (optic flow)?',
        'Do non-monotonically-tuned ensembles share anatomical location?',
        'Can causal claims be tested via optogenetic or pharmacological intervention?',
        'Does joint-effect structure generalize across task variants or animals?',
    ]

    # Left column: section summaries
    tb_l = slide.shapes.add_textbox(Inches(0.3), Inches(0.83),
                                     Inches(5.7), Inches(4.65))
    tf_l = tb_l.text_frame; tf_l.word_wrap = True
    for i, (sec, body) in enumerate(FINDINGS):
        p_s = tf_l.paragraphs[0] if i == 0 else tf_l.add_paragraph()
        p_s.space_before = Pt(5 if i > 0 else 0)
        r_s = p_s.add_run()
        r_s.text = sec
        r_s.font.size = Pt(10); r_s.font.bold = True; r_s.font.color.rgb = ACCENT

        p_b = tf_l.add_paragraph()
        p_b.space_before = Pt(1)
        r_b = p_b.add_run()
        r_b.text = body
        r_b.font.size = Pt(9); r_b.font.color.rgb = DARK

    # Right column: open questions
    tb_r = slide.shapes.add_textbox(Inches(6.3), Inches(0.83),
                                     Inches(3.5), Inches(4.65))
    tf_r = tb_r.text_frame; tf_r.word_wrap = True
    p_h = tf_r.paragraphs[0]
    r_h = p_h.add_run()
    r_h.text = 'Open Questions'
    r_h.font.size = Pt(11); r_h.font.bold = True; r_h.font.color.rgb = DARK
    for q in QUESTIONS:
        p_q = tf_r.add_paragraph(); p_q.space_before = Pt(6)
        r_q = p_q.add_run()
        r_q.text = '• ' + q
        r_q.font.size = Pt(9); r_q.font.color.rgb = DARK

    return len(prs.slides) - 1


# ── OPEN ──────────────────────────────────────────────────────────────────────
prs = Presentation(SRC)
n_orig = len(prs.slides)
print(f'Opened v9: {n_orig} slides')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Figure replacements (all before any index-shifting operations)
# ══════════════════════════════════════════════════════════════════════════════

abl_dir = os.path.join(root, 'outputs', 'ablation_vs_attribution')

REPLACEMENTS = {
    33: (os.path.join(mdir, 'gpv_group_ensemble_heatmap.png'),
         'GPV heatmap (MLP arch label)'),
    37: (os.path.join(mdir, 'ig_per_ensemble_heatmap.png'),
         'IG heatmap (MLP arch label + fixed cbar pad)'),
    49: ('/mnt/c/Users/amits/Desktop/head_angle_tuning/hypothesis_summary.png',
         'S50 tuning curves → landscape hypothesis_summary (non-monotonic tuning story)'),
    51: (os.path.join(mdir, 'head_angle_scatter_2x2.png'),
         'scatter (y-axis label left-column only)'),
    52: (os.path.join(mdir, 'head_angle_stability.png'),
         'stability (regenerated)'),
    53: (os.path.join(mdir, 'head_angle_tuning_6panel.png'),
         '6-panel → redesigned 2×2 with larger fonts'),
    # S60 in v9 (idx 59) = case study E07/E23 — replace with readable version
    # (old figure was 12×10" and unreadable at slide scale; new is FIG.FULL)
    59: (os.path.join(mdir, 'combined_attribution_vs_ablation_r2.png'),
         'E07/E23 case study (FIG.FULL, readable fonts)'),
}

for idx, (img_path, desc) in REPLACEMENTS.items():
    if not os.path.exists(img_path):
        print(f'  SKIP S{idx+1:02d}: {img_path} not found')
        continue
    replace_figure(prs.slides[idx], img_path)
    print(f'  Replaced S{idx+1:02d}: {desc}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Flip S42 title (in-place, no index shift)
# ══════════════════════════════════════════════════════════════════════════════

COHEN_IDX = 41   # 'ML Attribution Tracks Naive Data-Effect Size'
new_title = 'Attribution ≠ Naive Effect Size — MLP Captures Novel Multi-Feature Variance'
flipped = flip_title_text(prs.slides[COHEN_IDX], new_title)
print(f'  S42 title flip: {"done" if flipped else "no text found — check manually"}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Move joint_effect_scatter (S41 / idx 40) to idx 27
#           This makes it the last slide of the Encoding Models section,
#           right before the Example Predictions divider.
# ══════════════════════════════════════════════════════════════════════════════

JOINT_FROM = 40   # S41 = joint_effect_scatter
JOINT_TO   = 27   # becomes new S28; former S28 (Example Predictions) shifts to S29

move_slide(prs, JOINT_FROM, JOINT_TO)
print(f'  Moved joint_effect_scatter: S{JOINT_FROM+1} → S{JOINT_TO+1} '
      f'(end of Encoding Models section)')

# After this move:
#   Slides idx 27-39 shifted to 28-40 (they made room for the moved slide)
#   Slide at idx 40 went to idx 27
#   Slides at idx 41+ unchanged
# Key indices after move:
#   S42 title-flipped slide (was idx 41) → still idx 41 ✓
#   S44 MLP-only (was idx 43)           → still idx 43 ✓

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Add conclusion slide (BEFORE removing S44)
#
# IMPORTANT ORDER: conclusion must be added while len(_sldIdLst) == 64 so
# python-pptx names the new part slide65.xml.  If we removed S44 first,
# len == 63 and the new part would be named slide64.xml — which already exists
# in the package — producing a 'Duplicate name' collision in the saved zip.
# ══════════════════════════════════════════════════════════════════════════════

new_conc_idx = add_conclusion_slide(prs)   # slide65.xml; appended at end (idx 64)
print(f'  Created conclusion slide at idx {new_conc_idx} (slide65.xml, no collision)')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Remove MLP-only cross-seed consistency slide (S44 / idx 43)
# ══════════════════════════════════════════════════════════════════════════════
# After adding conclusion in step 4: 65 slides total.
# S44 (MLP-only) is still at idx 43 (< 64, so unaffected by conclusion append).
# S45 = 'Embedding Consistency Across Models'  ← keep
# S46 = 'Cross-Ensemble Attribution Consistency' ← keep
# Hidden slides still at idx 61, 62, 63.
# Conclusion at idx 64.

MLPONLY_IDX = 43
remove_slide(prs, MLPONLY_IDX)
print(f'  Removed S{MLPONLY_IDX+1:02d} (MLP-only cross-seed consistency slide)')

# After removal (simple _sldIdLst deletion, no file removed):
# Total: 64 slides in _sldIdLst.
# Slides that were at idx 44+ shift to 43+:
#   Hidden slides: idx 61,62,63 → 60,61,62
#   Conclusion: idx 64 → 63

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Move conclusion to idx 60 (before hidden slides)
# ══════════════════════════════════════════════════════════════════════════════
# After step 5: conclusion is at idx 63, hidden slides at idx 60, 61, 62.
# Move conclusion from idx 63 → idx 60 so it sits just before the hidden block.

move_slide(prs, 63, 60)
print(f'  Moved conclusion slide to S61 (before hidden reference slides)')

# ── SAVE + STRIP ORPHANED PARTS ───────────────────────────────────────────────
# python-pptx writes ALL parts it loaded (including unreferenced ones) to the
# zip, which can produce duplicate entries or orphaned slide XML files that make
# PowerPoint show a "hard to repair" dialog.  After saving, rewrite the zip
# keeping only parts that are reachable from presentation.xml.rels.

import zipfile, shutil, xml.etree.ElementTree as ET

print(f'\nFinal slide count: {len(prs.slides)}  (was {n_orig})')

def _clean_orphaned_parts(pptx_path):
    """Rewrite the PPTX zip removing unreferenced slide parts."""
    REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
    SLIDE_RELTYPE = ('http://schemas.openxmlformats.org/officeDocument/2006/'
                     'relationships/slide')

    with zipfile.ZipFile(pptx_path, 'r') as z:
        rels_data = z.read('ppt/_rels/presentation.xml.rels')
        root = ET.fromstring(rels_data)
        # Collect the set of slide filenames that are actually referenced
        referenced = set()
        for rel in root.findall(f'{{{REL_NS}}}Relationship'):
            if rel.get('Type') == SLIDE_RELTYPE:
                target = rel.get('Target', '')   # e.g. 'slides/slide3.xml'
                referenced.add('ppt/' + target.lstrip('/'))

        # Find all slide XML entries in the zip (not layouts/masters)
        slide_entries = [n for n in z.namelist()
                         if 'ppt/slides/slide' in n
                         and 'slideLayout' not in n
                         and 'slideMaster' not in n]

        # Unreferenced entries (orphans) and their .rels companions
        orphans = set()
        for name in slide_entries:
            bare = name.replace('.rels', '').replace('/slides/_rels/', '/slides/')
            if bare not in referenced:
                orphans.add(name)
                orphans.add(name.replace('/slides/', '/slides/_rels/').replace('.xml', '.xml.rels'))

        # Also deduplicate: if any name appears more than once, keep only first
        seen  = set()
        skip  = set()
        for info in z.infolist():
            if info.filename in seen:
                skip.add(info.filename)
            seen.add(info.filename)
        to_remove = orphans | skip

        if not to_remove:
            print('  No orphaned or duplicate parts — zip is clean.')
            return

        print(f'  Removing {len(to_remove)} orphaned/duplicate parts: '
              f'{sorted(to_remove)}')

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

for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
    _clean_orphaned_parts(out)
    print(f'Cleaned → {out}')

print('Done.')
