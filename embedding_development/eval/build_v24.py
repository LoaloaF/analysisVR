#!/usr/bin/env python3
"""
build_v24.py  — 8 structural fixes from user feedback, v23 (85) → v24 (85 slides).

1. S38: replace gpv_task_ensembles.png (highlighting + red boxes removed)
2. S47: replace attribution_consistency_merged.png (n= clutter removed, correct
        right-panel title 'Cross-seed GPV agreement', no stretching)
3. S48: remove stray thin horizontal bar shape
4. S49: replace cross_ensemble_prediction.png (dual R²≥0.1 validity filter;
        self-target values now all ≥0.1; fix subtitle overlap)
5. S50: remove (content already in S47 right panel)
6. S51: rename 'Head Angle Tuning Curves' → 'Head Angle Analysis' section
        header, styled like S36 (bold large title + subtitle + thin rule)
7. Add head angle section divider (like S36) — S51 IS the divider; just fix
   its visual to match S36 exactly
8. S62 (E07/E23): replace case_studies_e07_e23.png with new
   e07_e23_covariate_validation.png; update title + subtitle to tell the
   'our variables beat supervisor labels' story

Net: 85 - 1 (remove S50) + 1 (already balanced by not adding) = 84 slides
Wait: removing S50 = 84. Adding nothing = 84.
Actually net = 85 - 1 = 84 slides.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir = os.path.join(root, 'outputs', 'cebra_comparison')
adir = os.path.join(root, 'outputs', 'ablation_vs_attribution')

SRC  = os.path.join(root, 'outputs', 'ultimate_presentation_v23.pptx')
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v24.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v24.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)
C_BG    = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE = RGBColor(0x1A, 0x23, 0x3A)
C_SUB   = RGBColor(0x33, 0x33, 0x33)
C_RULE  = RGBColor(0xBB, 0xBB, 0xBB)


# ── Helpers ───────────────────────────────────────────────────────────────────
def remove_slide(prs, idx):
    slide      = prs.slides[idx]
    slide_part = slide.part
    prs_part   = prs.slides.part
    rId = None
    for rel_id, rel in prs_part.rels.items():
        try:
            if rel.target_part is slide_part:
                rId = rel_id; break
        except Exception:
            pass
    if rId:
        prs_part.rels.pop(rId)
    xml = prs.slides._sldIdLst
    xml.remove(list(xml)[idx])


def find_idx(prs, fragment):
    for i, sl in enumerate(prs.slides):
        for sh in sl.shapes:
            if sh.has_text_frame and fragment.lower() in sh.text_frame.text.lower():
                return i
    return None


def replace_picture(slide, new_img, left_in, top_in, w_in, h_in):
    for sh in list(slide.shapes):
        if sh.shape_type == 13:
            sh._element.getparent().remove(sh._element)
            break
    slide.shapes.add_picture(new_img,
        int(Inches(left_in)), int(Inches(top_in)),
        int(Inches(w_in)),    int(Inches(h_in)))


def patch_run(slide, old, new):
    found = False
    for sh in slide.shapes:
        if not sh.has_text_frame: continue
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
                    found = True
    return found


# ── Load ──────────────────────────────────────────────────────────────────────
prs = Presentation(SRC)
print(f'Opened v23: {len(prs.slides)} slides')


# ══ 1. S38 — replace GPV heatmap (no highlighting) ═══════════════════════════
idx = find_idx(prs, 'GPV by Feature Group')
if idx is not None:
    # gpv_task_ensembles is 11"×4.5" → insert at 9.40"×3.85" (correct ratio)
    replace_picture(prs.slides[idx],
                    os.path.join(mdir, 'gpv_task_ensembles.png'),
                    0.30, 0.62, 9.40, 3.85)
    print(f'  S{idx+1}: replaced gpv_task_ensembles.png')


# ══ 2. S47 — replace attribution_consistency_merged ═══════════════════════════
idx = find_idx(prs, 'Attribution Profile Consistency')
if idx is not None:
    # figure is 9.50"×4.20"
    replace_picture(prs.slides[idx],
                    os.path.join(cdir, 'attribution_consistency_merged.png'),
                    0.30, 0.62, 9.40, 4.16)
    print(f'  S{idx+1}: replaced attribution_consistency_merged.png')


# ══ 3. S48 — remove stray thin bar shape ══════════════════════════════════════
idx = find_idx(prs, 'Embedding Geometric Consistency')
if idx is not None:
    sl = prs.slides[idx]
    removed = 0
    for sh in list(sl.shapes):
        if sh.left is None: continue
        h = sh.height / 914400
        w = sh.width  / 914400
        if h < 0.10 and w > 5.0 and not sh.has_text_frame:
            sh._element.getparent().remove(sh._element)
            removed += 1
        elif sh.has_text_frame and not sh.text_frame.text.strip() and h < 0.15:
            # empty text box used as a visual rule
            sh._element.getparent().remove(sh._element)
            removed += 1
    print(f'  S{idx+1}: removed {removed} stray bar shape(s)')


# ══ 4. S49 — replace cross_ensemble_prediction + fix subtitle overlap ═════════
idx = find_idx(prs, 'Cross-Ensemble Neural Prediction')
if idx is not None:
    sl = prs.slides[idx]
    # Remove subtitle text box that was overlapping picture
    for sh in list(sl.shapes):
        if sh.has_text_frame and 'Self-target' in sh.text_frame.text:
            sh._element.getparent().remove(sh._element)
            break
    # figure is 9.50"×4.20"
    replace_picture(sl,
                    os.path.join(mdir, 'cross_ensemble_prediction.png'),
                    0.30, 0.62, 9.40, 4.16)
    print(f'  S{idx+1}: replaced cross_ensemble_prediction.png + removed subtitle')


# ══ 5. Remove S50 (redundant with S47 right panel) ════════════════════════════
idx = find_idx(prs, 'Cross-Ensemble Attribution Consistency Within Sessions')
if idx is not None:
    remove_slide(prs, idx)
    print(f'  Removed S{idx+1}: Cross-Ensemble Attribution Consistency')
# → 84 slides


# ══ 6. Fix S51 head angle section header to match S36 style ══════════════════
idx = find_idx(prs, 'Head Angle Tuning Curves')
if idx is not None:
    sl = prs.slides[idx]

    # Clear all existing shapes
    for sh in list(sl.shapes):
        sh._element.getparent().remove(sh._element)

    # Background
    fill = sl.background.fill; fill.solid(); fill.fore_color.rgb = C_BG

    # Thin horizontal rule (like S36: 4.0"×0.06" centred at y=1.76")
    rect = sl.shapes.add_shape(1,
        int(Inches(3.0)), int(Inches(1.76)),
        int(Inches(4.0)), int(Inches(0.06)))
    rect.fill.solid(); rect.fill.fore_color.rgb = C_RULE
    rect.line.fill.background()

    # Title (bold, 38pt, dark navy — like S36)
    tb = sl.shapes.add_textbox(Inches(0.5), Inches(2.06), Inches(9.0), Inches(0.90))
    p  = tb.text_frame.paragraphs[0]
    r  = p.add_run(); r.text = 'Head Angle Analysis'
    r.font.size = Pt(38); r.font.bold = True; r.font.color.rgb = C_TITLE

    # Subtitle (18pt, grey — like S36)
    tb2 = sl.shapes.add_textbox(Inches(0.5), Inches(2.93), Inches(9.0), Inches(0.55))
    p2  = tb2.text_frame.paragraphs[0]
    r2  = p2.add_run()
    r2.text = 'Dominant feature by GPV and IG  ·  Stable tuning  ·  Non-monotonic shapes'
    r2.font.size = Pt(18); r2.font.color.rgb = C_SUB

    print(f'  S{idx+1}: rebuilt as section divider (matching S36 style)')


# ══ 7. S62 (E07/E23) — new figure + updated title ════════════════════════════
idx = find_idx(prs, 'E07: Speed-Dominated')
if idx is not None:
    sl = prs.slides[idx]

    # Update section header title and subtitle
    patch_run(sl,
              'E07: Speed-Dominated  ·  E23: Head Angle-Dominated',
              'E07 & E23: Attribution-Validated Case Studies')
    patch_run(sl,
              'Attribution profiles validated against independent neural measures',
              'Our variables outpredict supervisor labels in most sessions')
    print(f'  S{idx+1}: updated E07/E23 section header text')

idx_fig = find_idx(prs, 'Case Studies: E07 (Speed-Dominated)')
if idx_fig is not None:
    sl = prs.slides[idx_fig]
    # Update title
    patch_run(sl,
              'Case Studies: E07 (Speed-Dominated) vs E23 (Head-Angle Dominated)',
              'Speed (E07) and Head Angle (E23) Outpredict Cue and Choice Labels')
    # Replace figure
    replace_picture(sl,
                    os.path.join(adir, 'e07_e23_covariate_validation.png'),
                    0.25, 0.62, 9.50, 4.50)
    print(f'  S{idx_fig+1}: replaced with e07_e23_covariate_validation.png')


# ══ Report ════════════════════════════════════════════════════════════════════
print(f'\nFinal slide count: {len(prs.slides)}')

checks = [
    ('Head Angle Analysis',                      'S51 section divider'),
    ('Attribution Profile Consistency',           'S47 attribution consistency'),
    ('Embedding Geometric Consistency',           'S48 geometric consistency'),
    ('Cross-Ensemble Neural Prediction',          'S49 cross-ensemble prediction'),
    ('Cross-Ensemble Attribution Consistency',    'S50 should be GONE'),
    ('Speed.*Outpredict',                         'S62 new figure'),
]
print('\nKey checks:')
for frag, label in checks:
    idx = find_idx(prs, frag.replace('.*', ' '))
    status = f'S{idx+1}' if idx is not None else '✗ GONE'
    print(f'  {label:<40} → {status}')

for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'\nSaved → {out}')
