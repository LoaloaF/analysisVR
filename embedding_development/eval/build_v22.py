#!/usr/bin/env python3
"""
build_v22.py  —  Replace "collinear/collinearity" with "co-varying/co-variation"
                 throughout v21 (85 slides → 85 slides, no count change).

Changes:
  Slide text patches (7 text boxes across 6 slides):
    S40  title:       "...Collinearity-Inflated"
                   →  "...Co-variation-Inflated"
    S57  body:        "Position coding via collinear features"
                   →  "Position coding via co-varying features"
    S59  title:       "Position Is Captured via Collinear Features:
                        Task Events Are Position-Triggered"
                   →  "Position Is Captured via Co-varying Task Events:
                        Position-Triggered Cues Dominate"
    S59  stray box:   "position_collinearity.png"  → deleted
    S66  body:        "Removes collinearity inflation; ..."
                   →  "Removes co-variation inflation; ..."
    S71  definition:  "...reducing collinearity inflation."
                   →  "...reducing co-variation inflation."
    S71  computation: "...= collinearity inflated"
                   →  "...= co-variation inflated"
    S84  body:        "position and task events are collinear
                        (position-triggered events)."
                   →  "position and task events co-vary
                        (task events are position-triggered)."

  Figure replacements (2 slides):
    S40  global_vs_cond_pv_scatter.png  (footnote updated)
    S63  case_studies_e07_e23.png       (footnote updated)
"""
import os, re
from pptx import Presentation
from pptx.util import Inches

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')

SRC  = os.path.join(root, 'outputs', 'ultimate_presentation_v21.pptx')
OUTS = [
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v22.pptx',
    os.path.join(root, 'outputs', 'ultimate_presentation_v22.pptx'),
]

SLIDE_W = Inches(10.0)
SLIDE_H = Inches(5.625)


# ── Helpers ───────────────────────────────────────────────────────────────────

def patch_text(slide, old, new):
    """Replace old→new in every run of every text frame on the slide."""
    found = False
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)
                    found = True
    return found


def delete_shape_containing(slide, text):
    """Delete the first shape whose full text contains the given string."""
    for sh in list(slide.shapes):
        if sh.has_text_frame and text in sh.text_frame.text:
            sp = sh._element
            sp.getparent().remove(sp)
            return True
    return False


def replace_picture(slide, new_img_path):
    """Replace the first picture shape on a slide with new_img_path,
    preserving position and size."""
    for sh in slide.shapes:
        if sh.shape_type == 13:   # MSO_SHAPE_TYPE.PICTURE = 13
            left, top, width, height = sh.left, sh.top, sh.width, sh.height
            sp = sh._element
            sp.getparent().remove(sp)
            slide.shapes.add_picture(new_img_path, left, top, width, height)
            return True
    return False


# ── Load ──────────────────────────────────────────────────────────────────────
prs = Presentation(SRC)
print(f'Opened v21: {len(prs.slides)} slides')

changes = []

# ── S40 (idx 39) — title + figure replacement ─────────────────────────────────
sl40 = prs.slides[39]
if patch_text(sl40,
              'Points Below Diagonal Are Collinearity-Inflated',
              'Points Below Diagonal Are Co-variation-Inflated'):
    changes.append('S40: patched title')

img40 = os.path.join(mdir, 'global_vs_cond_pv_scatter.png')
if replace_picture(sl40, img40):
    changes.append('S40: replaced global_vs_cond_pv_scatter.png')

# ── S57 (idx 56) — section header ─────────────────────────────────────────────
sl57 = prs.slides[56]
if patch_text(sl57,
              'Position coding via collinear features',
              'Position coding via co-varying features'):
    changes.append('S57: patched section header')

# ── S59 (idx 58) — title + delete stray filename text box ─────────────────────
# Title is split across runs:
#   run[0] = 'Position Is Captured via Collinear Features'
#   run[1] = ': '
#   run[2] = 'Task Events Are Position-Triggered'
sl59 = prs.slides[58]
patched = 0
patched += patch_text(sl59,
                      'Position Is Captured via Collinear Features',
                      'Position Is Captured via Co-varying Task Events')
patched += patch_text(sl59,
                      'Task Events Are Position-Triggered',
                      'Position-Triggered Cues Dominate')
if patched:
    changes.append('S59: patched title (2 runs)')

if delete_shape_containing(sl59, 'position_collinearity.png'):
    changes.append('S59: deleted stray filename text box')

# ── S66 (idx 65) — hidden methods card ───────────────────────────────────────
sl66 = prs.slides[65]
if patch_text(sl66,
              'Removes collinearity inflation;',
              'Removes co-variation inflation;'):
    changes.append('S66: patched CPV methods card')

# ── S71 (idx 70) — appendix A03 ──────────────────────────────────────────────
sl71 = prs.slides[70]
if patch_text(sl71,
              'reducing collinearity inflation.',
              'reducing co-variation inflation.'):
    changes.append('S71: patched A03 definition')

if patch_text(sl71,
              'collinearity inflated',
              'co-variation inflated'):
    changes.append('S71: patched A03 computation')

# ── S84 (idx 83) — appendix A16 ──────────────────────────────────────────────
sl84 = prs.slides[83]
if patch_text(sl84,
              'position and task events are collinear (position-triggered events).',
              'position and task events co-vary (task events are position-triggered).'):
    changes.append('S84: patched A16 position ablation')

# ── S63 (idx 62) — case studies figure replacement ────────────────────────────
sl63 = prs.slides[62]
img63 = os.path.join(mdir, 'case_studies_e07_e23.png')
if replace_picture(sl63, img63):
    changes.append('S63: replaced case_studies_e07_e23.png')

# ── Report ────────────────────────────────────────────────────────────────────
print(f'\nChanges applied ({len(changes)}):')
for c in changes:
    print(f'  {c}')

# Verify no remaining instances
remaining = []
for i, sl in enumerate(prs.slides):
    for sh in sl.shapes:
        if sh.has_text_frame:
            t = sh.text_frame.text
            if re.search(r'collinear', t, re.IGNORECASE):
                remaining.append(f'  S{i+1} (idx {i}): {t[:80]!r}')
if remaining:
    print(f'\n⚠  Remaining "collinear" occurrences:')
    for r in remaining: print(r)
else:
    print('\n✓  No remaining "collinear" occurrences in slide text')

print(f'\nFinal slide count: {len(prs.slides)}')
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'Saved  → {out}')
