#!/usr/bin/env python3
"""
build_v37.py  —  Add E07 joint-encoding analysis slides after S66.

Base: user's Desktop ultimate_presentation_v36.pptx
v36 (89) → v37 (91 slides, +2):
  S67  "E07 × Cue 2: model captures the spatial pattern"
       Figure: e07_assembly_heatmap.png
  S68  "Signal routes through speed, not cue or position directly"
       Figure: e07_joint_captured.png

Verification:
  - All source figures exist and have correct aspect ratio (9.50" × 4.20")
  - All slide titles fit within one line at Pt(16) (≤80 chars)
  - No duplicate zip entries in output
"""
import os, zipfile
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.parts.presentation import PresentationPart


@property
def _safe_next_slide_partname(self):
    return self.package.next_partname('/ppt/slides/slide%d.xml')

PresentationPart._next_slide_partname = _safe_next_slide_partname

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
adir = os.path.join(root, 'outputs', 'ablation_vs_attribution')

SRC  = '/mnt/c/Users/amits/Desktop/ultimate_presentation_v36.pptx'
OUTS = [
    os.path.join(root, 'outputs', 'ultimate_presentation_v37.pptx'),
    '/mnt/c/Users/amits/Desktop/ultimate_presentation_v37.pptx',
]

C_BG    = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE = RGBColor(0x1A, 0x23, 0x3A)

EXPECTED_W_IN = 9.50
EXPECTED_H_IN = 4.20
TARGET_DPI    = 200


# ── Verification helpers ───────────────────────────────────────────────────────
def verify_figure(path, label):
    """Check file exists and has correct dimensions. Returns True if OK."""
    if not os.path.exists(path):
        print(f'  ✗ MISSING: {label} → {path}')
        return False
    with Image.open(path) as im:
        w_px, h_px = im.size
    w_in = w_px / TARGET_DPI
    h_in = h_px / TARGET_DPI
    ok = abs(w_in - EXPECTED_W_IN) < 0.1 and abs(h_in - EXPECTED_H_IN) < 0.5
    status = '✓' if ok else f'✗ SIZE MISMATCH ({w_in:.2f}"×{h_in:.2f}")'
    print(f'  {status} {label}: {w_px}×{h_px}px = {w_in:.2f}"×{h_in:.2f}"')
    return ok


def verify_title(text, label, max_chars=75):
    ok = len(text) <= max_chars
    status = '✓' if ok else f'✗ TOO LONG ({len(text)} chars)'
    print(f'  {status} title "{label}": {len(text)} chars')
    return ok


def verify_zip(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        dupes = [n for n in set(names) if names.count(n) > 1]
        slides = [n for n in names if 'slides/slide' in n
                  and n.endswith('.xml') and 'rels' not in n]
    ok = len(dupes) == 0
    print(f'  {"✓" if ok else "✗ DUPLICATES: "+str(dupes)} zip: {len(slides)} slide XMLs, {len(dupes)} duplicates')
    return ok


# ── Build helpers ──────────────────────────────────────────────────────────────
def find_idx(prs, fragment):
    for i, sl in enumerate(prs.slides):
        for sh in sl.shapes:
            if sh.has_text_frame and fragment.lower() in sh.text_frame.text.lower():
                return i
    return None


def move_slide(prs, from_idx, to_idx):
    xml = prs.slides._sldIdLst
    els = list(xml); el = els.pop(from_idx); els.insert(to_idx, el)
    for c in list(xml): xml.remove(c)
    for c in els:        xml.append(c)


def add_figure_slide(prs, title_text, img_path, fig_top=0.62, fig_h=4.20):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill  = slide.background.fill; fill.solid(); fill.fore_color.rgb = C_BG
    tb = slide.shapes.add_textbox(
        Inches(0.20), Inches(0.06), Inches(9.60), Inches(0.52))
    p = tb.text_frame.paragraphs[0]; r = p.add_run()
    r.text = title_text
    r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = C_TITLE
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path,
            int(Inches(0.25)), int(Inches(fig_top)),
            int(Inches(9.50)), int(Inches(fig_h)))
    else:
        print(f'  ⚠  Missing figure: {img_path}')
    return len(prs.slides) - 1


# ═════════════════════════════════════════════════════════════════════════════
print('═' * 60)
print('PRE-BUILD VERIFICATION')
print('═' * 60)

SLIDES = [
    ('E07 assembly heatmap',
     os.path.join(adir, 'e07_assembly_heatmap.png'),
     'E07 × Cue 2: model captures the spatial pattern'),
    ('E07 joint captured',
     os.path.join(adir, 'e07_joint_captured.png'),
     'Signal routes through speed, not cue or position directly'),
]

all_ok = True
for label, path, title in SLIDES:
    all_ok &= verify_figure(path, label)
    all_ok &= verify_title(title, label)

if not os.path.exists(SRC):
    print(f'  ✗ SOURCE MISSING: {SRC}')
    all_ok = False
else:
    print(f'  ✓ Source exists: {os.path.basename(SRC)}')

if not all_ok:
    print('\n⚠  Verification failed — aborting build.')
    raise SystemExit(1)

print('\n✓ All checks passed — proceeding with build.\n')

# ═════════════════════════════════════════════════════════════════════════════
prs = Presentation(SRC)
print(f'Opened {os.path.basename(SRC)}: {len(prs.slides)} slides')

# Find insertion point: after S66 "E07: joint position × cue encoding"
anchor_idx = find_idx(prs, 'joint position')
if anchor_idx is None:
    anchor_idx = find_idx(prs, 'E07: joint')
if anchor_idx is None:
    print('  ⚠  Anchor slide not found — appending at end')
    anchor_idx = len(prs.slides) - 1
print(f'  Inserting after S{anchor_idx+1}: '
      f'{prs.slides[anchor_idx].shapes[0].text_frame.text[:50]}')

# Add slides at end, then move into position
for i, (label, path, title) in enumerate(SLIDES):
    new_idx = add_figure_slide(prs, title, path)
    target  = anchor_idx + 1 + i
    # Re-find previous slide in case indices shifted
    if i > 0:
        prev_title = SLIDES[i-1][2][:30]
        prev_idx = find_idx(prs, prev_title)
        target = prev_idx + 1 if prev_idx is not None else target
    move_slide(prs, new_idx, target)
    print(f'  → S{target+1:02d}: "{title}"')

print(f'\nFinal slide count: {len(prs.slides)}')

# Context check
anchor_idx = find_idx(prs, 'joint position')
if anchor_idx is not None:
    print('\nContext:')
    for i in range(anchor_idx, min(anchor_idx + 4, len(prs.slides))):
        sl = prs.slides[i]
        texts = [sh.text_frame.text.strip()[:70]
                 for sh in sl.shapes if sh.has_text_frame]
        print(f'  S{i+1:02d}: {next((t for t in texts if t), "(empty)")}')

# Save
for out in OUTS:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    prs.save(out)
    print(f'\nSaved → {out}')

# Post-build verification
print('\n' + '═' * 60)
print('POST-BUILD VERIFICATION')
print('═' * 60)
for out in OUTS:
    if os.path.exists(out):
        verify_zip(out)
