#!/usr/bin/env python3
"""
check_slide_overlaps.py

Checks ultimate_presentation_vN.pptx for two classes of layout problems:

  1. SHAPE OVERLAPS — two non-text shapes (pictures, rectangles) overlap each other
     on the same slide. Reports the pair and the overlap area in square inches.

  2. FIGURE CLIPPING — a picture shape extends beyond the slide boundary (off-screen).

  3. ASPECT RATIO MISMATCH — a picture's insertion aspect ratio differs from the
     source file's pixel aspect ratio by >5%. Indicates a stretched/squashed image.
     (Only checked if the picture's image data is accessible as a PNG/JPEG.)

Usage:
    python eval/check_slide_overlaps.py [path_to_pptx]

Defaults to outputs/ultimate_presentation_v23.pptx if no argument given.
Only reports problems — silent on clean slides.
"""
import os, sys
import io
from pptx import Presentation
from pptx.util import Inches

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def emu_to_in(emu):
    return emu / 914400


def shape_rect(sh):
    """Return (left, top, right, bottom) in inches."""
    l = emu_to_in(sh.left)
    t = emu_to_in(sh.top)
    w = emu_to_in(sh.width)
    h = emu_to_in(sh.height)
    return l, t, l + w, t + h


def overlap_area(r1, r2):
    """Overlap area in sq inches between two (l,t,r,b) rects. 0 if no overlap."""
    ox = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
    oy = max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    return ox * oy


def check_pptx(path):
    prs     = Presentation(path)
    sw      = emu_to_in(prs.slide_width)
    sh_     = emu_to_in(prs.slide_height)
    issues  = []
    n_clean = 0

    for slide_idx, slide in enumerate(prs.slides):
        slide_issues = []
        slide_num    = slide_idx + 1

        # Collect all shapes with geometry
        geom_shapes = [sh for sh in slide.shapes
                       if sh.left is not None and sh.width is not None]

        pictures = [sh for sh in geom_shapes if sh.shape_type == 13]

        # 1. Shape overlaps (picture × any other non-empty shape)
        for i, sh_a in enumerate(geom_shapes):
            for sh_b in geom_shapes[i+1:]:
                # Skip pure empty text boxes (no text, no fill)
                def is_empty_text(sh):
                    return (sh.has_text_frame and
                            not sh.text_frame.text.strip() and
                            sh.shape_type != 13)
                if is_empty_text(sh_a) or is_empty_text(sh_b):
                    continue
                area = overlap_area(shape_rect(sh_a), shape_rect(sh_b))
                if area > 0.01:   # ignore sub-0.01 sq-inch rounding overlaps
                    label_a = (sh_a.text_frame.text[:30].strip()
                               if sh_a.has_text_frame else f'[img]')
                    label_b = (sh_b.text_frame.text[:30].strip()
                               if sh_b.has_text_frame else f'[img]')
                    slide_issues.append(
                        f'  OVERLAP  {area:.2f} sq-in:  '
                        f'"{label_a or "[shape]"}"  ×  "{label_b or "[shape]"}"')

        # 2. Picture clipping (off-slide)
        for sh in pictures:
            l, t, r, b = shape_rect(sh)
            if l < -0.05 or t < -0.05 or r > sw + 0.05 or b > sh_ + 0.05:
                slide_issues.append(
                    f'  CLIPPED  picture at ({l:.2f}", {t:.2f}") '
                    f'{r-l:.2f}"×{b-t:.2f}" — slide is {sw:.2f}"×{sh_:.2f}"')

        # 3. Aspect ratio mismatch for pictures
        if HAS_PIL:
            for sh in pictures:
                try:
                    img_bytes = sh.image.blob
                    img       = PILImage.open(io.BytesIO(img_bytes))
                    px_w, px_h = img.size
                    px_ratio  = px_w / px_h
                    ins_w     = emu_to_in(sh.width)
                    ins_h     = emu_to_in(sh.height)
                    ins_ratio = ins_w / ins_h
                    diff      = abs(px_ratio - ins_ratio) / px_ratio
                    if diff > 0.05:
                        slide_issues.append(
                            f'  STRETCHED  picture: pixel ratio {px_ratio:.3f} '
                            f'vs insertion ratio {ins_ratio:.3f} '
                            f'({diff*100:.0f}% distortion)  '
                            f'[{px_w}×{px_h}px inserted as {ins_w:.2f}"×{ins_h:.2f}"]')
                except Exception:
                    pass

        if slide_issues:
            issues.append((slide_num, slide_issues))
        else:
            n_clean += 1

    return issues, n_clean, len(prs.slides)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
           os.path.join(os.path.dirname(__file__), '..', 'outputs',
                        'ultimate_presentation_v23.pptx')
    path = os.path.abspath(path)

    if not os.path.exists(path):
        print(f'File not found: {path}')
        sys.exit(1)

    print(f'Checking: {os.path.basename(path)}')
    if not HAS_PIL:
        print('  (PIL not available — aspect ratio checks skipped)')

    issues, n_clean, n_total = check_pptx(path)

    if not issues:
        print(f'✓  No layout issues found across all {n_total} slides.')
    else:
        print(f'\n{len(issues)} slide(s) with issues ({n_clean} clean):\n')
        for slide_num, slide_issues in issues:
            print(f'S{slide_num:02d}:')
            for msg in slide_issues:
                print(msg)
        print(f'\n{n_total - n_clean} / {n_total} slides have issues.')


if __name__ == '__main__':
    main()
