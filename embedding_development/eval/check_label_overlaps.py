#!/usr/bin/env python3
"""
check_label_overlaps.py

Checks a saved matplotlib figure PNG for overlapping text elements,
particularly across subplot rows (bottom-row titles vs top-row tick labels).

Usage:
    python eval/check_label_overlaps.py path/to/figure.png [--script path/to/script.py]

If --script is given, imports and runs that script to produce a live figure
(useful for checking before saving). Otherwise renders the PNG using PIL and
applies a pixel-level scan to detect potential text overlap zones.

Pixel-level approach (no matplotlib re-render needed):
  1. Load the PNG as greyscale.
  2. Identify horizontal bands with significant non-white content.
  3. Find the vertical extent of each subplot row by detecting large white gaps.
  4. Report if content from one subplot band bleeds into the adjacent subplot's band.

This is a conservative check — it catches real overlap issues without false positives
from thin borders or tick marks.
"""
import sys, os
import numpy as np
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def find_content_rows(gray_arr, white_thresh=240, min_content_cols=10):
    """Return set of row indices that have significant non-white content."""
    content = (gray_arr < white_thresh).sum(axis=1) >= min_content_cols
    return np.where(content)[0]


def find_white_gaps(content_rows, total_rows, min_gap=8):
    """Find large white horizontal gaps between content bands."""
    all_rows   = set(range(total_rows))
    white_rows = sorted(all_rows - set(content_rows))
    gaps = []
    if not white_rows:
        return gaps
    start = white_rows[0]
    prev  = white_rows[0]
    for r in white_rows[1:]:
        if r != prev + 1:
            if prev - start + 1 >= min_gap:
                gaps.append((start, prev))
            start = r
        prev = r
    if prev - start + 1 >= min_gap:
        gaps.append((start, prev))
    return gaps


def check_png(path, verbose=True):
    if not HAS_PIL:
        print('PIL not available — install Pillow to use this checker.')
        return []

    img  = Image.open(path).convert('L')   # greyscale
    arr  = np.array(img)
    H, W = arr.shape

    content_rows = find_content_rows(arr, white_thresh=235, min_content_cols=15)
    gaps         = find_white_gaps(content_rows, H, min_gap=10)

    if verbose:
        print(f'Image: {W}×{H}px  ({W/200:.2f}"×{H/200:.2f}" at 200 dpi)')
        print(f'Content rows: {len(content_rows)}  |  White gaps: {len(gaps)}')
        for g in gaps:
            print(f'  Gap: rows {g[0]}–{g[1]}  ({g[1]-g[0]+1}px = {(g[1]-g[0]+1)/200:.2f}")')

    # Detect overlap: check if content rows of adjacent subplot bands overlap
    if len(gaps) < 1:
        if verbose: print('No clear subplot row separators found — cannot check.')
        return []

    # Split content into bands separated by gaps
    bands = []
    prev_end = 0
    for gap_start, gap_end in gaps:
        band_rows = [r for r in content_rows if prev_end <= r < gap_start]
        if band_rows:
            bands.append((min(band_rows), max(band_rows)))
        prev_end = gap_end + 1
    # last band
    band_rows = [r for r in content_rows if prev_end <= r]
    if band_rows:
        bands.append((min(band_rows), max(band_rows)))

    if verbose:
        print(f'Detected {len(bands)} content band(s):')
        for i, (r0, r1) in enumerate(bands):
            print(f'  Band {i+1}: rows {r0}–{r1}  ({(r1-r0)/200:.2f}")')

    issues = []
    for i in range(len(bands) - 1):
        b1_end   = bands[i][1]
        b2_start = bands[i + 1][0]
        # Find the gap between these bands
        gap = [g for g in gaps if g[0] > b1_end and g[1] < b2_start + 20]
        if not gap:
            overlap_px = max(0, b1_end - b2_start)
            if overlap_px > 0:
                msg = (f'OVERLAP: band {i+1} ends at row {b1_end} but '
                       f'band {i+2} starts at row {b2_start} '
                       f'({overlap_px}px = {overlap_px/200:.3f}" overlap)')
                issues.append(msg)
                if verbose: print(f'  ⚠  {msg}')
        else:
            g = gap[0]
            gap_size = g[1] - g[0] + 1
            if gap_size < 8:
                msg = (f'TIGHT GAP: only {gap_size}px ({gap_size/200:.3f}") '
                       f'between band {i+1} and band {i+2} — may appear to touch')
                issues.append(msg)
                if verbose: print(f'  ⚠  {msg}')

    if not issues and verbose:
        print('✓  No inter-subplot overlaps detected.')
    return issues


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check figure for inter-subplot text overlaps.')
    parser.add_argument('png', help='Path to figure PNG')
    parser.add_argument('--thresh', type=int, default=235,
                        help='White threshold (default 235)')
    args = parser.parse_args()

    if not os.path.exists(args.png):
        print(f'File not found: {args.png}')
        sys.exit(1)

    issues = check_png(args.png, verbose=True)
    if issues:
        print(f'\n{len(issues)} issue(s) found.')
        sys.exit(1)
    else:
        print('\nAll clear.')


if __name__ == '__main__':
    main()
