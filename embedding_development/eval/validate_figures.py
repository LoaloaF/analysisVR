"""
eval/validate_figures.py

Level 2.1 automated figure checks from eval_plan.md.
Run after regenerating all figures, before building the PPTX.

Usage:
    python eval/validate_figures.py

Checks every PNG in the manifest against:
  - Native size matches expected figsize (within 0.15")
  - Image is not blank (mean pixel < 250/255)
  - Image is landscape (width > height) for full-width figures
  - Filename does not contain raw column name fragments
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from utils.figure_style import FIG, DPI, validate_figure_file

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: PIL not available. Install Pillow for full validation.")

# ── Manifest: filename → (expected_figsize, must_be_landscape) ────────────────
# Update this dict whenever the manifest changes.
MANIFEST = {
    # Section 2 — Data
    'behavioral_trace.png':               (FIG.FULL,       True),

    # Section 3 — Models
    'r2_bar_linear.png':                  (FIG.HALF,       True),
    'r2_bar_mlp.png':                     (FIG.FULL,       True),
    'r2_consistency_bar.png':             ((6.5, 4.2),     True),

    # Section 5 — Attribution results
    'gpv_group_ensemble_heatmap.png':     (FIG.FULL,       True),
    'global_vs_cond_pv_scatter.png':      ((6.0, 4.2),     True),
    'ig_per_ensemble_heatmap.png':        (FIG.FULL,       True),
    'case_studies_e07_e23.png':           (FIG.FULL,       True),

    # Section 6 — Head angle
    'head_angle_tuning_6panel.png':       (FIG.FULL,       True),
    'head_angle_scatter_2x2.png':         (FIG.FULL,       True),
    'head_angle_stability.png':           (FIG.FULL,       True),
    'position_tuning.png':                ((6.0, 4.2),     True),

    # Section 7 — Frequency
    'trend_noise_comparison.png':         (FIG.FULL,       True),

    # Section 8 — TempConv results
    'r2_grand_mean_bars.png':             ((4.0, 4.0),     False),
    'two_thresholds_scatter.png':         ((5.0, 4.0),     True),
    'attribution_agreement_scatter.png':  (FIG.FULL,       True),

    # Section 9 — ML vs naive analysis
    'ml_vs_naive_scatter.png':            (FIG.FULL,       True),
    'nonlinearity_advantage.png':         (FIG.FULL,       True),
    'mlp_vs_linear_r2.png':              (FIG.FULL,       True),
    'ablation_proof.png':                (FIG.FULL,       True),

    # Section 10 — Cross-model attribution
    'group_attribution_comparison.png':   (FIG.FULL,       True),
}

# Raw column name fragments that must NOT appear as text in figure filenames
# (catches cases where a script used the raw name in the output filename)
RAW_NAME_FRAGMENTS = [
    'frame_raw_500ms', 'frame_YawPitch', 'frame_raw_abs',
    'head_angle_vel_',  # raw name used as filename component
]

SEARCH_DIRS = [
    './outputs/cebra_comparison',
    './outputs/mlps/ensembles_multiseed',
    './outputs/mlps/ml_vs_naive',
    './outputs/cebra_eval/ensembles',
    './outputs/temporal_advantage',
    '/mnt/c/Users/amits/Desktop',
]

def check_not_blank(path):
    if not PIL_AVAILABLE:
        return True
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    mean_val = arr.mean()
    if mean_val > 250:
        print(f"  [FAIL] {os.path.basename(path)}: image appears blank (mean pixel = {mean_val:.1f})")
        return False
    return True


def check_label_clipping(path):
    """
    Detect content (text / axes) cut off at the figure border.
    A 3-pixel strip at each edge is expected to be white background.
    Dark pixels there indicate that tight_layout + size-restore left labels clipped.
    """
    if not PIL_AVAILABLE:
        return True
    img = Image.open(path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    edges = {
        'top':    arr[:3,    :,    :],
        'bottom': arr[h-3:,  :,    :],
        'left':   arr[:,     :3,   :],
        'right':  arr[:,     w-3:, :],
    }
    ok = True
    for side, strip in edges.items():
        dark_frac = ((strip < 200).any(axis=2)).mean()
        if dark_frac > 0.008:
            print(f"  [FAIL] {os.path.basename(path)}: "
                  f"dark pixels at {side} edge ({dark_frac:.1%}) — label may be clipped")
            ok = False
    return ok

def check_landscape(path):
    if not PIL_AVAILABLE:
        return True
    img = Image.open(path)
    w, h = img.size
    if w <= h:
        print(f"  [FAIL] {os.path.basename(path)}: expected landscape but got {w}×{h}")
        return False
    return True

def check_no_raw_names_in_filename(filename):
    for frag in RAW_NAME_FRAGMENTS:
        if frag in filename:
            print(f"  [FAIL] filename contains raw column name fragment '{frag}': {filename}")
            return False
    return True

def run_validation():
    found = {}
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith('.png') and fname in MANIFEST:
                found[fname] = os.path.join(d, fname)

    print(f"\n{'='*60}")
    print(f"Figure validation — {len(found)}/{len(MANIFEST)} manifest files found")
    print(f"{'='*60}\n")

    failures = []
    missing  = []

    for fname, (expected_size, must_landscape) in sorted(MANIFEST.items()):
        if fname not in found:
            missing.append(fname)
            print(f"  [MISSING] {fname}")
            continue

        path = found[fname]
        ok = True

        ok &= check_no_raw_names_in_filename(fname)
        ok &= check_not_blank(path)
        ok &= check_label_clipping(path)

        if must_landscape:
            ok &= check_landscape(path)

        if PIL_AVAILABLE:
            try:
                validate_figure_file(path, expected_size)
            except AssertionError as e:
                print(f"  [FAIL] {e}")
                ok = False

        if not ok:
            failures.append(fname)

    print(f"\n{'='*60}")
    print(f"Results: {len(failures)} failures, {len(missing)} missing")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  {f}")
    if missing:
        print("Missing (not yet generated):")
        for f in missing:
            print(f"  {f}")
    if not failures and not missing:
        print("All figures passed.")
    print(f"{'='*60}\n")

    return len(failures) == 0 and len(missing) == 0

if __name__ == '__main__':
    success = run_validation()
    sys.exit(0 if success else 1)
