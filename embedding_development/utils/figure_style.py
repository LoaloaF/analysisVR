"""
utils/figure_style.py

Single source of truth for figure style, canonical naming, and figure sizes.
Every eval/analysis script should import from here before generating any plot.

Usage:
    from utils.figure_style import (
        FEATURE_NAMES, MODEL_NAMES, AXIS_LABELS,
        FIG, apply_style, label_feature_axis, savefig_manifest
    )
    fig, ax = plt.subplots(figsize=FIG.FULL)
    apply_style(fig, ax)
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

# ── Canonical feature names ────────────────────────────────────────────────────
# Keys: raw column names used in the data files.
# Values: human-readable labels used on all axes and legends.
FEATURE_NAMES = {
    'frame_raw_500msMedian':                   'Forward Speed (cm/s)',
    'frame_raw_abs_acc_500msMedian':           'Forward Acceleration (cm/s²)',
    'frame_YawPitch_abs_vel_sum_500msMedian':  'Rotational Velocity (°/s)',
    'frame_YawPitch_abs_acc_sum_500msMedian':  'Rotational Acceleration (°/s²)',
    'head_angle_vel':                          'Head Angular Velocity (°/s)',
    'head_angle':                              'Head Angle (°)',
    'frame_position':                          'Track Position (cm)',
    'cue_visible':                             'Cue Visible',
    'upcoming_choice':                         'Upcoming Choice',
    'reward_window':                           'Reward Window',
    'lick_detected':                           'Lick Detected',
}

# Short versions for tick labels on dense heatmaps (≤ 12 chars)
FEATURE_NAMES_SHORT = {
    'frame_raw_500msMedian':                   'Fwd Speed',
    'frame_raw_abs_acc_500msMedian':           'Fwd Accel.',
    'frame_YawPitch_abs_vel_sum_500msMedian':  'Rot. Vel.',
    'frame_YawPitch_abs_acc_sum_500msMedian':  'Rot. Accel.',
    'head_angle_vel':                          'Head Ang. Vel.',
    'head_angle':                              'Head Angle',
    'frame_position':                          'Track Pos.',
    'cue_visible':                             'Cue Visible',
    'upcoming_choice':                         'Up. Choice',
    'reward_window':                           'Reward Win.',
    'lick_detected':                           'Lick Det.',
}

# ── Canonical model names ──────────────────────────────────────────────────────
MODEL_NAMES = {
    'MLP':           'MLP',
    'TempConv-Cont': 'TempConv-Cont',
    'TempConv-Pred': 'TempConv-Pred',
    'Linear':        'Linear',
}

MODEL_COLORS = {
    'Linear':        '#8B6552',
    'MLP':           '#2CA02C',
    'TempConv-Cont': '#1F77B4',
    'TempConv-Pred': '#FF7F0E',
}

# ── Canonical axis label strings ───────────────────────────────────────────────
AXIS_LABELS = {
    'r2':              'R²',
    'r2_drop':         'Permutation importance (ΔR²)',
    'cond_r2_drop':    'Conditional permutation importance (ΔR²)',
    'ig':              'Mean |IG| attribution',
    'pearson_r':       'Mean Pearson r (cross-seed, test set)',
    'cohens_d':        "Cohen's d",
    'session':         'Session',
    'ensemble':        'Ensemble (sorted by mean R²)',
    'head_angle':      'Head Angle (°)',
    'head_angle_vel':  'Head Angular Velocity (°/s)',
    'position':        'Track Position (cm)',    # confirmed: −169 to 270 cm
    'activity':        'Ensemble activity (z-scored)',
    'trend_r2':        'Slow-component R²',
    'noise_r2':        'Fast-component R²',
}

# ── Figure size constants (width, height) in inches ───────────────────────────
# All figures should use one of these to ensure consistent sizing in the PPTX.
# Slide content area: ~9.5" wide × 4.5" tall (10" slide minus title bar).
# Save at 200 dpi. Insert into PPTX at EXACT native size — no rescaling.
class FIG:
    FULL        = (9.5, 4.2)   # single figure filling content area
    TWO_THIRDS  = (7.0, 4.2)   # figure with bullet column to the right
    HALF        = (4.5, 4.2)   # one panel of a two-panel side-by-side slide
    WIDE_SHORT  = (9.5, 3.5)   # multi-row grid (many small panels)
    SQUARE      = (4.0, 4.0)   # square scatter or bar
    SMALL_SQ    = (3.0, 3.5)   # small square for 2×2 grids

DPI = 200

# ── Font size constants ────────────────────────────────────────────────────────
class FONT:
    TICK        = 13   # axis tick labels — minimum for readability on projector
    LABEL       = 15   # axis labels (xlabel, ylabel)
    LEGEND      = 13   # legend text
    PANEL       = 16   # panel labels (A, B, C, ...)
    FOOTNOTE    = 11   # technical footnotes placed outside axes
    ANNOTATION  = 12   # in-plot plain-text annotations (no box, no arrow)

# ── Base matplotlib style ──────────────────────────────────────────────────────
def apply_style(fig=None, axes=None):
    """
    Apply the standard presentation style to a figure and its axes.

    Call immediately after plt.subplots():
        fig, ax = plt.subplots(figsize=FIG.FULL)
        apply_style(fig, ax)

    Rules enforced:
    - No figure suptitle (set the slide title in PowerPoint instead)
    - No axes title (ax.set_title is removed — use panel labels if needed)
    - Top and right spines removed
    - Font sizes set to FONT constants
    - Grid: light horizontal lines only (no vertical)
    """
    plt.style.use(['science', 'no-latex'])

    rc = {
        'axes.titlesize':    0,          # suppress ax.set_title rendering size; callers should not call set_title
        'axes.labelsize':    FONT.LABEL,
        'xtick.labelsize':   FONT.TICK,
        'ytick.labelsize':   FONT.TICK,
        'legend.fontsize':   FONT.LEGEND,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linewidth':    0.6,
        # Nature-style sans-serif font. Nimbus Sans is a Helvetica metric clone
        # available on most Linux systems; DejaVu Sans is the universal fallback.
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Nimbus Sans', 'Ubuntu Sans', 'Arial', 'DejaVu Sans'],
        # scienceplots sets inward ticks on all 4 sides + minor ticks visible.
        # Override to outward ticks on bottom/left only — standard for
        # presentation figures.
        'xtick.direction':        'out',
        'ytick.direction':        'out',
        'xtick.top':              False,
        'ytick.right':            False,
        'xtick.minor.visible':    False,
        'ytick.minor.visible':    False,
        # scienceplots sets savefig.bbox='tight', which crops every savefig call
        # to the content bounding box and breaks the figsize invariant for PPTX
        # placement.  Override to None so the canvas size is preserved exactly.
        'savefig.bbox': None,
    }
    mpl.rcParams.update(rc)

    if axes is not None:
        ax_list = list(axes) if hasattr(axes, '__iter__') else [axes]
        for ax in ax_list:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=FONT.TICK)
            ax.xaxis.label.set_size(FONT.LABEL)
            ax.yaxis.label.set_size(FONT.LABEL)
            # horizontal grid only
            ax.yaxis.grid(True, alpha=0.3, linewidth=0.6)
            ax.xaxis.grid(False)



def add_footnote(fig, text):
    """
    Add a technical footnote at the bottom of the figure (outside axes).
    Use for: sample sizes, thresholds, seed counts.
    Do NOT use for statistical results — those go as plain text inside the axes.

    Example:
        add_footnote(fig, "5 seeds × 29 sessions; pairs with R² < 0.01 excluded")
    """
    fig.text(
        0.01, 0.01, text,
        ha='left', va='bottom',
        fontsize=FONT.FOOTNOTE,
        color='dimgray',
        transform=fig.transFigure,
    )


def add_panel_label(ax, label, x=-0.08, y=1.05):
    """
    Add a bold panel label (A, B, C...) in the upper-left corner of an axis.
    Coordinates are in axis fraction.
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONT.PANEL, fontweight='bold', va='top', ha='right')


def label_feature_axis(ax, axis='y', short=False):
    """
    Replace raw column-name tick labels on an axis with canonical feature names.

    axis: 'x' or 'y'
    short: use FEATURE_NAMES_SHORT instead of FEATURE_NAMES (for dense heatmaps)

    Example (after seaborn heatmap sets y-tick labels from raw column names):
        label_feature_axis(ax, axis='y', short=True)
    """
    name_map = FEATURE_NAMES_SHORT if short else FEATURE_NAMES
    if axis == 'y':
        labels = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticklabels([name_map.get(l, l) for l in labels],
                           fontsize=FONT.TICK)
    else:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        ax.set_xticklabels([name_map.get(l, l) for l in labels],
                           fontsize=FONT.TICK, rotation=45, ha='right')


# ── Overlap / out-of-bounds checks ────────────────────────────────────────────

def _overlaps_1d(a0, a1, b0, b1, tol=1.0):
    """True if intervals [a0,a1] and [b0,b1] overlap by more than tol pixels."""
    return (a1 - b0) > tol and (b1 - a0) > tol


def _warn_overlaps(fig, filename, renderer):
    """
    Check a rendered figure for layout problems and print [OVERLAP] warnings.
    Called by savefig_manifest after layout is finalised.

    Checks:
      1. Adjacent x-tick labels whose bounding boxes intersect
      2. Y-axis label whose bbox extends left of the figure canvas (clipped)
      3. X-axis label whose bbox extends below the figure canvas (clipped)
      4. Legend bbox that intersects any ax.texts annotation bbox
      5. Two axes bboxes that overlap by >10 % of the smaller axis area
         (catches a colorbar placed on top of a data panel)
    """
    found = []

    for ax_i, ax in enumerate(fig.axes):
        tag = f"{filename} ax[{ax_i}]"

        # 1. x-tick label overlap (sorted left-to-right by x0)
        xticks = sorted(
            [t for t in ax.get_xticklabels() if t.get_text().strip()],
            key=lambda t: t.get_window_extent(renderer).x0,
        )
        for k in range(len(xticks) - 1):
            bb_k  = xticks[k].get_window_extent(renderer)
            bb_k1 = xticks[k + 1].get_window_extent(renderer)
            if _overlaps_1d(bb_k.x0, bb_k.x1, bb_k1.x0, bb_k1.x1):
                px = bb_k.x1 - bb_k1.x0
                found.append(f"{tag}: x-tick labels overlap by {px:.0f}px "
                              f"('{xticks[k].get_text()}' ∩ '{xticks[k+1].get_text()}')")

        # 2. y-axis label left-edge clip
        yl = ax.yaxis.label
        if yl.get_text().strip():
            bb = yl.get_window_extent(renderer)
            if bb.x0 < -1:
                found.append(f"{tag}: y-label '{yl.get_text()[:30]}' "
                              f"clips left edge by {-bb.x0:.0f}px")

        # 3. x-axis label bottom-edge clip
        xl = ax.xaxis.label
        if xl.get_text().strip():
            bb = xl.get_window_extent(renderer)
            if bb.y0 < -1:
                found.append(f"{tag}: x-label '{xl.get_text()[:30]}' "
                              f"clips bottom edge by {-bb.y0:.0f}px")

        # 4. legend vs. annotation text overlap
        leg = ax.get_legend()
        if leg is not None:
            try:
                leg_bb = leg.get_window_extent(renderer)
                for txt in ax.texts:
                    if not txt.get_text().strip():
                        continue
                    t_bb = txt.get_window_extent(renderer)
                    if (_overlaps_1d(leg_bb.x0, leg_bb.x1, t_bb.x0, t_bb.x1) and
                            _overlaps_1d(leg_bb.y0, leg_bb.y1, t_bb.y0, t_bb.y1)):
                        found.append(f"{tag}: legend overlaps annotation "
                                     f"'{txt.get_text()[:25]}'")
            except Exception:
                pass

    # 5. axes-bbox overlap (in figure-fraction coordinates)
    positions = [(i, ax.get_position()) for i, ax in enumerate(fig.axes)]
    for i in range(len(positions)):
        ai, pi = positions[i]
        for j in range(i + 1, len(positions)):
            aj, pj = positions[j]
            ix0, ix1 = max(pi.x0, pj.x0), min(pi.x1, pj.x1)
            iy0, iy1 = max(pi.y0, pj.y0), min(pi.y1, pj.y1)
            if ix1 > ix0 and iy1 > iy0:
                overlap   = (ix1 - ix0) * (iy1 - iy0)
                min_area  = min(pi.width * pi.height, pj.width * pj.height)
                if min_area > 1e-6 and overlap / min_area > 0.10:
                    found.append(f"{filename}: axes[{ai}] and axes[{aj}] overlap "
                                 f"({overlap / min_area:.0%} of smaller axis)")

    for msg in found:
        print(f"  [OVERLAP] {msg}")
    return found


def savefig_manifest(fig, filename, out_dirs, skip_tight_layout=False):
    """
    Save figure to one or more output directories at DPI=200.
    Enforces that no rescaling happens: figure is saved at its native figsize.

    filename: e.g. 'r2_bar_mlp.png'
    out_dirs: list of directory paths (e.g. ['outputs/cebra_comparison', '/mnt/c/Users/amits/Desktop'])
    skip_tight_layout: set True for figures with manually-positioned axes (e.g. GridSpec
        with explicit margins or a manually-placed colorbar), so tight_layout does not
        interfere with the pre-set layout.

    After saving, prints the native size in inches so it can be verified against
    the manifest's expected figsize.
    """
    # Save intended size first — tight_layout can resize the canvas as a side
    # effect in some matplotlib/scienceplots combinations.  Restoring enforces
    # the size invariant required for PPTX placement at native size.
    w_in = fig.get_figwidth()
    h_in = fig.get_figheight()

    if not skip_tight_layout:
        try:
            # rect=[0.04, …] provides a 4 % left margin so that rotated y-axis
            # labels (which read bottom-to-top) cannot be clipped by the left
            # canvas edge, while the 4 % bottom/top reserves space for footnotes
            # and panel labels respectively.
            fig.tight_layout(rect=[0.04, 0.04, 0.99, 0.96])
        except Exception:
            pass

    # Disable constrained_layout so it cannot override our size during savefig,
    # then restore canvas to the intended figsize.
    try:
        fig.set_constrained_layout(False)
    except Exception:
        pass
    fig.set_size_inches(w_in, h_in)

    # Overlap / out-of-bounds check — runs at the figure's native DPI; best-effort.
    try:
        fig.canvas.draw()
        _warn_overlaps(fig, filename, fig.canvas.get_renderer())
    except Exception:
        pass

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        # Do NOT use bbox_inches='tight' — it resizes the output and breaks
        # the size invariant required for PPTX placement at native size.
        fig.savefig(path, dpi=DPI)
        print(f"  Saved {path}  [{w_in:.2f}\" × {h_in:.2f}\" at {DPI} dpi]")
    plt.close(fig)


# ── Validation helper ──────────────────────────────────────────────────────────
def validate_figure_file(png_path, expected_figsize, tolerance_in=0.15):
    """
    Assert that a saved PNG has the expected native size in inches.
    Raises AssertionError if outside tolerance.

    expected_figsize: (width_in, height_in) from the manifest
    tolerance_in: allowed deviation in inches (default 0.15")

    Returns (actual_w_in, actual_h_in).
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [validate] PIL not available — skipping size check")
        return None

    img = Image.open(png_path)
    px_w, px_h = img.size
    actual_w = px_w / DPI
    actual_h = px_h / DPI
    exp_w, exp_h = expected_figsize

    ok_w = abs(actual_w - exp_w) <= tolerance_in
    ok_h = abs(actual_h - exp_h) <= tolerance_in
    status = "OK" if (ok_w and ok_h) else "FAIL"
    print(f"  [validate] {os.path.basename(png_path)}: "
          f"{actual_w:.2f}\"×{actual_h:.2f}\" "
          f"(expected {exp_w:.2f}\"×{exp_h:.2f}\") — {status}")
    if not (ok_w and ok_h):
        raise AssertionError(
            f"Figure size mismatch: got {actual_w:.2f}\"×{actual_h:.2f}\", "
            f"expected {exp_w:.2f}\"×{exp_h:.2f}\". "
            f"Check figsize= in the generating script against the manifest."
        )
    return actual_w, actual_h
