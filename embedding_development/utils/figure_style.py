"""
utils/figure_style.py

Single source of truth for figure style, canonical naming, and figure sizes.
Every eval/analysis script should import from here before generating any plot.

Usage:
    from utils.figure_style import (
        FIG, DPI, FONT, LINE, MARKER,
        MODEL_COLORS, FEATURE_NAMES_SHORT, PALETTE, CUE_COLORS, CUE_LABELS,
        apply_style, add_footnote, add_panel_label, savefig_manifest,
    )
    fig, ax = plt.subplots(figsize=FIG.FULL)
    apply_style(fig, ax)

Style standard: Nature Methods — sans-serif, no top/right spines, light horizontal
grid, outward ticks, no minor ticks, 200 dpi output.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (imported for side-effects: registers 'science' style)
import matplotlib.font_manager as _mpl_fm

# Register Arial from Windows fonts (WSL path) so it's available to all scripts
for _fp in [
    '/mnt/c/Windows/Fonts/arial.ttf',
    '/mnt/c/Windows/Fonts/arialbd.ttf',
    '/mnt/c/Windows/Fonts/ariali.ttf',
]:
    if os.path.exists(_fp):
        _mpl_fm.fontManager.addfont(_fp)

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

# ── Line width constants (points) ─────────────────────────────────────────────
class LINE:
    DATA   = 1.2   # primary data lines (traces, curves)
    THICK  = 1.8   # emphasis lines (e.g. group mean over individual traces)
    REF    = 0.8   # reference/diagonal/zero lines
    GRID   = 0.6   # grid lines (set in apply_style)
    SPINE  = 0.8   # axis spines

# ── Marker size constants ──────────────────────────────────────────────────────
class MARKER:
    SCATTER = 18   # matplotlib scatter `s` parameter (area units)
    JITTER  = 14   # jittered strip-plot dots
    SMALL   = 8    # small annotation dots

# ── Nature-Methods colour palette ─────────────────────────────────────────────
# Derived from Springer Nature's recommended discrete palette for figures.
# Use PALETTE[i] for categorical series when MODEL_COLORS doesn't apply.
PALETTE = [
    '#E64B35',  # 0 red
    '#4DBBD5',  # 1 cyan
    '#00A087',  # 2 teal
    '#3C5488',  # 3 navy
    '#F39B7F',  # 4 salmon
    '#8491B4',  # 5 periwinkle
    '#91D1C2',  # 6 mint
    '#DC0000',  # 7 dark red
    '#7E6148',  # 8 brown
    '#B09C85',  # 9 tan
]

# ── Cue-condition colours (used in all E07 cue-zone figures) ───────────────────
CUE_COLORS = {0: '#AAAAAA', 1: '#FF7F0E', 2: '#D62728'}
CUE_LABELS = {0: 'No cue',  1: 'Cue 1',   2: 'Cue 2'}

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
        # ── Fonts ──────────────────────────────────────────────────────────────
        'axes.titlesize':    FONT.LABEL,
        'axes.labelsize':    FONT.LABEL,
        'xtick.labelsize':   FONT.TICK,
        'ytick.labelsize':   FONT.TICK,
        'legend.fontsize':   FONT.LEGEND,
        # Nature-style sans-serif. Nimbus Sans = Helvetica clone; DejaVu fallback.
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Nimbus Sans', 'Ubuntu Sans', 'DejaVu Sans'],
        # ── Spines & ticks ─────────────────────────────────────────────────────
        'axes.spines.top':        False,
        'axes.spines.right':      False,
        'axes.linewidth':         LINE.SPINE,
        'xtick.direction':        'out',
        'ytick.direction':        'out',
        'xtick.top':              False,
        'ytick.right':            False,
        'xtick.minor.visible':    False,
        'ytick.minor.visible':    False,
        'xtick.major.width':      LINE.SPINE,
        'ytick.major.width':      LINE.SPINE,
        # ── Lines & markers ────────────────────────────────────────────────────
        'lines.linewidth':        LINE.DATA,
        'lines.markersize':       4,          # pts; use MARKER.SCATTER for scatter `s`
        'patch.linewidth':        LINE.SPINE,
        # ── Grid ───────────────────────────────────────────────────────────────
        'axes.grid':              True,
        'grid.alpha':             0.30,
        'grid.linewidth':         LINE.GRID,
        # ── Colour cycle (Nature Methods palette) ──────────────────────────────
        'axes.prop_cycle':        mpl.cycler('color', PALETTE),
        # ── Saving ─────────────────────────────────────────────────────────────
        # scienceplots sets savefig.bbox='tight', which breaks figsize invariant.
        'savefig.bbox':           None,
    }
    mpl.rcParams.update(rc)

    if axes is not None:
        ax_list = list(axes) if hasattr(axes, '__iter__') else [axes]
        for ax in ax_list:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ('left', 'bottom'):
                ax.spines[spine].set_linewidth(LINE.SPINE)
            ax.tick_params(labelsize=FONT.TICK, width=LINE.SPINE)
            ax.xaxis.label.set_size(FONT.LABEL)
            ax.yaxis.label.set_size(FONT.LABEL)
            # horizontal grid only (Nature style)
            ax.yaxis.grid(True, alpha=0.30, linewidth=LINE.GRID)
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

    # 6. Y-axis label pixel-bbox overlaps with a sibling axes window extent.
    #    Catches non-leftmost-column axes whose rotated y-label intrudes into the
    #    previous column's data area (a recurring issue in multi-column grids).
    try:
        for ax_i, ax in enumerate(fig.axes):
            yl = ax.yaxis.label
            if not yl.get_text().strip():
                continue
            yl_bb = yl.get_window_extent(renderer)
            for ax_j, ax2 in enumerate(fig.axes):
                if ax_i == ax_j:
                    continue
                ax2_bb = ax2.get_window_extent(renderer)
                if yl_bb.overlaps(ax2_bb):
                    found.append(
                        f"{filename}: ax[{ax_i}] y-label '{yl.get_text()[:20]}' "
                        f"overlaps ax[{ax_j}] data area — "
                        f"suppress ylabel on non-leftmost column panels"
                    )
    except Exception:
        pass

    # 7. Panel-label text (placed with transAxes) outside its own axes overlaps a
    #    sibling axes data area or its own title.  Only fires for transAxes texts to
    #    avoid false positives from data-coordinate bar/scatter annotations.
    try:
        ax_bboxes = [ax_j.get_window_extent(renderer) for ax_j in fig.axes]
        for ax_i, ax in enumerate(fig.axes):
            ax_own_bb = ax.get_window_extent(renderer)
            for txt in ax.texts:
                if not txt.get_text().strip():
                    continue
                # Only check texts placed in axis-fraction coordinates (panel labels)
                if txt.get_transform() is not ax.transAxes:
                    continue
                try:
                    t_bb = txt.get_window_extent(renderer)
                except Exception:
                    continue
                tol = 2  # px
                outside = (t_bb.x0 < ax_own_bb.x0 - tol or
                           t_bb.x1 > ax_own_bb.x1 + tol or
                           t_bb.y0 < ax_own_bb.y0 - tol or
                           t_bb.y1 > ax_own_bb.y1 + tol)
                if not outside:
                    continue
                for ax_j_i, ax_j in enumerate(fig.axes):
                    if ax_j is ax:
                        continue
                    if t_bb.overlaps(ax_bboxes[ax_j_i]):
                        found.append(
                            f"{filename}: ax[{ax_i}] panel label '{txt.get_text()[:12]}' "
                            f"is outside its axes and overlaps ax[{ax_j_i}] — "
                            f"move label inside axes or widen subplot spacing"
                        )
                        break
                title_obj = ax.title
                if title_obj.get_text().strip():
                    try:
                        title_bb = title_obj.get_window_extent(renderer)
                        if t_bb.overlaps(title_bb):
                            found.append(
                                f"{filename}: ax[{ax_i}] panel label '{txt.get_text()[:12]}' "
                                f"overlaps own title '{title_obj.get_text()[:25]}' — "
                                f"remove set_title or move label inside axes"
                            )
                    except Exception:
                        pass
    except Exception:
        pass

    # 8. Axes title bounding box clips beyond axes left/right edges (long title in narrow subplot).
    #    Skips axes narrower than 80 px (colorbar axes) where centered titles are intentionally wider.
    try:
        for ax_i, ax in enumerate(fig.axes):
            title_obj = ax.title
            if not title_obj.get_text().strip():
                continue
            try:
                t_bb  = title_obj.get_window_extent(renderer)
                ax_bb = ax.get_window_extent(renderer)
                if ax_bb.width < 80:  # colorbar or other intentionally narrow axis
                    continue
                if t_bb.x0 < ax_bb.x0 - 1:
                    found.append(
                        f"{filename}: ax[{ax_i}] title '{title_obj.get_text()[:30]}' "
                        f"clips left edge by {ax_bb.x0 - t_bb.x0:.0f}px — shorten title"
                    )
                if t_bb.x1 > ax_bb.x1 + 1:
                    found.append(
                        f"{filename}: ax[{ax_i}] title '{title_obj.get_text()[:30]}' "
                        f"clips right edge by {t_bb.x1 - ax_bb.x1:.0f}px — shorten title"
                    )
            except Exception:
                pass
    except Exception:
        pass

    # 9. Y-axis label bounding box extends beyond the axes top/bottom edge.
    #    Indicates the label text is too long and will bleed into a neighbouring row.
    try:
        for ax_i, ax in enumerate(fig.axes):
            ylab = ax.yaxis.label
            if not ylab.get_text().strip():
                continue
            try:
                yl_bb = ylab.get_window_extent(renderer)
                ax_bb = ax.get_window_extent(renderer)
                if ax_bb.height < 80:
                    continue
                if yl_bb.y0 < ax_bb.y0 - 4:
                    found.append(
                        f"{filename}: ax[{ax_i}] y-label '{ylab.get_text()[:25]}' "
                        f"clips {ax_bb.y0 - yl_bb.y0:.0f}px below axes — shorten label"
                    )
                if yl_bb.y1 > ax_bb.y1 + 4:
                    found.append(
                        f"{filename}: ax[{ax_i}] y-label '{ylab.get_text()[:25]}' "
                        f"clips {yl_bb.y1 - ax_bb.y1:.0f}px above axes — shorten label"
                    )
            except Exception:
                pass
    except Exception:
        pass

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
