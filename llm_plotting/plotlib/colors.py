"""The color a trial label is drawn in, wherever it gets drawn.

`modality_colors` lived in svm_plotting_lib.py (and is still re-exported from there, since
svm_analysis_2026/lm4_plots import it by that name). It moved here when dim_plotting_lib's
trial grouping needed the same colors: a cue-1 trial should be the same color in the SVM
figures and in the population-geometry ones, and neither library imports the other - only
plotlib, which both already depend on.
"""

import numpy as np
import matplotlib.pyplot as plt


modality_colors = {
    'cue': {0: 'orange', 1: 'purple'},
    'choice_R1': {0: "#3094db", 1: "#0044ff"},
    'choice_R2': {0: "#7effda", 1: "#3aff5b"},
    'trial_outcome': {0: "#ff8585", 1: "#9dff9d"},
    'correct_R1': {0: "#690000", 1: "#004c00"},
    'correct_R2': {0: "#690000", 1: "#004c00"},
}

# modality_colors is keyed by 0/1 *level*, which is how the SVM fits encode every label -
# but Behavior40msAligned stores cue as 1/2 and everything else as 0/1, so a raw behavior
# value needs this subtracted before it can index the dict. The SVM side does the identical
# shift inline where it reads behavior (w_plotting_lib's `(per_trial['cue'] - 1).clip(0, 1)`);
# named here so a grouping figure doesn't have to rediscover it.
MODALITY_LEVEL_OFFSET = {'cue': 1}

# For a grouping that isn't one of the labelled modalities at all. Deliberately clear of the
# orange/dark red/light blue the neuron-contribution summary row already spends (see
# dim_plotting_lib._draw_neuron_contribution_summary).
FALLBACK_GROUP_COLORS = ('#762a83', '#01665e', '#c51b7d', '#7f6000', '#4d4d4d')

# The two reward zones' greys, taken from the stage diagram on the poster itself, where R1
# is the DARK zone and R2 the light one, both outlined. NB this is the opposite way round
# from lm4_plots.py's double_stop_performance cell (first location 'lightgrey', second
# 'darkgrey') - the diagram is what the poster reader sees next to these panels, so it wins,
# and each zone is then the same grey in the diagram and in every figure.
# Alpha belongs on the FACE only, never on the patch as a whole: these sit under the curves,
# so the fill has to stay see-through, but an outline faded to 35% stops reading as the crisp
# edge the diagram has. Moved here from behavior_plotting_lib (which re-exports all five
# under these names) when the poster's neural panel needed the same R1 band on its own
# position row - sibling *_plotting_libs cannot import each other.
ZONE_SPAN_COLORS = {'reward1Zone': '#4d4d4d', 'reward2Zone': '#d9d9d9'}
ZONE_SPAN_FALLBACK_COLOR = '#d9d9d9'
ZONE_SPAN_EDGE_COLOR = 'black'
ZONE_SPAN_LINEWIDTH = 1.0
ZONE_SPAN_ALPHA = 0.35

# How the stretch of track the CUE is rendered over is marked, wherever a figure marks it -
# hatched and unfilled rather than filled, because the cue zone is a different KIND of region
# from a reward location (a stimulus on the wall, not a place where an action pays off), and a
# third grey next to the reward zones' light/dark greys reads as a third reward zone.
# Defined in behavior_plotting_lib (which still re-exports all three under these names, so no
# call site there changed) until the poster's neural panel needed the identical band on its
# own kinematics row - the same one-layer-down move COMBO_DASH and modality_colors made.
CUE_SPAN_HATCH = '///'
CUE_SPAN_EDGE_COLOR = 'darkgrey'
CUE_SPAN_ALPHA = 0.55

# Ordered groupings (early -> late thirds of a session) get a sequential colormap instead of
# categorical colors, so their order is readable off the figure without the legend. Sampled
# from viridis' green->yellow end only: its dark-blue/purple start is hard to tell apart from
# black (the pooled line these are drawn over), so the useful span starts where the map turns
# green - hence 0.45-0.95 rather than the full range.
SEQUENTIAL_GROUPBYS = ('session_time',)
SEQUENTIAL_CMAP = 'viridis'
SEQUENTIAL_CMAP_RANGE = (0.45, 0.95)


def group_colors(groupby, values):
    """One color per entry of `values` (a grouping's own values, in display order).

    An ordered grouping (see SEQUENTIAL_GROUPBYS) is sampled from SEQUENTIAL_CMAP; anything
    else takes the canonical `modality_colors` entry for that label's level, falling back to
    FALLBACK_GROUP_COLORS for a grouping/level the dict doesn't cover (an unlabelled behavior
    column, or a `trial_outcome` value outside 0/1) rather than raising - a color is never
    worth failing a figure over, and the fallback is still stable per position.
    """
    if groupby in SEQUENTIAL_GROUPBYS:
        lo, hi = SEQUENTIAL_CMAP_RANGE
        fracs = np.linspace(lo, hi, len(values)) if len(values) > 1 else [(lo + hi) / 2]
        return [plt.get_cmap(SEQUENTIAL_CMAP)(f) for f in fracs]

    offset = MODALITY_LEVEL_OFFSET.get(groupby, 0)
    levels = modality_colors.get(groupby, {})
    colors = []
    for i, value in enumerate(values):
        try:
            level = int(value) - offset
        except (TypeError, ValueError):   # a string/None group value - no level to look up
            level = None
        colors.append(levels.get(level) or FALLBACK_GROUP_COLORS[i % len(FALLBACK_GROUP_COLORS)])
    return colors
