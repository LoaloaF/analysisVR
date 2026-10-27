"""What a trial label is called, which colour it gets, and which level each trial has.

Split out of `svm_plotting_lib` (which still re-exports every name here under its old
private alias, so no call site changed) because a third library needed it: the activation
row lives in `var_plotting_lib`, and `var_plotting_lib` cannot import from
`w_plotting_lib` - the import order is svm -> var and w -> svm, so reaching back would
close a cycle. Same reasoning that moved `modality_colors` into `plotlib/colors.py`: the
knowledge belongs one layer down from whichever figure needed it first.

`trial_covariate_levels` is the only function here that touches data; everything else is
naming/colour tables.
"""

import numpy as np

from plotlib.colors import modality_colors


LEVEL_LABELS = {
    'cue': {0: 'C1', 1: 'C2'},
    'choice_R1': {0: 'skip', 1: 'stop'},
    'choice_R2': {0: 'skip', 1: 'stop'},
    'trial_outcome': {0: 'noR', 1: 'R'},
    'correct_R1': {0: 'wrong', 1: 'correct'},
    'correct_R2': {0: 'wrong', 1: 'correct'},
    # analysis='baseline_vs_t' rows: not a trial label - the two classes are the fixed
    # pre-event baseline window vs the sliding window at timebin t, same trials in both
    'baseline_vs_t': {0: 'pre', 1: 'at t'},
}

# Whether the R1/R2 choice was correct given that trial's cue. Cue1 (0) should stop
# at R1 and skip R2; cue2 (1) should skip R1 and stop at R2 (stop=1, skip=0 in
# choice_R1/choice_R2's own 0/1 encoding - see LEVEL_LABELS above).
CORRECTNESS_SPECS = [
    {'name': 'correct_R1', 'choice_name': 'choice_R1', 'is_correct': lambda cue, choice: cue != choice},
    {'name': 'correct_R2', 'choice_name': 'choice_R2', 'is_correct': lambda cue, choice: cue == choice},
]


# How a curve describing a JOINT condition cell (two labels at once, e.g. cue1 & skip) is
# drawn: the same line twice, once in each label's own colour, dashed with complementary
# phases so the two interleave into one alternating bicolour line rather than reading as two
# separate curves. Keep the phases complementary if these are ever retuned.
#
# Lived in var_plotting_lib (which still re-exports both names, so no call site there
# changed) until neuro_plotting_lib needed the identical encoding - a cue1&skip line must be
# the same line in both families of figures. Same reason modality_colors moved to
# plotlib/colors.py: the knowledge belongs one layer down from whichever figure needed it
# first. NB a legend swatch built from these needs HandlerTuple(ndivide=1), not the default
# ndivide=None, or the handle is halved into two short segments instead of one alternating
# line (see plot_w_variance_legend).
COMBO_DASH = ((0, (4, 4)), (4, (4, 4)))
COMBO_LINEWIDTH = 1.6


def level_label(y_name, level):
    return LEVEL_LABELS.get(y_name, {}).get(level, f'y={level}')


def covariate_color(y_name, level):
    return modality_colors.get(y_name, {}).get(level, 'gray')


def trial_covariate_levels(t0_events, s_id, trial_ids, covariate):
    """0/1 level for `covariate` - one of the fitted trial labels/'trial_outcome', or a
    derived CORRECTNESS_SPECS name ('correct_R1'/'correct_R2') - for each of
    `trial_ids`, in the same order. t0_events has one row per (trial, t0_event_name),
    but every one of these is constant per trial (see select_t0_events's base_info in
    get_TrialWiseT0Events40ms), so groupby('trial_id').first() collapses that safely.
    Plain covariates: t0_events' own 'cue'/'trial_outcome' are still raw (1/2, and
    0/1/2/3...) rather than the 0/1 encoding covariate_color/level_label/LEVEL_LABELS
    expect - that conversion normally happens inside get_SVMCueOutcomeChoicePred's
    _process_y_values, a later stage this does not go through - so it is replicated here.
    choice_R1/choice_R2 are already native 0/1. Derived (correct_R1/correct_R2): computed
    the same way plot_correctness_f1_single does, from this same adjusted cue plus the raw
    (already native 0/1) choice column that spec names."""
    per_trial = t0_events.loc[s_id].groupby('trial_id').first()

    spec = next((s for s in CORRECTNESS_SPECS if s['name'] == covariate), None)
    if spec is not None:
        cue = (per_trial['cue'] - 1).clip(0, 1)
        values = spec['is_correct'](cue, per_trial[spec['choice_name']]).astype(int)
    else:
        values = per_trial[covariate]
        if covariate == 'cue':
            values = (values - 1).clip(0, 1)
        elif covariate == 'trial_outcome':
            values = values.clip(0, 1)

    return np.asarray(values.loc[trial_ids].values)
