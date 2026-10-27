"""Is a fitted decoder direction a *dominant* pattern in the population, or a small one?

Fourth library of the SVM analysis line, next to `svm_plotting_lib` (one session's fit
sweep), `svm_heatmap_plotting_lib` (all sessions at a glance) and `w_plotting_lib` (one
hand-picked W, examined closely). The unit of analysis here is different again: *every*
timebin's W at once, as one extra row of the main per-session figure, answering two
questions the decoding-accuracy row above it cannot -

  * does this W carry a decent share of the population's variance, or is it a thin
    direction that happens to separate the classes                    (the black line)
  * is that subspace specific to this moment of the task, or is it there generally
                                                    (solid vs dashed - see below)

The first question is read against two references in the same panel: a ceiling (what one
of the population's leading directions carries, drawn as a bracket) and a floor (what a
random direction carries). The second is read off the same W's scored against the whole
recording, inter-trial time included.

The samples this is computed on replicate what the classifier was actually fitted on,
rather than approximating it: one sample per trial, being that trial's mean over the
three native 40ms bins of the fit window, L2-normalized if that fit's Normalizer step was
enabled (see plotlib.fit_samples - it is recorded per row, not assumed). See
`timebin_samples` for the provenance of every step of that, including the off-by-one in
the stored `timebin`.

Unlike `w_plotting_lib` (see its KNOWN DEBT note), nothing here loads its own data -
firing rate arrives as an argument, already sliced to one X_modality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from plotlib.export import save_figure
from plotlib.trial_traces import (smooth as _smooth, block_x as _block_x,
                                  next_block_offset as _next_block_offset, T0_LINE_KW,
                                  interval_short_label as _interval_short_label)
from plotlib import trial_labels
from plotlib.variance import (var_explained, var_explained_from_basis, prepare_var_basis,
                              var_explained_subset_null,
                               cv_pc_var_explained, insample_pc_var_explained,
                               random_direction_null, total_variance)
from plotlib import fit_samples as fitspl


# Re-exported, not defined here: the fit-window width and the L2 step it pairs with now
# live in plotlib.fit_samples, which is what `w_plotting_lib` projects through too - the
# two libraries drifting apart on this is exactly what that module exists to prevent.
TIMEBINS_PER_FIT = fitspl.TIMEBINS_PER_FIT

W_COLOR = 'black'
PC_COLOR = '#8a7fb8'
NULL_COLOR = '0.6'

# A joint (2x2) split's curve is drawn TWICE - once per label, in that label's own level
# colour, dashed with complementary phases so the dashes interleave into one alternating
# bicolour line. That encoding is the whole point of the joint panel: "cue1 AND stop" has
# to read as a combination of two conditions rather than as a fifth condition with a
# colour of its own, and a dashed line separates it at a glance from the solid single-
# label curves and the solid black pooled one. Keep the two patterns complementary (same
# on/off lengths, phase offset by exactly the on-length) or the line reads as two curves.
# moved to plotlib/trial_labels.py when neuro_plotting_lib needed the same encoding;
# re-exported here under the old names, so nothing in this module or its callers changed
COMBO_DASH = trial_labels.COMBO_DASH
COMBO_LINEWIDTH = trial_labels.COMBO_LINEWIDTH


# ------------------------------------------------- the classifier's own sample matrix


def get_interval_trial_stack(s_id, t0_events, fr_x, interval_name):
    """(n_trials, n_bins, n_features) activity for one session/interval, plus the
    matching trial_id array.

    Same per-trial windowing as `w_plotting_lib.get_trial_w_projection` and
    `get_session_data`'s `all_pos` (t0_events' own full_intrvl per trial, same modal-
    length trim), but kept as a full three-dimensional stack rather than projected or
    flattened - `timebin_samples` slices per timebin off this. Built once per interval
    and reused across every timebin on purpose: the `index.overlaps` lookup per trial is
    the expensive part here, and re-running it for each of ~30 timebins would dominate
    the whole row's cost for no reason.
    """
    trial_ids, trial_windows = [], []
    for trial_id, full_intrvl in t0_events.loc[s_id, interval_name].dropna().items():
        trial_ids.append(trial_id)
        trial_windows.append(fr_x[fr_x.index.overlaps(full_intrvl)].values)

    lengths = pd.Series([len(x) for x in trial_windows])
    correct_length = lengths.value_counts().index[0]
    keep = [len(x) == correct_length for x in trial_windows]
    stack = np.stack([x for x, k in zip(trial_windows, keep) if k])
    return stack, np.array([t for t, k in zip(trial_ids, keep) if k])


def timebin_samples(stack, timebin, timebins_per_fit=TIMEBINS_PER_FIT, normalize=True):
    """(n_trials, n_features) - the sample matrix the classifier at `timebin` was
    actually fitted on. Returns None if the window would run off either end of `stack`.

    Three facts from get_SVMCueOutcomeChoicePred are baked in here, all of them checked
    against that source rather than inferred:

    * **one sample per trial, being a mean over bins, not a per-bin sample.** The fit
      builds `X = np.stack([X_data[m].mean() for m in X_data_masks])` - each trial's
      feature vector is its average over the whole fit window. So the variance measured
      on this matrix is pure trial-to-trial variance, with no within-window temporal
      variability folded in (which is what would happen if the window's bins were
      flattened into the sample axis instead, as `get_interval_window_activity` does for
      a different purpose).
    * **the stored `timebin` is 1-based and names the window's CENTER**, not its start:
      the fit emits `result['timebin'] = ith_t + 1  # center of fit interval` while its
      mask covers native bins `ith_t .. ith_t + 2`. So stored timebin t is native bins
      `t-1 .. t+1`, and the naive `stack[:, t:t+3]` would be off by one bin.
    * **L2 normalization comes before the classifier** *when that step is enabled* - it
      is the first step of the fit's Pipeline (`Normalizer(norm='l2')` then the SVC), and
      applying it here is what makes a variance measured on these samples comparable to
      what the decoder saw. As a side effect it also makes the fraction comparable across
      sessions without further correction, since every sample then has unit norm and the
      total variance is exactly 1 - ||mean sample||^2. But the step is NOT always on: it
      is commented out in postproc_mea1k_ephys.py's current working tree, and
      svm_task_subspaces.py records per row (`normalizer`) whether it ran - which is why
      `normalize` here is resolved off the fitted table by default, see
      `w_variance_profile` and plotlib.fit_samples. `normalize=False` (i.e. FIT_RAW)
      keeps the fit window's mean but skips the L2.

    `normalize`: a bool or one of fit_samples' FIT_L2/FIT_RAW (see `fit_samples.fit_mode`
    for why RAW/L2_BIN cannot apply to a per-timebin sample matrix).
    """
    mode = fitspl.fit_mode(normalize)
    half = timebins_per_fit // 2
    start, end = timebin - half, timebin - half + timebins_per_fit
    if start < 0 or end > stack.shape[1]:
        return None
    # equivalently fit_samples.window_mean(stack)[:, timebin, :] - same centered window,
    # spelled out here because this function is called once per timebin and would
    # otherwise re-roll the whole stack each time
    samples = stack[:, start:end, :].mean(axis=1)
    return fitspl.l2_rows(samples) if mode == fitspl.FIT_L2 else samples


def session_samples(fr_x, timebins_per_fit=TIMEBINS_PER_FIT, normalize=True):
    """(n_session_bins, n_features) - the whole recording as one sample set: every native
    40ms bin, inter-trial time included, no t0_events or trial structure involved.

    Preprocessed identically to `timebin_samples` (rolling `timebins_per_fit`-bin mean,
    then L2 normalization if the fit used it) so a fraction measured here lands on the
    same axis as the per-timebin curve rather than merely near it. Without the rolling
    mean the samples would be single bins, which are noisier and would understate every
    fraction for reasons that have nothing to do with the direction being tested.

    Shares that preprocessing with `w_plotting_lib`'s whole-session projection via
    plotlib.fit_samples - the bins with no full window drop out here (they cannot be
    NaN-padded into a sample set), which is the one difference from the trial-aligned path.
    """
    mode = fitspl.fit_mode(normalize)
    rolled = fitspl.drop_nan_rows(fitspl.window_mean(fr_x.values, timebins_per_fit))
    if rolled.shape[0] == 0:
        return np.zeros((0, fr_x.shape[1]))
    return fitspl.l2_rows(rolled) if mode == fitspl.FIT_L2 else rolled


# ------------------------------------------------------------------- the profile


def split_names(split):
    """The label name(s) a split refers to, always as a tuple - ('cue',) or ('cue',
    'choice_R1'). Everything below branches on the *number* of names rather than on the
    type of the spec, so a joint split is not a special case threaded through by hand."""
    return (split,) if isinstance(split, str) else tuple(split)


def split_level_keys(split):
    """Every level key a split produces, in draw/legend order: 0 and 1 for one label, the
    four (level_a, level_b) combinations for a pair."""
    if isinstance(split, str):
        return [0, 1]
    return [(a, b) for a in (0, 1) for b in (0, 1)]


def split_level_label(split, key):
    """'stop', or 'C1 & stop' for a joint key."""
    if isinstance(split, str):
        return trial_labels.level_label(split, key)
    return ' & '.join(trial_labels.level_label(n, lvl) for n, lvl in zip(split_names(split), key))


def split_level_colors(split, key):
    """One colour per label in the split: a 1-tuple for a single label, a 2-tuple for a
    joint key - which is what makes the combo line bicolour (see COMBO_DASH)."""
    if isinstance(split, str):
        return (trial_labels.covariate_color(split, key),)
    return tuple(trial_labels.covariate_color(n, lvl)
                 for n, lvl in zip(split_names(split), key))


def split_level_masks(split, t0_events, s_id, trial_ids):
    """{level key: boolean mask over `trial_ids`} - one entry per split_level_keys entry.

    For a joint split this is the 2x2 cell membership, built by ANDing the two labels'
    own per-trial levels rather than by splitting twice, so an empty cell (no C1&skip
    trials at all) comes out as an all-False mask and simply leaves that curve NaN."""
    levels = [trial_labels.trial_covariate_levels(t0_events, s_id, trial_ids, name)
              for name in split_names(split)]
    masks = {}
    for key in split_level_keys(split):
        wanted = (key,) if isinstance(split, str) else key
        mask = np.ones(len(trial_ids), dtype=bool)
        for values, level in zip(levels, wanted):
            mask &= (values == level)
        masks[key] = mask
    return masks


def resolve_variance_splits(split, y_name=None):
    """Normalize `w_variance_split` into a list of one dict per PANEL.

    Three forms, and the third is the point of the feature:

    * `'cue'` - one panel, split by cue, **pooling over everything else**. Cheap and
      coarse: a cue1-vs-cue2 comparison here averages over skip and stop trials.
    * `{'split': 'choice_R1', 'given': ('cue', 1)}` - one panel using only cue==1 trials,
      split by choice_R1. The 2x2 cell comparison, with cue held fixed.
    * `{'split': 'choice_R1', 'given': 'cue'}` - **expands to one panel per level of cue**,
      i.e. the full 2x2 as two panels: (cue1: skip vs stop) and (cue2: skip vs stop). This
      is the form to reach for when the two conditioned panels are what you want to compare
      side by side, since pooling over the other dimension is what hides an interaction.
    * `('cue', 'choice_R1')` - a **TUPLE of two labels: the same 2x2 in ONE panel**, as four
      combo curves (see split_level_keys/COMBO_DASH) over one pooled black curve. Same four
      trial subsets as the `given` form, stacked instead of spread over panels - which is
      the readable version once both mirror directions are wanted, since
      `[{'split': 'choice_R1', 'given': 'cue'}, {'split': 'cue', 'given': 'choice_R1'}]`
      costs four panels to draw each of the four cells twice.

    **list vs tuple is the distinction**: a LIST is a list of panels, a TUPLE of two label
    names is one joint panel. So `['cue', 'choice_R1']` is two panels (each pooling over the
    other label) while `('cue', 'choice_R1')` is one panel of four combos, and
    `[('cue', 'choice_R1')]` is that same single panel written as a one-element panel list.
    None -> no panel at all. `y_name` is only used for the error message.

    Which variable is held fixed is the caller's decision and deliberately not inferred:
    for a cue decoder, holding cue fixed asks "inside one cue, does the axis distinguish
    what the animal did", while holding choice fixed asks the mirror question. Unknown keys
    raise rather than defaulting to a valid-looking panel.
    """
    if split is None:
        return []
    # a tuple IS one joint panel, so only a list is a list of panels (see the docstring)
    specs = split if isinstance(split, list) else [split]
    out = []
    for item in specs:
        if isinstance(item, str):
            out.append(dict(split=item, given=None))
            continue
        if isinstance(item, tuple):
            if len(item) != 2 or not all(isinstance(n, str) for n in item):
                raise ValueError(f"a joint w_variance_split is a tuple of exactly two label "
                                  f"names, got {item!r}")
            out.append(dict(split=item, given=None))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"w_variance_split entries are a label name, a two-label tuple "
                              f"or a dict, got {item!r}")
        unknown = set(item) - {'split', 'given'}
        if unknown:
            raise ValueError(f"unknown w_variance_split keys {sorted(unknown)} in {item!r}; "
                              f"allowed: 'split', 'given'")
        if 'split' not in item:
            raise ValueError(f"w_variance_split dict needs a 'split' label name: {item!r}")
        given = item.get('given')
        if isinstance(item['split'], tuple):
            # the joint form already IS the 2x2; conditioning it as well would leave a
            # single cell, which is a 'given' panel with one curve - ask for that directly
            if given is not None:
                raise ValueError(f"a joint split {item['split']!r} cannot also take "
                                  f"'given' - it is already the full 2x2")
            out.append(dict(split=tuple(item['split']), given=None))
        elif given is None:
            out.append(dict(split=item['split'], given=None))
        elif isinstance(given, str):
            # one panel per level of the held-fixed label - the 2x2 as two panels
            out.extend(dict(split=item['split'], given=(given, level)) for level in (0, 1))
        else:
            name, level = given
            out.append(dict(split=item['split'], given=(name, int(level))))
    return out


def _panel_label(spec):
    """"cue" for a pooled panel, "skip/stop | C2" for a conditioned one - and the colour of
    the held-fixed level, so the panel can say in one glance which subset of trials it is
    about (Simon's ask: "use color to annotate these are eg cue2 subset of trials"). A joint
    panel names both labels and gets no single colour - its curves carry two each."""
    split, given = spec['split'], spec['given']
    if not isinstance(split, str):
        return ' x '.join(split_names(split)), None
    levels = f"{trial_labels.level_label(split, 0)}/{trial_labels.level_label(split, 1)}"
    if given is None:
        return levels, None
    name, level = given
    return f"{levels} | {trial_labels.level_label(name, level)}", \
           trial_labels.covariate_color(name, level)


def w_variance_profile(s_id, t0_events, fr_x, svm_res, which_intervals, n_pcs=3,
                        show_session=False, normalize=fitspl.AUTO, rng=None, split=None,
                        given=None, split_perm_reps=200):
    """Everything `plot_w_variance_row` draws, computed once per (session, column).

    Per interval, per timebin, on that timebin's own fit samples (`timebin_samples`):
    the variance explained by that timebin's own W (in the trial window and against the
    whole recording), by the top `n_pcs` principal directions both cross-validated and
    in-sample, plus the random-direction chance band. All except session_w_var are
    fractions of that timebin's own total population variance, so they share one
    denominator and can be read against each other directly; session_w_var is a fraction
    of the whole recording's total instead - see plot_w_variance_row.

    `n_pcs` sets how many leading directions the ceiling averages over (see
    plot_w_variance_row); n_pcs=1 makes it PC1 alone. `show_session` adds the same W's
    scored against the whole recording - off by default because it is also the expensive
    part of this function (a rolling mean over every bin of the session), so skipping it
    is a real saving, and `session_w_var` stays all-nan when off.

    `normalize` is resolved off `svm_res` itself by default ('auto' ->
    fit_samples.mode_for_frame): whether the fit's L2 step ran is a property of the table
    being plotted, not a display preference, and a table predating the `normalizer` column
    falls back to fit_samples.LEGACY_POSTPROC_MODE with a printed notice.

    `split`, if given (a per-trial label name resolved by
    plotlib.trial_labels.trial_covariate_levels, or a TUPLE of two such names for the joint
    2x2 - see resolve_variance_splits), additionally scores the same W separately
    on each level's own trials - `w_var_split[level]`, keyed by 0/1 for one label and by
    (level_a, level_b) for a pair, **each as a fraction of that level's
    own total variance**, not of the pooled one. That choice is what makes the two curves
    comparable to each other and to the black pooled curve on one axis: the n-weighted
    additive terms (`pooled = Σ_g (n_g−1)/(n−1)·Var_g + between`, which does hold exactly)
    differ by the group sizes before any biology does, so two curves drawn from them read as
    a group difference when they are a count difference. Levels with fewer than 3 trials
    stay NaN. Measured consequence worth knowing: the pooled curve sits ABOVE both condition
    curves in every session tested, and that offset IS the between-condition (discriminative)
    part - a median 24-32% of the pooled value.

    Returns a dict with 'intervals' ({interval: {timebins, w_var, session_w_var, pc_var,
    pc_var_insample, null_lo, null_hi, n_trials, w_var_split, n_split}}), 'n_features',
    'n_pcs' and 'split'. Fractions, not percentages - the plotting function scales.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    svm_res = svm_res[svm_res.interval_name.isin(which_intervals)]
    # one resolution for the whole profile: every timebin's samples and the whole-session
    # basis have to be preprocessed the same way to share a denominator
    normalize = fitspl.mode_for_frame(svm_res, normalize)

    # Whole-session sample set, built once and reused for every timebin's W. This is the
    # "versus the overall data" half of the figure: the same W scored against everything
    # the population does across the recording, instead of against trial-to-trial spread
    # inside one task-aligned window.
    sess_centered, sess_total = (None, np.nan)
    if show_session:
        sess_centered, sess_total = prepare_var_basis(session_samples(fr_x, normalize=normalize))

    intervals = {}
    n_features = None
    for interval_name in which_intervals:
        rows = svm_res[svm_res.interval_name == interval_name].sort_values('timebin')
        if rows.empty:
            continue
        stack, trial_ids = get_interval_trial_stack(s_id, t0_events, fr_x, interval_name)
        n_features = stack.shape[2]
        # `given` restricts the panel to one level of another label FIRST, so every curve
        # and both reference bands describe that subset of trials and nothing else - the
        # black curve is then the subset's own pooled value, which is what the two split
        # curves below it decompose.
        if given is not None:
            g_name, g_level = given
            keep = trial_labels.trial_covariate_levels(t0_events, s_id, trial_ids, g_name) == g_level
            stack, trial_ids = stack[keep], np.asarray(trial_ids)[keep]
            if stack.shape[0] < 3:
                continue
        # by trial_id, not by position: the stack's modal-length trim can drop trials.
        # One mask per level key - two for a single label, four for a joint (2x2) split.
        masks = (split_level_masks(split, t0_events, s_id, trial_ids)
                 if split is not None else {})

        timebins = rows['timebin'].to_numpy(dtype=int)
        w_var = np.full(len(timebins), np.nan)
        session_w_var = np.full(len(timebins), np.nan)
        pc_var = np.full((n_pcs, len(timebins)), np.nan)
        pc_var_insample = np.full((n_pcs, len(timebins)), np.nan)
        null_lo = np.full(len(timebins), np.nan)
        null_hi = np.full(len(timebins), np.nan)
        n_trials = np.zeros(len(timebins), dtype=int)
        # per-condition curves: each level's OWN variance fraction (see the split note in
        # this function's docstring for why not the n-weighted additive terms)
        split_var = {k: np.full(len(timebins), np.nan) for k in masks}
        split_n = {k: int(m.sum()) for k, m in masks.items()}
        # permutation band per condition: what a RANDOM split of that condition's own size
        # gives, the only reference that accounts for the two groups having different trial
        # counts - see plotlib.variance.var_explained_subset_null.
        # Two-level splits only, and that is a readability limit rather than a technical
        # one: four bands behind four bicolour curves is exactly the pile-up the joint panel
        # exists to avoid, and it would also quadruple the reps. A joint panel's curves are
        # read against the grey random-direction band and against each other.
        split_null = ({k: np.full((2, len(timebins)), np.nan) for k in split_var}
                      if split_perm_reps and len(split_var) <= 2 else {})

        for i, (_, row) in enumerate(rows.iterrows()):
            samples = timebin_samples(stack, int(row['timebin']), normalize=normalize)
            if samples is None:
                continue  # fit window runs off the end of this interval's bins
            tot = total_variance(samples)
            if not np.isfinite(tot) or tot <= 0:
                continue
            n_trials[i] = samples.shape[0]
            w_t = np.asarray(row['w'], dtype=float).ravel()
            if w_t.size != n_features:
                # a collapsed fit (SVM_TaskSubspaces stores w=[] when no model survived)
                # has no direction to score - leave this timebin NaN, i.e. a gap in the
                # curve, rather than scoring a zero/garbage vector against the population
                continue
            w_var[i] = var_explained(samples, w_t)
            for level, values in split_var.items():
                group = samples[masks[level]]
                if group.shape[0] >= 3:   # 2 is the estimator's minimum; 3 keeps it sane
                    values[i] = var_explained(group, w_t)
                    if level in split_null:
                        split_null[level][:, i] = var_explained_subset_null(
                            samples, w_t, group.shape[0], reps=split_perm_reps, rng=rng)
            if show_session:
                session_w_var[i] = var_explained_from_basis(sess_centered, sess_total, w_t)
            pc_var[:, i] = cv_pc_var_explained(samples, n_pcs=n_pcs, rng=rng)
            pc_var_insample[:, i] = insample_pc_var_explained(samples, n_pcs=n_pcs)
            null_lo[i], null_hi[i] = random_direction_null(samples, rng=rng)

        intervals[interval_name] = dict(timebins=timebins, w_var=w_var,
                                        session_w_var=session_w_var, pc_var=pc_var, pc_var_insample=pc_var_insample,
                                        null_lo=null_lo, null_hi=null_hi,
                                        n_trials=n_trials,
                                        w_var_split=split_var, n_split=split_n,
                                        split_null=split_null)

    return dict(intervals=intervals, n_features=n_features, n_pcs=n_pcs, split=split,
                given=given)


# ------------------------------------------------------------------- the row itself


ACTIVATION_ZERO_KW = dict(color='0.35', linewidth=0.8, linestyle=':', zorder=1)
# Fixed and symmetric about the boundary by default, for the same reason w_variance_ylim
# is fixed: an autoscaled row makes one session's small displacements look like another's
# large ones on an assembled canvas. Symmetric because the quantity is signed and the two
# sides are the two classes - an off-centre view of it reads as a bias that isn't there.
# A scalar is taken as +-that. Pass None to autoscale.
ACTIVATION_YLIM = 0.3


def w_activation_profile(s_id, t0_events, fr_x, svm_res, which_intervals,
                          normalize=fitspl.AUTO, split=None, given=None):
    """Where each condition's trials SIT along the W, per timebin - the displacement
    question the variance row structurally cannot answer (see plot_w_variance_row: spread
    is not displacement, and a condition whose trials all activate consistently has the
    LOWEST within-condition variance).

    Per timebin: the trial mean of `(X @ w + b) / ||w||`, i.e. the **signed distance to
    that timebin's own decision boundary**, in the fit's own feature units. Dividing by
    ||w|| is what makes the number a distance rather than an arbitrarily scaled score, and
    is why timebins with differently-scaled weights are comparable along one curve; `b`
    (fit_samples.bias_for_w_row) is what makes zero mean the boundary rather than an
    arbitrary offset. A table with no stored `b` falls back to 0 with a one-time notice
    and `has_bias=False` - the curves are then still comparable to each other but their
    zero is not the boundary, which the row draws differently.

    Read it for the contrast the label coding does NOT force. Positive-vs-negative for
    the decoder's own two classes is circular: an SVM maximizes exactly that mean
    separation and w points from class 0 to class 1 by construction, so "class 1 sits
    higher" is a restatement of the fit. What is not forced is the OTHER label's effect
    inside one class - e.g. on a cue decoder, whether stop and skip trials of the same cue
    sit at different distances - which is why the joint 2x2 split is the form worth
    drawing here.

    `normalize`, `split`, `given` behave exactly as in w_variance_profile (same trial
    stack, same fit-sample replication, same masks). Returns dict(intervals={interval:
    {timebins, pooled, act_split, sem_split, n_split}}, split, given, has_bias).
    """
    svm_res = svm_res[svm_res.interval_name.isin(which_intervals)]
    normalize = fitspl.mode_for_frame(svm_res, normalize)
    has_bias = 'b' in svm_res.columns and bool(svm_res['b'].notna().any())

    intervals = {}
    for interval_name in which_intervals:
        rows = svm_res[svm_res.interval_name == interval_name].sort_values('timebin')
        if rows.empty:
            continue
        stack, trial_ids = get_interval_trial_stack(s_id, t0_events, fr_x, interval_name)
        n_features = stack.shape[2]
        if given is not None:
            g_name, g_level = given
            keep = trial_labels.trial_covariate_levels(t0_events, s_id, trial_ids, g_name) == g_level
            stack, trial_ids = stack[keep], np.asarray(trial_ids)[keep]
            if stack.shape[0] < 3:
                continue
        masks = (split_level_masks(split, t0_events, s_id, trial_ids)
                 if split is not None else {})

        timebins = rows['timebin'].to_numpy(dtype=int)
        pooled = np.full(len(timebins), np.nan)
        act = {k: np.full(len(timebins), np.nan) for k in masks}
        sem = {k: np.full(len(timebins), np.nan) for k in masks}
        for i, (_, row) in enumerate(rows.iterrows()):
            samples = timebin_samples(stack, int(row['timebin']), normalize=normalize)
            if samples is None:
                continue  # fit window runs off the end of this interval's bins
            w_t = np.asarray(row['w'], dtype=float).ravel()
            norm = float(np.linalg.norm(w_t))
            # same two gaps the variance row has: a collapsed fit (w=[]) and an all-zero w
            # (L1 shrank everything) have no direction, so the curve breaks rather than
            # reporting a number computed from a zero vector
            if w_t.size != n_features or not np.isfinite(norm) or norm == 0:
                continue
            d = (samples @ w_t + fitspl.bias_for_w_row(row)) / norm
            pooled[i] = np.nanmean(d)
            for k, m in masks.items():
                if m.sum() < 3:
                    continue
                act[k][i] = np.nanmean(d[m])
                sem[k][i] = np.nanstd(d[m], ddof=1) / np.sqrt(int(m.sum()))
        intervals[interval_name] = dict(timebins=timebins, pooled=pooled, act_split=act,
                                        sem_split=sem, n_split={k: int(m.sum()) for k, m in masks.items()})
    return dict(intervals=intervals, split=split, given=given, has_bias=has_bias)


def plot_w_activation_row(ax, profile, which_intervals, interval_x0s, smooth=None,
                           show_ylabel=True, ylim=ACTIVATION_YLIM, show_sem=True):
    """One row: each condition cell's mean signed distance to the decision boundary.

    Same x scaffold, same colours and the same bicolour COMBO_DASH encoding as
    plot_w_variance_row, so a cell is the same line in both rows - the two answer
    different questions about it (how far out it sits vs how much it fluctuates).
    Opacity again scales with the cell's trial count (see _split_alpha).

    `ylim` defaults to ACTIVATION_YLIM (a scalar, read as +-it), fixed rather than
    autoscaled so the row means the same thing in every panel of a canvas. It clips - the
    curves run wider than +-0.3 on some sessions, and clipping the excursions is the right
    trade for comparability, but raise it if a cell spends whole intervals off-axis.

    The dotted horizontal line is the decision boundary (y=0) when the table stores `b`;
    with no stored intercept it is dropped entirely rather than drawn somewhere
    meaningless. `show_sem` adds each cell's +-1 SEM as a faint band - worth keeping,
    because a 2x2 cell can be 5-9 trials and the mean curve alone gives no sense of that.

    What NOT to read off it: the split between the decoder's own two classes is forced by
    the fit (see w_activation_profile). The comparison this row exists for is the other
    label's effect within one class.
    """
    intervals = profile['intervals']
    split = profile.get('split')
    offset = 0
    drawn_intervals = [iv for iv in which_intervals if iv in intervals]

    for i, interval_name in enumerate(drawn_intervals):
        d = intervals[interval_name]
        x = _block_x(d['timebins'], offset)
        first = i == 0
        if profile.get('has_bias'):
            ax.axhline(0, **ACTIVATION_ZERO_KW)
        n_max = max(d['n_split'].values() or [0])
        for level in split_level_keys(split) if split is not None else []:
            values = d['act_split'].get(level)
            if values is None:
                continue
            y = _smooth(values, smooth)
            colors = split_level_colors(split, level)
            alpha = _split_alpha(d['n_split'].get(level, 0), n_max)
            label = (f"{split_level_label(split, level)} (n={d['n_split'][level]})") if first else None
            if show_sem:
                e = _smooth(d['sem_split'][level], smooth)
                ax.fill_between(x, y - e, y + e, color=colors[0], alpha=0.12 * alpha,
                                linewidth=0, zorder=2)
            if len(colors) == 1:
                ax.plot(x, y, color=colors[0], linewidth=1.4, alpha=alpha, zorder=3, label=label)
            else:
                for j, (color, dashes) in enumerate(zip(colors, COMBO_DASH)):
                    ax.plot(x, y, color=color, linewidth=COMBO_LINEWIDTH, alpha=alpha,
                            zorder=3, linestyle=dashes, label=label if j == 0 else None)
        if split is None:
            ax.plot(x, _smooth(d['pooled'], smooth), color=W_COLOR, linewidth=1.6, alpha=0.75,
                    zorder=4, label='all trials' if first else None)

        if interval_x0s and interval_name in interval_x0s:
            ax.axvline(x=interval_x0s[interval_name] + offset, zorder=0, **T0_LINE_KW)
        offset = _next_block_offset(offset, len(d['timebins']), is_last=i == len(drawn_intervals) - 1)

    if ylim is not None:
        lo, hi = (-ylim, ylim) if np.isscalar(ylim) else ylim
        ax.set_ylim(lo, hi)
    if show_ylabel:
        ax.set_ylabel('dist. to bound.' if profile.get('has_bias') else 'proj. (no b)')
    ax.spines[['top', 'right']].set_visible(False)


def _pc_band_label(n_pcs):
    """The ceiling band's label. One definition, used by both the row and the standalone
    legend, so the k they name can never disagree with each other."""
    return (f'top {n_pcs} PCs, avg (CV - in-sample)' if n_pcs > 1
            else 'PC1 (CV - in-sample)')


def plot_w_variance_legend(save_dir=None, fname='w_variance_legend', n_pcs=3,
                            show_session=False, split=None, fontsize=7, figsize=(2.6, 0.6),
                            dpi=150, ncol=2):
    """The variance row's key, as its own small figure - draw once, place by hand.

    The row has no in-panel legend at all, and not because it looked untidy: at a readable
    font this box needs ~2.6in of width, while a single-interval column of the main figure
    is ~1.4in (width_per_interval in plot_session_analysis), so an in-axes legend overflows
    the panel entirely and prints across whatever sits to its left. Exporting the key
    separately is the same answer show_ylabel already encodes for these figures - draw the
    shared furniture once and position it against the assembled canvas.

    `split`, if given, adds one entry per condition curve, in the same order and the same
    style the row draws them - including a joint split's bicolour dashed swatch, which is
    drawn as the two Line2D's overlaid across the full handle (HandlerTuple(ndivide=1);
    the default ndivide=None halves the handle instead and gives two short segments). Pass
    the SAME value the row was drawn with. Its per-condition trial counts are deliberately
    not here: they are per session, while this key is drawn once for a whole canvas, so
    they live in the panel's own corner label.

    `n_pcs` and `show_session` must match what the row was drawn with, or the key will
    describe a different figure. Saved to save_dir as PNG+SVG if given (see
    plotlib.export.save_figure). Returns the figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')
    handles = [
        Patch(facecolor=NULL_COLOR, alpha=0.25, label='chance (random direction)'),
        Patch(facecolor=PC_COLOR, alpha=0.30, label=_pc_band_label(n_pcs)),
        Line2D([], [], color=W_COLOR, linewidth=2, alpha=0.75,
                label="each bin's W, in trial window"),
    ]
    if show_session:
        handles.append(Line2D([], [], color=W_COLOR, linewidth=1.4, alpha=0.75,
                               linestyle='--', label="same W, whole session"))
    labels = [h.get_label() for h in handles]
    for level in (split_level_keys(split) if split is not None else []):
        colors = split_level_colors(split, level)
        if len(colors) == 1:
            handles.append(Line2D([], [], color=colors[0], linewidth=1.4, alpha=0.85))
        else:
            handles.append(tuple(Line2D([], [], color=c, linewidth=COMBO_LINEWIDTH,
                                        alpha=0.9, linestyle=dashes)
                                 for c, dashes in zip(colors, COMBO_DASH)))
        labels.append(f'{split_level_label(split, level)} trials only')
    ax.legend(handles, labels, loc='center', ncol=ncol, fontsize=fontsize, frameon=False,
               handlelength=2.4 if split is not None and not isinstance(split, str) else 1.6,
               labelspacing=0.4, columnspacing=1.0, borderpad=0,
               handler_map={tuple: HandlerTuple(ndivide=1)})
    fig.tight_layout(pad=0.1)
    if save_dir is not None:
        save_figure(fig, save_dir, fname, dpi=dpi)
    return fig


SPLIT_ALPHA_FLOOR = 0.3


def _split_alpha(n, n_max):
    """Curve opacity from that cell's trial count, relative to the biggest cell in the
    session: a 5-trial curve should not read as confidently as a 40-trial one. Floored at
    SPLIT_ALPHA_FLOOR so a small cell stays visible rather than fading out entirely."""
    if not n_max:
        return 1.0
    return float(np.clip(n / n_max, SPLIT_ALPHA_FLOOR, 1.0))


def plot_w_variance_row(ax, profile, which_intervals, interval_x0s, smooth=None,
                         show_ylabel=True, show_session=False, show_pc_band=True,
                         ylim=(0, 8), pc_smooth=2):
    """One row of the main per-session figure: variance explained over timebins.

    Placed between the decoding-score row and the W-similarity heatmap, sharing their
    x-axis scaffold exactly - plotlib.trial_traces.block_x/next_block_offset, i.e. the
    heatmap's own block-row index, one separator row between intervals - so a dip here
    lines up with the timebin it belongs to in the rows above and below.

    Three things, a measurement between a ceiling and a floor:

    * **black solid** - each timebin's own W, scored on the trials in that bin's own fit
      window. Alone, this is uninterpretable: 5% could be a lot or nothing. Hence the two
      references.
    * **purple band, the ceiling** - "the average share of the variance carried by each of
      the population's top `n_pcs` (3) directions". Same per-direction units as the W, so
      it compares directly. Averaged rather than PC1 alone for two reasons: PC1 is the
      unattainable maximum while "a typical leading direction" is what being a dominant
      pattern actually looks like, and the top-k *subspace* is well estimated at this
      sample size even when the ordering inside it is not. Set `n_pcs=1` on
      w_variance_profile to get PC1 alone instead.

      It is a **bracket, not a line**: lower edge cross-validated, upper edge in-sample.
      At ~80 trials against ~54 units neither is trustworthy alone and their biases run
      opposite - on synthetic data with a near-flat spectrum, CV read 1.98% and in-sample
      6.05% against a truth of 2.75%. A wide band is itself information: the spectrum is
      flat and no leading direction is well defined.
    * **gray band, the floor** - what a random direction produces. A band rather than a
      line at 1/n_features because the spread around that mean is wide (see
      random_direction_null). Expect the black line to sit inside it wherever decoding is
      at chance: the stored W there is an average of ~200 bootstrap fits to noise, so it
      should behave like a random direction. If it does not, that is a real result, not a
      bug - the axis is locked onto a large non-discriminative pattern.

    `show_session` adds a **black dashed** curve: the same W's scored against the whole
    recording instead of the trial window. Solid above dashed says the direction accounts
    for more of what varies across trials at this moment than of what the population does
    in general - i.e. a task subspace rather than just a big direction. Off by default:
    it needs `w_variance_profile(show_session=True)` to have computed anything, and the
    two denominators are deliberately different (each curve is a share of the variance in
    its own setting), so it is a comparison to reach for deliberately rather than carry
    always. Read the gap; don't subtract.

    With a `split` profile (see w_variance_profile) two more curves appear, in
    `plotlib.trial_labels.covariate_color`: the same W scored on one condition's trials at a
    time, each against that condition's own total variance, and each with a **permutation
    band in its own colour**: 5-95% of what a random split of that condition's trial count
    gives (`split_perm_reps`, see plotlib.variance.var_explained_subset_null). A curve
    inside its own band is a subset of trials, not a condition - and that band widens by
    itself for the smaller group, which the shared grey band cannot do, being estimated at
    the full trial count (measured 1.26pp wide against this null's 1.5pp).
    Read three ways - the two
    against each other (which condition the axis actually fluctuates in), their distance
    below the black curve (that offset is the between-condition, i.e. discriminative, part -
    a median 24-32% of the pooled value, and the pooled curve sat above both condition
    curves in every session tested), and each against the grey chance band. Measured on real
    sessions: pooled clears its chance p95 in 67-95% of bins, level 1 in 30-76%, **level 0
    in only 5-30%** - i.e. "this W is a dominant direction" can be a claim carried by one
    condition, which the pooled curve alone cannot show. **Nothing here is forced by the label
    coding**, which is the reason this row carries the comparison rather than a projection
    row: an SVM maximizes the mean separation between the classes, so any mean-difference
    readout is circular and its sign is fixed by which class is called 1 - but it does not
    optimize the within-condition variance ratio, so this comparison is far less circular
    (asymmetry outside a label-permutation null in 38-57% of bins, 5% expected).

    `smooth` applies the same rolling mean as the score row above (see _smooth), per
    interval, to the W curves; `pc_smooth` (default 2) does the same to the PC ceiling
    band's two edges. The grey chance band stays unsmoothed - it is a backdrop, not an
    estimate whose bin-to-bin noise is in the way.

    Each condition curve's **opacity scales with its own trial count**, relative to the
    largest cell in that session and floored at SPLIT_ALPHA_FLOOR - so a 2x2 cell with 5
    trials is visibly fainter than one with 40 without disappearing.

    `ylim` is fixed at (0, 8) percent rather than autoscaled, so the row means the same
    thing in every session panel of an assembled canvas - autoscaling made a session with
    one wide ceiling band look, at a glance, like one where the W sat lower. Pass None to
    autoscale from 0. Note a fixed limit can clip: the ceiling band's in-sample edge runs
    high when the spectrum is flat, and clipping it is usually the right trade (it is the
    optimistic edge of a reference, not a measurement), but raise the top if the W curve
    itself ever reaches it.
    """
    intervals = profile['intervals']
    offset = 0
    drawn_intervals = [iv for iv in which_intervals if iv in intervals]

    for i, interval_name in enumerate(drawn_intervals):
        d = intervals[interval_name]
        x = _block_x(d['timebins'], offset)
        first = i == 0

        ax.fill_between(x, d['null_lo'] * 100, d['null_hi'] * 100, color=NULL_COLOR, alpha=0.25,
                        linewidth=0, zorder=1, label='chance (random dir.)' if first else None)

        if show_pc_band:
            k = profile['n_pcs']
            # smoothed like the W curves (pc_smooth, default 2): the bracket's two edges
            # are noisy per-bin estimates and the band is read as a level, not per timebin
            ax.fill_between(x, _smooth(np.nanmean(d['pc_var'][:k], axis=0), pc_smooth) * 100,
                            _smooth(np.nanmean(d['pc_var_insample'][:k], axis=0), pc_smooth) * 100,
                            color=PC_COLOR, alpha=0.30, linewidth=0, zorder=2,
                            label=_pc_band_label(k) if first else None)

        ax.plot(x, _smooth(d['w_var'], smooth) * 100, color=W_COLOR, linewidth=2, alpha=0.75,
                zorder=4, label="each bin's W, in trial window" if first else None)

        # Per-condition curves (profile built with split=): the same W scored on one
        # condition's trials at a time, each against that condition's OWN total variance -
        # so all three curves and both reference bands share one axis and one meaning.
        # Three readings, which is why this is worth the two extra lines: the two curves
        # against each other (which condition the axis fluctuates in), their distance below
        # the black curve (the between-condition part, i.e. the discriminative bit), and
        # each against the grey chance band (measured: the per-condition chance p95 is
        # within 0.2pp of the pooled one, so that band is a valid reference for all three -
        # no extra furniture needed).
        split = profile.get('split')
        for level, band in sorted(d.get('split_null', {}).items()):  # two-level splits only
            # 5-95% of what a random split of THIS condition's size gives. A curve inside
            # its own band is a subset of trials, not a condition. Unsmoothed, like the
            # other two reference bands - a backdrop, not a measurement.
            ax.fill_between(x, band[0] * 100, band[1] * 100,
                            color=split_level_colors(split, level)[0], alpha=0.16,
                            linewidth=0, zorder=1,
                            label=(f"{split_level_label(split, level)}: random split "
                                   f"of the same size") if first else None)
        for level in split_level_keys(split) if split is not None else []:
            values = d.get('w_var_split', {}).get(level)
            if values is None:
                continue
            y = _smooth(values, smooth) * 100
            colors = split_level_colors(split, level)
            alpha = _split_alpha(d['n_split'].get(level, 0), max(d['n_split'].values() or [0]))
            label = (f"{split_level_label(split, level)} trials only"
                     f" (n={d['n_split'][level]})") if first else None
            if len(colors) == 1:
                ax.plot(x, y, color=colors[0], linewidth=1.4, alpha=alpha, zorder=3,
                        label=label)
            else:
                # one curve, drawn once per label in that label's own colour with
                # complementary dash phases: the dashes interleave into a single bicolour
                # dashed line, which is what says "this is a combination of two conditions"
                # (see COMBO_DASH). Only the first pass carries the legend label, so the
                # curve is one legend entry, not two.
                for j, (color, dashes) in enumerate(zip(colors, COMBO_DASH)):
                    ax.plot(x, y, color=color, linewidth=COMBO_LINEWIDTH, alpha=alpha,
                            zorder=3, linestyle=dashes, label=label if j == 0 else None)

        if show_session:
            ax.plot(x, _smooth(d['session_w_var'], smooth) * 100, color=W_COLOR, linewidth=1.4,
                    alpha=0.75, linestyle='--', zorder=4,
                    label="same W, whole session" if first else None)

        # t0 marker, same geometry as every other row of this figure (see _interval_x0
        # in svm_plotting_lib - interval_x0s is measured from t0_events, never hardcoded).
        if interval_x0s and interval_name in interval_x0s:
            ax.axvline(x=interval_x0s[interval_name] + offset, zorder=0, **T0_LINE_KW)

        offset = _next_block_offset(offset, len(d['timebins']),
                                    is_last=i == len(drawn_intervals) - 1)

    # Which trials this panel is about, coloured by the held-fixed level - so a "| C2"
    # panel is labelled in cue2's own colour and a stack of panels can be told apart at a
    # glance. Colour only in the text: tinting the axes background was tried and dropped,
    # because the two levels' colours here are orange and purple and the purple tint is
    # indistinguishable from the PC ceiling band drawn over it.
    # The trial counts belong in this label, not in a legend: a conditioned panel can be
    # down to a handful of trials in one of its two curves (5 skip trials inside cue1 on a
    # real session), and a curve that noisy has to be labelled as such where it is read.
    label, label_color = _panel_label(profile) if profile.get('split') else (None, None)
    if label is not None:
        split = profile['split']
        n_split = intervals[drawn_intervals[0]]['n_split'] if drawn_intervals else {}
        if n_split and isinstance(split, str):
            label += f"  ({n_split.get(0, 0)}/{n_split.get(1, 0)})"
        elif n_split:
            # A joint panel has four curves and no in-panel legend, so the cell names go
            # here with their own counts, two per line - a bare '(31/5/12/40)' would need
            # the reader to know the key order, and these counts are exactly what tells a
            # thin curve from a real one (a 2x2 cell really does come down to 5 trials).
            keys = split_level_keys(split)
            label += '\n' + '\n'.join(
                '   '.join(f'{split_level_label(split, k)} {n_split.get(k, 0)}'
                           for k in keys[i:i + 2]) for i in (0, 2))
        ax.text(0.99, 0.95, label, transform=ax.transAxes, fontsize=6, ha='right', va='top',
                color=label_color or '0.35',
                bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none',
                          alpha=0.65))

    if show_ylabel:
        # Short and single-line on purpose: this row is ~half the height of the score row
        # above it, so a two-line rotated label overruns into the neighbouring column.
        ax.set_ylabel('% var. expl.', fontsize=8)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelbottom=False, labelsize=7)
    # At the height this row gets (w_variance_row_ratio, ~half the score row above it) the
    # default locator packs in more ticks than there is room to label.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    return ax
