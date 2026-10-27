"""Loading and shaping the session tables every notebook here starts from.

The one module in `plotlib/` that deliberately does I/O - it exists so the identical
~25-line preamble stopped being copy-pasted into svm_analysis_2026.py, lm4_plots.py and
dim_analysis.py with small silent divergences between the copies.

Its other job is the ANALYTIC VERSION PAIRING (see ANALYTIC_PAIRS below). Several
generations of both the t0-events table and the SVM fit table exist on the NAS under
different names, with *different interval geometry*, and mixing two generations produces
figures that are silently misaligned rather than an error. `check_geometry` turns that
into a loud failure.
"""

import pandas as pd

from analytics_processing import analytics
from analytics_processing.sessions_from_nas_parsing import (
    sessionlist_fullfnames_from_args, fullfnames2snames)


BIN_US = 40_000            # native Behavior40msAligned / FiringRate40ms* bin width
TIMEBINS_PER_FIT = 3       # timeline_decoding's sliding window, postproc_mea1k_ephys.py

# Unit columns are ordered HP first, then mPFC, on one probe's sequential sorted-cluster
# index (Unit0001..Unit00NN). The split point is a property of the recording, not of any
# one plot - it was previously hardcoded as a bare `.iloc[:, 20:]` in three places
# (svm_analysis_2026's _x_modality_fr, dim_analysis inline, and the fit itself in
# get_SVMCueOutcomeChoicePred). Slicing by unit NAME rather than position, as
# modality_slice does, additionally survives a table that carries extra non-unit columns.
N_HP_UNITS = 20
ALL_X_MODALITIES = ['HP', 'mPFC', 'HP+mPFC']


# Which t0-events table each SVM fit table was actually computed against. Verified
# 2026-08-11 by inverting each fit's own timebin count (window_bins = n_timebins +
# TIMEBINS_PER_FIT - 1) and comparing it to the real interval widths in each t0 table:
#
#   analytic                        sessions  cue/R1/R2 window bins   CCGP  n_bins_left
#   SVMCueOutcomeChoicePred            26      40 / 45 / 40           no    no
#   SVML1CueOutcomeChoicePred          15      42 / 42 / 42           no    no
#   SVML1CueOutcomeChoicePred2         28      25 / 34 / 40           NO    YES
#   SVML1CueOutcomeChoicePredCCGP      12      40 / 40 / 40           YES   no
#   TrialWiseT0Events40ms  (stale)             40 / 40|45 / 40
#   TrialWiseT0Events40ms2 (current)           25 / 34 / 40
#
# Note the awkward split that this table makes explicit: the *current-geometry* fit
# (SVML1CueOutcomeChoicePred2) has no real CCGP rows, and the only fit that does
# (…CCGP) is on the stale geometry. Any CCGP figure is therefore still on the old
# t0 table until that analytic is recomputed.
ANALYTIC_PAIRS = {
    # Successor analytic (ephys_preprocessing/svm_task_subspaces.py). Listed so
    # t0_analytic_for does not raise on it, but it does not depend on the pairing the way
    # the older tables do: it stamps n_bins_left/n_bins_right/window_bins on every row from
    # the t0 table it was actually cut against, so a mismatch is visible in the data itself.
    'SVM_TaskSubspaces':             't0_events2',
    'SVML1CueOutcomeChoicePred2':    't0_events2',   # current; use this by default
    'SVML1CueOutcomeChoicePredCCGP': 't0_events',    # stale geometry, but the only CCGP
    # Orphaned: its 42-bin fit windows match NEITHER t0 table (stale stores 40/45,
    # current 25/34). check_geometry always reports a mismatch for it - that is correct,
    # not a false alarm. Nothing plotted from it can be aligned; refit or drop it.
    'SVML1CueOutcomeChoicePred':     't0_events',
    'SVMCueOutcomeChoicePred':       't0_events',    # oldest
}
T0_ANALYTICS = {'t0_events': 'TrialWiseT0Events40ms',      # stale
                't0_events2': 'TrialWiseT0Events40ms2'}    # current
DEFAULT_SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2'


def t0_analytic_for(svm_analytic):
    """Name of the t0-events analytic that `svm_analytic` was fit against - see
    ANALYTIC_PAIRS. Raises on an unknown name rather than guessing, since guessing wrong
    is exactly the silent-misalignment failure this module exists to prevent."""
    if svm_analytic not in ANALYTIC_PAIRS:
        raise KeyError(f"unknown SVM analytic {svm_analytic!r}; known: {sorted(ANALYTIC_PAIRS)}. "
                       f"Add it to ANALYTIC_PAIRS with the t0 table it was fit against.")
    return T0_ANALYTICS[ANALYTIC_PAIRS[svm_analytic]]


def get_session_names(paradigm_ids=(1100,), animal_ids=(6,), session_ids=None):
    """(session_names, ith_to_dt_session) for one animal/paradigm selection.

    `ith_to_dt_session` maps a short session id ('2024-11-26_16-39', which is what every
    analytic's own index uses) to its 0-based chronological index - the "S7"/"S13" numbers
    in every figure label. Both notebooks built this dict inline, under two different
    names (`ith_to_dt_session` here, `session_to_dt_session` in dim_analysis) for the same
    thing."""
    session_dirs, _ = sessionlist_fullfnames_from_args(list(paradigm_ids), list(animal_ids), session_ids)
    session_names = fullfnames2snames(session_dirs)
    ith_to_dt_session = {sn[:16]: i for i, sn in enumerate(session_names)}
    return session_names, ith_to_dt_session


def load_t0_events(session_names, analytic='TrialWiseT0Events40ms2'):
    """t0-events table indexed by (session_id, trial_id), one row per detected t0 event.

    Each `<name>_interval` column holds a pd.Interval of ephys time for that event's own
    window and is NaN on every row belonging to a different event - see the notes in
    svm_analysis_2026.py. Defaults to the current (…40ms2) table; pass the bare name to
    match an older fit (see t0_analytic_for)."""
    t0_events = analytics.get_analytics(analytic, session_names=session_names)
    t0_events.index = t0_events.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
    t0_events.set_index('trial_id', append=True, inplace=True)
    return t0_events


def load_behavior(session_names, columns=None, choice_r1_as_bool=False):
    """Behavior40msAligned on a right-closed 40ms IntervalIndex of ephys time, indexed
    (interval, session_id) - the shape every plotting function here expects, so that a
    trial's window can be selected with `.index.overlaps(full_intrvl)`.

    `columns` narrows the fetch (dim_analysis passes an explicit list; the SVM notebooks
    take everything). `choice_r1_as_bool` reproduces svm_analysis_2026's one extra step;
    it is off by default because nothing actually depends on the dtype."""
    behavior = analytics.get_analytics('Behavior40msAligned', session_names=session_names,
                                        columns=columns)
    behavior = behavior.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
    behavior.index = pd.IntervalIndex.from_arrays(
        left=behavior.to_ephys_timestamp - BIN_US,
        right=behavior.to_ephys_timestamp,
        closed='right')
    behavior.set_index('session_id', append=True, inplace=True)
    if choice_r1_as_bool:
        behavior['choice_R1'] = behavior['choice_R1'].astype(bool)
    return behavior


def load_trialwise(session_names, ith_to_dt_session=None):
    """BehaviorTrialwise - one row per trial - indexed by (session_id, trial_id), with an
    `s_id` column holding the chronological session number when `ith_to_dt_session` is
    given (see get_session_names).

    The trial-level source of truth for everything a figure groups trials by: 'cue' (raw
    1/2 here, not the fits' 0/1 - see plotlib.trial_labels.trial_covariate_levels),
    'choice_R1'/'choice_R2' (native 0/1, skip/stop), their matching
    'choice_R1_correct'/'choice_R2_correct', 'trial_outcome', and the task-configuration
    flags ('both_R1_R2_rewarded', 'flip_Cue1R1_Cue2R2', 'maximum_reward_number').

    NB `choice_R1_correct` is stored, not derived - verified 2026-09-06 to agree on every
    trial of sessions 0-15 with the cue-vs-choice rule CORRECTNESS_SPECS applies
    (cue1 should stop at R1 and skip R2, cue2 the reverse), so the two are
    interchangeable and the stored column is the cheaper one."""
    trialwise = analytics.get_analytics('BehaviorTrialwise', session_names=session_names)
    trialwise = trialwise.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
    if ith_to_dt_session is not None:
        trialwise['s_id'] = trialwise['session_id'].map(lambda sn: ith_to_dt_session.get(sn[:16]))
    return trialwise.set_index(['session_id', 'trial_id'])


def load_trackwise(session_names, columns=None, choices_from=None):
    """BehaviorTrackwise - one row per (trial, 1cm position bin) - indexed by session_id.

    The position-binned counterpart of load_behavior's time-binned frame: 'from_position_bin'
    is the bin's start in cm along the track (about -169..258 for P1100), and each
    `posbin_*` column is that bin's value of one movement metric, with a `_500msMedian`
    variant already smoothed (see behavior_plotting_lib.KINEMATICS_METRICS). Trial labels
    ('trial_id', 'cue', 'choice_R1', 'choice_R2', 'trial_outcome') are repeated on every row
    of their trial.

    `choices_from`: an already-loaded load_trialwise frame to take 'choice_R1'/'choice_R2'
    from instead of trusting this table's own copies. lm4_plots.py's kinematics cell did
    this because the two disagreed for ~1/3 of trials and trialwise was the internally
    consistent one; re-checked 2026-09-06 on sessions 7/12/15, where the tables now agree
    on every trial (and trackwise's copies are constant within a trial), so this is
    currently a no-op kept as the cheap guard against that regressing - pass the frame the
    notebook already loaded and the figures cannot silently go back to the bad column."""
    trackwise = analytics.get_analytics('BehaviorTrackwise', session_names=session_names,
                                        columns=columns)
    trackwise.index = trackwise.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
    if choices_from is not None:
        for choice_col in ('choice_R1', 'choice_R2'):
            per_trial = choices_from.reset_index().set_index(['session_id', 'trial_id'])[choice_col]
            keys = pd.MultiIndex.from_arrays([trackwise.index.get_level_values('session_id'),
                                              trackwise['trial_id']])
            trackwise[choice_col] = per_trial.reindex(keys).values
    return trackwise


def load_firing_rate(session_names, analytic='FiringRate40msZ', trial_id_from=None):
    """FiringRate40msZ/Hz on the same index shape as load_behavior.

    `trial_id_from`, if given, is a behavior frame whose 'trial_id' column is copied
    across positionally - the two tables share the exact same 40ms ephys-bin grid
    row-for-row (both derive from the same FiringRate40msHz timestamps, see
    get_Behavior40msAligned), which is what makes the positional copy safe; it is checked
    here rather than assumed."""
    fr = analytics.get_analytics(analytic, session_names=session_names)
    fr = fr.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
    if trial_id_from is not None:
        if len(fr) != len(trial_id_from):
            raise ValueError(f"{analytic} has {len(fr)} rows but behavior has {len(trial_id_from)}; "
                             f"the positional trial_id copy is only valid on a shared bin grid")
        fr['trial_id'] = trial_id_from['trial_id'].values
    fr.index = pd.IntervalIndex.from_arrays(
        left=fr.to_ephys_timestamp - BIN_US,
        right=fr.to_ephys_timestamp,
        closed='right')
    fr.set_index('session_id', append=True, inplace=True)
    return fr


def unit_mean_rates(session_names=None, analytic='FiringRate40msHz', fr_hz=None):
    """Per-session mean firing rate (Hz) of every unit column - a DataFrame indexed by
    session_id, columns 'UnitNNNN'.

    Pass `fr_hz` when the caller already holds the raw-Hz table (dim_analysis.py loads it as
    `fr_hz_all` for its pooled-Z alternative): loading it again for a column mean is one of
    the most expensive I/O calls in these notebooks and buys nothing. `session_names` is only
    used to fetch it when `fr_hz` is not given; exactly one of the two is required.

    Exists for the unit-quality floor the dim/PCA figures apply (see
    dim_plotting_lib._apply_unit_rate_floor): those run on FiringRate40msZ, in which a unit
    firing one spike every few minutes is indistinguishable from a well-isolated one until you
    look at the rate it was z-scored against - a 0.004 Hz unit's single spike arrives as z~80
    and can own a whole timebin's covariance spectrum. Measured on animal 6: 5-9 of 57 mPFC
    units per session are below 0.1 Hz, 15-20 below 0.5 Hz.

    Deliberately computed from the *raw Hz* analytic rather than from the z-scored one, where
    every unit's mean is ~0 by construction and this question cannot be asked at all. Rates are
    over the whole session (every 40ms bin, not just the trial windows a figure plots), which is
    the same population the z-scoring itself was estimated over - so the number really does say
    what a unit's z-scores are relative to."""
    if fr_hz is None:
        if session_names is None:
            raise ValueError("pass either fr_hz (an already-loaded raw-Hz table) or "
                             "session_names to load one")
        fr_hz = analytics.get_analytics(analytic, session_names=session_names)
    unit_cols = [c for c in fr_hz.columns if c.startswith('Unit')]
    return fr_hz.groupby(level='session_id')[unit_cols].mean()


def unit_columns(fr, x_modality='HP+mPFC'):
    """The 'UnitNNNN' column names of `fr` belonging to `x_modality`, in table order.

    Single source of truth for the HP/mPFC split (see N_HP_UNITS), replacing three
    separate hardcoded `.iloc[:, 20:]` slices. Non-unit columns (timestamps, trial_id)
    are never included, whatever position they sit at."""
    units = [c for c in fr.columns if c.startswith('Unit')]
    if x_modality == 'HP':
        return units[:N_HP_UNITS]
    if x_modality == 'mPFC':
        return units[N_HP_UNITS:]
    if x_modality == 'HP+mPFC':
        return units
    raise ValueError(f"Unknown x_modality {x_modality!r}; known: {ALL_X_MODALITIES}")


def modality_slice(fr, x_modality='HP+mPFC', fillna=0):
    """`fr` reduced to just `x_modality`'s unit columns (see unit_columns).

    This must stay identical to how `X_full` is built at fit time in
    get_SVMCueOutcomeChoicePred (postproc_mea1k_ephys.py), or a dot product with a fitted
    `w` silently lines up against the wrong features. `fillna=0` matches that same
    construction; pass None to keep NaNs."""
    sliced = fr[unit_columns(fr, x_modality)]
    return sliced.fillna(fillna) if fillna is not None else sliced


def session_fr_for_modality(fr_raw_session, x_modality, fillna=0):
    """One session's raw `get_analytics('FiringRate40msZ', ...)` rows - still carrying
    from_/to_ephys_timestamp columns and no IntervalIndex - reshaped into the
    (IntervalIndex, modality-sliced) frame a `w` dot product needs.

    This was `_x_modality_fr` inline in svm_analysis_2026.py, needed by both the W
    projections and the whole-session PCA. It must keep matching
    get_SVMCueOutcomeChoicePred's own X_full construction (postproc_mea1k_ephys.py) or a
    projection silently lines up against the wrong features - see modality_slice."""
    fr = fr_raw_session.copy()
    fr.index = pd.IntervalIndex.from_arrays(fr.pop('from_ephys_timestamp'), fr.pop('to_ephys_timestamp'),
                                             closed='right')
    return modality_slice(fr, x_modality, fillna=fillna)


def interval_window_bins(t0_events, interval_name):
    """{session_id: (n_bins_left, n_bins_total)} for one interval, measured off the
    actual stored pd.Intervals rather than read from a config - this is what makes
    check_geometry independent of whatever interval_specifier currently says."""
    rows = t0_events[t0_events[interval_name].notna()]
    if len(rows) == 0:
        return {}
    left = ((rows['t0'] - rows[interval_name].array.left) / BIN_US).round().astype(int)
    total = ((rows[interval_name].array.right - rows[interval_name].array.left) / BIN_US).round().astype(int)
    total = pd.Series(total, index=rows.index)
    out = {}
    for s_id, idx in rows.groupby(level='session_id').groups.items():
        out[s_id] = (sorted(set(left.loc[idx])), sorted(set(total.loc[idx])))
    return out


def check_geometry(t0_events, svm_results, raise_on_mismatch=True):
    """Assert that `t0_events` and `svm_results` are the same generation.

    A fit slides a TIMEBINS_PER_FIT-wide window one native bin at a time across its
    interval, so `n_timebins == window_bins - TIMEBINS_PER_FIT + 1`. Inverting that gives
    the window the fit actually saw, which must equal the width of the interval stored in
    the t0 table the plots window their traces with. When the two disagree, every
    per-timebin trace is stretched against the position/velocity traces beside it and the
    t0 markers land in the wrong place - with no error anywhere. Returns a list of
    human-readable mismatch strings (empty when consistent)."""
    problems = []
    for interval_name, rows in svm_results.groupby('interval_name'):
        if interval_name not in t0_events.columns:
            problems.append(f"{interval_name}: fit exists but the t0 table has no such column")
            continue
        fit_window = rows['timebin'].nunique() + TIMEBINS_PER_FIT - 1
        widths = sorted({w for _, totals in interval_window_bins(t0_events, interval_name).items()
                         for w in totals[1]})
        if widths and fit_window not in widths:
            problems.append(f"{interval_name}: fit window is {fit_window} bins "
                            f"(from {rows['timebin'].nunique()} timebins) but the t0 table stores "
                            f"{widths} - traces would be misaligned")
    if problems and raise_on_mismatch:
        raise ValueError("t0_events and svm_results are different generations:\n  " +
                         "\n  ".join(problems) +
                         "\nSee plotlib.session_data.ANALYTIC_PAIRS for which tables belong together.")
    return problems


# --------------------------------------------------------------------------------------
# SVM_TaskSubspaces (ephys_preprocessing/svm_task_subspaces.py), the successor fit table
# --------------------------------------------------------------------------------------
# Its schema is deliberately not the old one: no f1, n_pos/n_neg instead of
# n_positives/n_negatives, no more_reliable, and the null control is a per-row list of
# scores (two of them - perm and shift) instead of a scalar mean. This used to be adapted
# to the old names here, by `to_legacy_svm_schema`; the plotting libs now resolve both
# spellings themselves (see svm_plotting_lib.METRIC_SPECS/_first_present), so a v2 table
# is plotted exactly as it comes off the analytic and nothing has to remember to convert
# it first. The one thing that cannot be aliased away: metric='f1' does not exist on this
# table (the analytic computes balanced accuracy only), so plot with
# metric='balanced_accuracy'.


def load_task_subspaces(session_names):
    """SVM_TaskSubspaces for `session_names`."""
    return analytics.get_analytics('SVM_TaskSubspaces', session_names=session_names)


def compare_svm_tables(new, old, metric_col='acc_aggr', label_new='new', label_old='old',
                       group_cols=('interval_name', 'predict_y_name', 'X_modality')):
    """Per row group, how the same score differs between two fit tables.

    Inner-joins on (group_cols, timebin) restricted to UNCONDITIONED rows of both - a
    conditioned row is a different fit and is not comparable to a naive one - and reports
    how many timebins matched, the mean signed and absolute difference, the correlation of
    the two curves, and each side's own mean. A row group present in only one table shows up
    as absent rather than as a difference, which is the first thing to check when a number
    moves: geometry, target availability and modality naming all differ between generations.
    """
    def _flat(df, tag):
        d = df.reset_index()
        if 'analysis' in d.columns:
            d = d[d['analysis'].isin(['naive', 'baseline_vs_t'])]
        keys = [c for c in (*group_cols, 'session_id', 'timebin') if c in d.columns]
        return d[keys + [metric_col]].rename(columns={metric_col: tag})

    a, b = _flat(new, label_new), _flat(old, label_old)
    keys = [c for c in a.columns if c != label_new]
    merged = a.merge(b, on=keys, how='inner')
    rows = []
    gcols = [c for c in (*group_cols, 'session_id') if c in merged.columns]
    for key_vals, grp in merged.groupby(gcols, dropna=False, sort=False):
        x, y = grp[label_new].to_numpy(float), grp[label_old].to_numpy(float)
        ok = pd.notna(x) & pd.notna(y)
        rec = dict(zip(gcols, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        rec.update(n_timebins=int(ok.sum()),
                   mean_new=round(float(x[ok].mean()), 4) if ok.any() else float('nan'),
                   mean_old=round(float(y[ok].mean()), 4) if ok.any() else float('nan'))
        rec['mean_diff'] = round(rec['mean_new'] - rec['mean_old'], 4)
        rec['mean_abs_diff'] = round(float(abs(x[ok] - y[ok]).mean()), 4) if ok.any() else float('nan')
        rec['corr'] = (round(float(pd.Series(x[ok]).corr(pd.Series(y[ok]))), 3)
                       if ok.sum() > 2 else float('nan'))
        rows.append(rec)
    return pd.DataFrame(rows)


def load_svm_context(paradigm_ids=(1100,), animal_ids=(6,), session_ids=None,
                     svm_analytic=DEFAULT_SVM_ANALYTIC, behavior_columns=None, verify=True):
    """Everything the SVM figure notebooks open with, as one dict:
    session_names, ith_to_dt_session, t0_events, behavior, svm_results, svm_analytic,
    t0_analytic.

    The t0 table is chosen to match `svm_analytic` (see ANALYTIC_PAIRS) instead of being
    picked independently, and `verify` cross-checks the two actually agree before any
    plot is drawn."""
    session_names, ith_to_dt_session = get_session_names(paradigm_ids, animal_ids, session_ids)
    t0_analytic = t0_analytic_for(svm_analytic)
    t0_events = load_t0_events(session_names, analytic=t0_analytic)
    behavior = load_behavior(session_names, columns=behavior_columns)
    svm_results = analytics.get_analytics(svm_analytic, session_names=session_names)
    if verify:
        check_geometry(t0_events, svm_results)
    return dict(session_names=session_names, ith_to_dt_session=ith_to_dt_session,
                t0_events=t0_events, behavior=behavior, svm_results=svm_results,
                svm_analytic=svm_analytic, t0_analytic=t0_analytic)
