"""SVM_TaskSubspaces - per-timebin linear decoding of task variables from population rates.

Successor to `postproc_mea1k_ephys.get_SVMCueOutcomeChoicePred`, written from scratch
rather than edited, and deliberately self-contained: everything this analytic needs is
in this module plus its two input analytics (FiringRate40msZ, TrialWiseT0Events40ms2).
The old function is left untouched so the existing cached tables and figures keep
working.

WHY A REWRITE - the two things that were actually wrong, both measured:

1. The reference the decoding curve was judged against was not valid. The shuffle null
   reused the C that had been grid-searched on the TRUE labels and ran 30 bootstraps
   against the observed fit's 200, so it sat systematically low; it was drawn
   independently per timebin, so nothing accounted for the ~30 correlated timebins a
   curve is read across; and the C grid's floor (0.0005) was selected often enough that
   17-26% of all cached rows are a constant classifier reading exactly 0.5, which is a
   model collapse, not a measurement. Measured consequence: in a session where no cue
   was ever shown, the real `cue` curve (max 0.610 over 23 timebins) is indistinguishable
   from curves fit to fully permuted labels (max 0.561-0.619).

   Fixed here by making the null the same estimator as the observation - C is re-selected
   inside every null pass, the bootstrap count is identical - and by holding each draw's
   relabelling FIXED ACROSS A ROW GROUP: one permuted label vector per draw, applied at
   every timebin, so the stored per-row null lists also give the max-over-timebins
   distribution (see sequence_stats). A collapsed fit is emitted as NaN with
   fit_collapsed=True instead of as 0.5.

2. Slow drift is a large, unmodelled axis. A label that only says "first vs second half
   of the session" decodes at 0.72-0.78 balanced accuracy in EVERY timebin of a real
   session, so any target correlated with session time inherits part of it. Hence two
   things: the `shift` null (circular shift of the trial-label vector, which preserves the
   label sequence's own temporal structure and therefore tests against drift rather than
   destroying it), and `session_half` as a permanent diagnostic target so every figure
   carries the drift axis for scale.

CONDITIONED (CCGP) FITS come out of the same run as the unconditioned one (`ccgp=True`),
off the same X, the same trial set and the same C grid. In the old pipeline they were a
separate analytic computed on a separate run, which is why a naive and a CCGP weight vector
were not comparable: different C selection scope (and therefore, under L1, different
sparsity), and for a while different interval geometry. Here a conditioned direction is one
more entry in the per-interval fit list - an extra sampler and extra rows, not a second code
path - so the two cannot drift apart. `analysis` says which a row is; `conditioner`,
`train_stratum`, `test_stratum` are None on unconditioned rows.

THE MATCHED CONTROL (`analysis='ccgp_control'`, on by default with `match_ccgp_n`) exists
because a conditioned row and a naive row are still not comparable even when they share an
estimator: the conditioned one trains on ~half the trials (biasing it DOWN), tests on a fixed
set that every one of the 60 bootstrap iterations votes on rather than the ~22 a naive row's
out-of-bag rows get (biasing its aggregated score UP), and scores only that half. Two of those
push opposite ways, so the raw naive-minus-ccgp gap is not even conservative. The control is
the same train-one-part/test-the-other machinery run on a RANDOM re-partition of the same
trials carrying the conditioner split's per-class counts, so all three asymmetries match by
construction and the only thing left different from its conditioned twin is that the split is
defined by the conditioner. It is the standard random-dichotomy control, and it is the row a
CCGP score should be read against - not the naive row, which stays in the table as the
full-data reference for absolute decodability. On a control row `conditioner`/`train_stratum`/
`test_stratum` name the split whose marginals were MATCHED, not one the data respects, which
is what makes it join 1:1 to its conditioned twin; `analysis` is what tells them apart.

DERIVED TARGETS (an XOR or a conjunction of two other labels) are registry entries built from
XOR_PAIRS / AND_TRIPLES and carrying `components`, not another elif of inline arithmetic. They
are optional: name them in `y_names`. Conditioning one on either of its own components would
leave the label constant (or collapse it onto the other component) in that stratum, so they
are absent from CONDITIONERS. Note that an XOR and a conjunction answer different questions:
only the XOR is a nonlinearity probe (see _and_of on why a 1-vs-3 dichotomy is not).

TERMS used below, since the table itself has neither: a FIT is one row - one (interval,
timebin, X_modality, predict_y_name, analysis, direction). A ROW GROUP is every row sharing
all of those except timebin, ordered by timebin, which is what one line of one panel is
drawn from.
The distinction matters because the two statistics belong to different levels: the per-row
`p_point_*` is about a single timebin, and `sequence_stats` is about a row group. Reading a
panel means scanning a row group, so that is the statistic to use unless the timebin was
chosen in advance. `sequence_stats` works on any table this module has produced, since it
only needs the stored null lists.
"""
import os
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC, LinearSVC

from CustomLogger import CustomLogger as Logger

# =============================================================================
# Configuration - the whole analytic's behaviour, in one place
# =============================================================================

BIN_LENGTH_US = 40_000
TIMEBINS_PER_FIT = 3        # bins averaged into one sample; also the sliding window width
BASELINE_BINS = 3           # first N bins of a paired interval are its negative class

# Targets fit in an ordinary interval. `random_control` and `session_half` are diagnostics
# and cost one pass each: random_control's observed curve IS a null draw on real data, so
# it shows directly whether the machinery is calibrated; session_half measures the drift
# axis every other curve has to be read against.
TRIAL_LABEL_TARGETS = ('cue', 'choice_R1', 'cueXORchoice_R1',
                       'random_control', 'session_half')

# Interval geometry (window length, t0 offset) is NOT defined here - it is read off the
# t0-events table, which is the one source of truth for it. This dict only says which
# intervals to visit and what to fit in each.
#
# The reward intervals get `baseline_vs_t` and not the trial labels because a *_sound t0
# only exists when that zone's reward actually fired (see get_TrialWiseT0Events40ms):
# reward1_sound implies cue==1 AND choice_R1==stop, so both of those labels are constant
# there by construction. The old code tried to fall back to trial labels and either raised
# or silently produced nothing, depending on the branch.
INTERVALS = {
    # 'track_start_interval':   TRIAL_LABEL_TARGETS,
    'cue_entry_interval':     TRIAL_LABEL_TARGETS,
    'cue_exit_interval':      TRIAL_LABEL_TARGETS,
    'R1_entry_interval':      TRIAL_LABEL_TARGETS,
    'R2_entry_interval':      TRIAL_LABEL_TARGETS,
    # 'reward1_sound_interval': ('baseline_vs_t',),
    # 'reward2_sound_interval': ('baseline_vs_t',),
    'ITI_interval':           TRIAL_LABEL_TARGETS,
}

# Conditioned (CCGP) fits: {target: conditioner}. Each (interval, timebin, modality, target)
# listed here gains two rows, one per train/test direction, from the SAME run as the
# unconditioned row - same X, same trial set, same C grid. That shared selection is what
# makes a conditioned weight vector comparable to an unconditioned one; the old pipeline
# produced them as two separate analytics with different C scopes and different geometry.
# Roughly triples the fit count for the targets listed.
#
# A derived target (a conjunction or an XOR of two others) is deliberately absent: it has no
# independent conditioner, since conditioning on one of its own components leaves the label
# constant in one stratum, and _fit_point's guard drops that direction rather than fitting it.
CONDITIONERS = {'choice_R1': 'cue', 'cue': 'choice_R1'}

SVM_PENALTY = 'l2'
# These four are the whole cost of a run, and the cost is dominated by the null: every draw
# refits the entire row group with the same estimator as the observation, so wall time scales
# as (1 + n_draws x n_null_kinds) x n_rows x (C grid + bootstrap). Defaults are set for a run
# you would actually wait for - measured throughput is ~15.7k SVM fits/s on 32 cores, and
# get_SVM_TaskSubspaces logs its projected fit count and ETA before dispatching, so a bad
# combination is visible in the first second rather than an hour in.
N_BOOTSTRAP = 60            # each trial still lands out-of-bag ~22 times, enough for a
                            # stable majority vote; only the 10-90 CI gets coarser
# DO NOT go below ~20: with N draws the smallest attainable p is 1/(N+1), so N=20 gives
# 0.048 and N=10 gives 0.09 - a run with 10 draws cannot reject anything at 0.05 no matter
# how strong the effect. This is the one cost knob that buys nothing when turned down.
N_PERMUTATIONS = 20
# Log-spaced over the range that is actually used: in the old cached tables nothing below
# 0.1 was ever selected except the collapsed floor (C=0.0005, 16% of rows) which is now
# excluded from selection anyway, so the low decades were pure cost. Selection re-runs
# inside every null pass, so each grid point costs n_cv_folds fits on every pass of every row.
C_GRID = np.logspace(-1, 2, 6)
N_CV_FOLDS = 5
# Only the iid permutation by default. The circular shift is the one that separates a result
# from slow drift, but it doubles the run and it is near-redundant for a label with no time
# structure of its own - cue is a per-trial coin flip. Turn it on for the labels where it
# earns its cost (stopping behaviour) with SVMTASK_NULLS=perm,shift.
DEFAULT_NULLS = ('perm',)
MIN_TRIALS = 10
MIN_PER_CLASS = 3
RANDOM_CONTROL_SEED = 20260813
NULL_SEED_BASE = 5_000_000
MATCHED_SPLIT_SEED_BASE = 7_000_000
# Measured on 32 cores, ~120 trials x ~77 units. Only used for the projected ETA logged
# before dispatch; the run reports its actual rate against it at the end, so a machine or a
# session that fits a different profile shows up as a discrepancy rather than a silent lie.
ETA_FITS_PER_S = 13_500


def env_overrides():
    """Dev overrides for iterating without editing this file or analytics.py.

    SVMTASK_INTERVALS=cue_entry_interval,R1_entry_interval   (subset of INTERVALS)
    SVMTASK_TARGETS=cue,choice_R1,cueXORchoice_R1            (any names from TARGETS)
    SVMTASK_N_PERM=5 / SVMTASK_N_BOOTSTRAP=50               (cheap smoke runs)
    SVMTASK_NULLS=perm                                      (drop the shift null)
    SVMTASK_CCGP=0                                          (skip the conditioned fits)
    SVMTASK_MATCH_N=0                                       (skip their matched controls)

    A full run is what happens with none of them set. The older SVMTS_ prefix is still read
    so a command line written before the rename keeps working.
    """
    def _get(name):
        return os.environ.get(f'SVMTASK_{name}') or os.environ.get(f'SVMTS_{name}')

    kwargs = {}
    if _get('INTERVALS'):
        kwargs['which_intervals'] = _get('INTERVALS').split(',')
    if _get('TARGETS'):
        kwargs['y_names'] = _get('TARGETS').split(',')
    if _get('N_PERM'):
        kwargs['n_permutations'] = int(_get('N_PERM'))
    if _get('N_BOOTSTRAP'):
        kwargs['n_bootstrap'] = int(_get('N_BOOTSTRAP'))
    if _get('NULLS'):
        kwargs['null_names'] = _get('NULLS').split(',')
    if _get('CCGP') is not None:
        kwargs['ccgp'] = _get('CCGP') not in ('0', 'false', 'False', '')
    if _get('MATCH_N') is not None:
        kwargs['match_ccgp_n'] = _get('MATCH_N') not in ('0', 'false', 'False', '')
    return kwargs


# =============================================================================
# Target registry - one entry per decodable label
# =============================================================================
# Each `fn` takes the interval's per-trial table (already restricted to the trials this
# interval can actually be fit on, sorted by trial_id) and returns an int label per trial.
# `kind` drives which null kinds apply and how the design is built, so adding a target does
# not touch anything below this section.

def _y_cue(trials):
    # cue1 -> 0, cue2 -> 1. The clip is for the single historical trial where cue was 0.
    return np.clip(trials['cue'].values - 1, 0, 1).astype(int)


def _y_choice_R1(trials):
    # native boolean, 1 = stop / 0 = skip
    return trials['choice_R1'].values.astype(bool).astype(int)


def _y_random_control(trials):
    # fixed-seed balanced random label: a negative control whose OBSERVED curve is itself
    # a null draw on this session's real population activity
    n = len(trials)
    Y = np.zeros(n, dtype=int)
    Y[np.random.default_rng(RANDOM_CONTROL_SEED).permutation(n)[:n // 2]] = 1
    return Y


def _y_session_half(trials):
    # the drift axis, in trial order: not a task variable, a scale for the others
    n = len(trials)
    return (np.arange(n) > (n - 1) / 2).astype(int)


def _y_choice_R2(trials):
    # native boolean, 1 = stop / 0 = skip. Registered mainly so an XOR pair can name it.
    return trials['choice_R2'].values.astype(bool).astype(int)


def _xor_of(a_name, b_name):
    """Y = 1 where the two labels disagree, 0 where they agree.

    With cue coded cue1=0/cue2=1 and choice coded skip=0/stop=1, disagreement is
    (cue1 & stop) or (cue2 & skip) - so for the (cue, choice_R1) pair the positive class is
    the R1-correct behaviour and the negative class is R1-incorrect. The polarity inverts for
    (cue, choice_R2), where R2 is the rewarded zone under cue2: there the AGREEING
    combinations (cue1 & skip, cue2 & stop) are the correct ones. The label itself is the
    plain XOR either way; only the reading of which side means "correct" changes.
    """
    def fn(trials):
        a = np.asarray(TARGETS[a_name]['fn'](trials), dtype=int)
        b = np.asarray(TARGETS[b_name]['fn'](trials), dtype=int)
        return (a != b).astype(int)
    return fn


TARGETS = {
    'cue':            dict(fn=_y_cue,            kind='trial_label', needs=('cue',),
                           levels=(0, 1), level_labels=('cue1', 'cue2')),
    'choice_R1':      dict(fn=_y_choice_R1,      kind='trial_label', needs=('choice_R1',),
                           levels=(0, 1), level_labels=('skip', 'stop')),
    'random_control': dict(fn=_y_random_control, kind='trial_label', needs=(),
                           levels=(0, 1), level_labels=('ctrl0', 'ctrl1')),
    'session_half':   dict(fn=_y_session_half,   kind='trial_label', needs=(),
                           levels=(0, 1), level_labels=('early', 'late')),
    'choice_R2':      dict(fn=_y_choice_R2,      kind='trial_label', needs=('choice_R2',),
                           levels=(0, 1), level_labels=('skip', 'stop')),
    # paired design: Y is not a trial label but which window a sample came from, so `fn`
    # is unused and the design builder handles it (see _paired_design)
    'baseline_vs_t':  dict(fn=None,              kind='paired',      needs=(),
                           levels=(0, 1), level_labels=('baseline', 'timebin')),
}

# XOR targets, registered from pairs of existing ones. Optional - not in
# TRIAL_LABEL_TARGETS, request by name (e.g. y_names=['cue', 'choice_R1', 'cueXORchoice_R1']).
# `components` records which two labels a derived target is built from, so a consumer does
# not have to infer non-independence from the name. Add a pair here to get another one.
XOR_PAIRS = (('cue', 'choice_R1'), ('cue', 'choice_R2'))

for _a, _b in XOR_PAIRS:
    TARGETS[f'{_a}XOR{_b}'] = dict(
        fn=_xor_of(_a, _b), kind='trial_label', components=(_a, _b),
        needs=tuple(dict.fromkeys(TARGETS[_a]['needs'] + TARGETS[_b]['needs'])),
        levels=(0, 1), level_labels=(f'{_a}=={_b}', f'{_a}!={_b}'))


# Conjunction targets: Y=1 on exactly ONE cell of the (a, b) grid, 0 on the other three.
# Registered from explicit LEVELS and not from the labels' 1-values, because the positive cell
# is not (1, 1): cue is coded cue1=0/cue2=1 here, so the R1-rewarded cell is (cue1, stop) -
# the same label the old `_process_y_values` built with a flip-then-check-zero. The cue level
# differs between the two entries because R1 is the rewarded zone under cue1 and R2 under cue2.
AND_TRIPLES = (('cueANDstopR1', ('cue', 0), ('choice_R1', 1)),
               ('cueANDstopR2', ('cue', 1), ('choice_R2', 1)))


def _and_of(a_name, a_level, b_name, b_level):
    """Y = 1 on one cell of the two labels' 2x2, 0 on the other three.

    WHAT THIS IS AND IS NOT. A 1-vs-3 dichotomy is linearly separable under a purely additive
    code - a corner of a rectangle can be cut off by a line - so unlike the XOR next to it in
    every figure, a high score here is NOT evidence of conjunctive coding: it follows from cue
    and choice each being decodable. What it is, is the "reward fires in this zone on this
    trial" readout, the condition whose occurrence is what makes a *_sound t0 exist at all.

    It is also nearly collinear with the targets already fit: cueANDstopR1 == cueXORchoice_R1
    AND cue1, i.e. the rewarded subset of the R1-correct trials, differing from the XOR only
    on cue2 trials. Its incremental content over {cue, choice_R1, cueXORchoice_R1} is the
    cue-gating.
    """
    def fn(trials):
        a = np.asarray(TARGETS[a_name]['fn'](trials), dtype=int)
        b = np.asarray(TARGETS[b_name]['fn'](trials), dtype=int)
        return ((a == a_level) & (b == b_level)).astype(int)
    return fn


for _name, (_a, _al), (_b, _bl) in AND_TRIPLES:
    _pos = f"{TARGETS[_a]['level_labels'][_al]}&{TARGETS[_b]['level_labels'][_bl]}"
    TARGETS[_name] = dict(
        fn=_and_of(_a, _al, _b, _bl), kind='trial_label', components=(_a, _b),
        needs=tuple(dict.fromkeys(TARGETS[_a]['needs'] + TARGETS[_b]['needs'])),
        levels=(0, 1), level_labels=(f'not({_pos})', _pos))


# =============================================================================
# Nulls - curve-coherent by construction
# =============================================================================
# A draw's label vector depends on (target, kind, draw index) and NOT on timebin or
# modality, so permutation d is the same relabelling everywhere along one curve. That is
# what makes the stored per-row lists usable as a max-over-timebins null downstream, at no
# extra cost over the per-timebin null the old code drew independently per fit.

# `groups` is the list of index sets each relabelling is confined to: one set covering all
# rows for a plain trial label, and [train_stratum, test_stratum] for a conditioned (CCGP)
# fit, where relabelling across the whole vector would also destroy the conditioner split
# the fit is defined by.

def _null_perm(rng, Y, groups):
    out = Y.copy()
    for g in groups:
        out[g] = rng.permutation(Y[g])
    return out


def _null_shift(rng, Y, groups):
    # circular shift keeps the label sequence's own temporal structure (its autocorrelation
    # and its correlation with session time) while breaking its tie to the neural data. A
    # label with no time structure of its own (cue is a per-trial coin flip: measured
    # |corr(cue, trial_index)| <= 0.16 in the early sessions) gets essentially the same
    # distribution from this as from _null_perm - it earns its place on labels that do drift
    # within a session, e.g. stopping behaviour.
    out = Y.copy()
    for g in groups:
        if len(g) > 1:
            out[g] = np.roll(Y[g], int(rng.integers(1, len(g))))
    return out


def _null_paired_swap(rng, Y, groups):
    # the paired-design analogue: independently swap each trial's two samples. Permuting
    # all 2n rows would also destroy the pairing, which is not the null being tested.
    # `groups` is unused: the pairing, not an index set, is what constrains this one.
    n = len(Y) // 2
    flip = rng.random(n) < 0.5
    return np.concatenate([flip.astype(int), (~flip).astype(int)])


NULL_KINDS = {
    'trial_label': {'perm': _null_perm, 'shift': _null_shift},
    'paired':      {'perm': _null_paired_swap},
    'ccgp':        {'perm': _null_perm, 'shift': _null_shift},
}
ALL_NULL_NAMES = ('perm', 'shift')


def _build_nulls(kind, Y, groups, null_names, n_draws):
    """{null_name: [label vector per draw]}, fixed across timebins and modalities.

    Built once per (interval, fit) so draw d is the same relabelling at every timebin of
    that row group - which is what makes the stored per-row lists combinable into a
    max-over-timebins distribution (see sequence_stats).
    """
    out = {}
    for null_name, fn in NULL_KINDS[kind].items():
        if null_name not in null_names:
            continue
        out[null_name] = [
            fn(np.random.default_rng(NULL_SEED_BASE
                                     + 1000 * ALL_NULL_NAMES.index(null_name) + d),
               Y, groups)
            for d in range(n_draws)]
    return out


# =============================================================================
# Design - per-trial windows, from the t0 table's own geometry
# =============================================================================

def _interval_geometry(t0_rows, interval_name):
    """(n_bins_left, n_bins_right, window_bins) as the t0-events table defines them."""
    bins = np.asarray(t0_rows[interval_name + '_bins'].iloc[0])
    return int(-bins[0]), int(bins[-1] + 1), int(len(bins))


def _bin_index_matrix(t0_rows, from_ts, interval_name, window_bins):
    """Integer bin indices of every trial's interval window: (n_trials, window_bins).

    Replaces the old per-trial/per-timebin `X_index.overlaps(...)` mask построение with an
    O(1) lookup. The two are equivalent because the rate bins are contiguous 40ms bins and
    every interval edge is itself a bin's from_ephys_timestamp (the t0 producer takes t0
    from that index), so the window that a right-closed 3-bin interval overlaps is exactly
    the 3 bins starting at the bin whose left edge equals the interval's left edge.
    Trials whose window is not fully covered by contiguous bins (recording edge or a gap)
    are reported as invalid rather than tripping an assert that kills the whole session.
    """
    lefts = np.array([iv.left for iv in t0_rows[interval_name].values], dtype=np.int64)
    i0 = np.searchsorted(from_ts, lefts, side='left')
    valid = i0 < len(from_ts) - window_bins + 1
    valid[valid] &= from_ts[i0[valid]] == lefts[valid]
    # contiguity: the last bin of the window must sit exactly window_bins-1 bins later
    expected_last = lefts + (window_bins - 1) * BIN_LENGTH_US
    idx_last = np.clip(i0 + window_bins - 1, 0, len(from_ts) - 1)
    valid[valid] &= from_ts[idx_last[valid]] == expected_last[valid]

    bin_idx = i0[:, None] + np.arange(window_bins)[None, :]
    return bin_idx, valid


def _window_X(unit_mat, bin_idx, start_bin, n_bins=TIMEBINS_PER_FIT):
    """(n_trials, n_units): each trial's mean rate over n_bins starting at start_bin."""
    return unit_mat[bin_idx[:, start_bin:start_bin + n_bins], :].mean(axis=1)


def _sweep_positions(window_bins, has_baseline):
    """Sliding-window start bins, as 0-based indices into the interval's own bins.

    `timebin` in the output is position+1, i.e. 1-based and centred on the window's middle
    bin, the same convention the old table used. For a paired interval the sweep starts
    after the baseline window, so those rows simply have no timebin 1..BASELINE_BINS - the
    axis stays comparable with the non-paired intervals rather than being renumbered.
    """
    first = BASELINE_BINS if has_baseline else 0
    return list(range(first, window_bins - TIMEBINS_PER_FIT + 1))


def _split_seed(*parts):
    """Stable seed from names, so a control's partition is reproducible across runs.

    Not `hash()`: that is salted per process, so the partition would change between runs of
    the same session and the table would stop being reproducible.
    """
    return MATCHED_SPLIT_SEED_BASE + zlib.crc32('/'.join(map(str, parts)).encode())


def _matched_split(seed, Y, n_train_pos, n_train_neg):
    """Random re-partition of the same trials into two parts with GIVEN class counts.

    The control partition for a conditioned fit (see the module docstring): same trials, same
    two-part structure, same per-class size in each part, but drawn at random instead of being
    the two levels of a conditioner. Because the conditioner's two strata partition all the
    trials, taking the remainder as the second part makes the two exactly comparable - same
    n_train, n_test, n_train_pos, ... down to every count the row records.

    Matching the CLASS counts and not merely the sizes is the part that matters: a conditioner
    stratum is usually imbalanced (under cue1 the animal mostly stops), and a control that
    kept balanced classes would still be the easier fit and would leave the comparison tilted
    the same way, just less.
    """
    rng = np.random.default_rng(seed)
    pos = rng.permutation(np.where(Y == 1)[0])
    neg = rng.permutation(np.where(Y == 0)[0])
    return (np.sort(np.concatenate([pos[:n_train_pos], neg[:n_train_neg]])),
            np.sort(np.concatenate([pos[n_train_pos:], neg[n_train_neg:]])))


def _stratum_extra(conditioner, train_stratum, test_stratum, Y, tr, te):
    """The per-row columns describing a two-stratum fit's train/test sets.

    Shared by the conditioned rows and their matched controls precisely so that a control's
    counts coming out identical to its twin's is a check the matching worked, visible in the
    table rather than only in this module.
    """
    return dict(conditioner=conditioner, train_stratum=train_stratum,
                test_stratum=test_stratum, n_train=len(tr), n_test=len(te),
                n_train_pos=int(Y[tr].sum()), n_train_neg=int(len(tr) - Y[tr].sum()),
                n_test_pos=int(Y[te].sum()), n_test_neg=int(len(te) - Y[te].sum()))


# =============================================================================
# Estimator
# =============================================================================

def _make_pipe(C, penalty):
    # Normalizer(norm='l2') is per-sample and is kept from the old estimator so this
    # table stays comparable with the cached ones. It discards each trial's overall
    # population-rate magnitude, which is a real modelling choice and is recorded on
    # every row (`normalizer`) so it can be A/B'd once the null is trusted.
    svc = (LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=C, max_iter=5000)
           if penalty == 'l1' else SVC(kernel='linear', gamma='scale', C=C))
    return Pipeline([('norm', Normalizer(norm='l2')), ('svc', svc)])


def _select_C(X, Y, c_grid, n_folds, penalty):
    """(best_C, cv_acc, n_collapsed) by CV balanced accuracy over `c_grid`.

    n_collapsed is how many of the grid's candidates collapsed (below), reported per row as
    `n_C_collapsed`. It is a diagnostic on the grid, not on the data: a high count means most
    of the grid is unusable here and the winner was chosen from few candidates.

    A candidate whose pooled held-out predictions use only one class has collapsed to the
    majority classifier; it scores 0.5 and is excluded from the argmax rather than being
    allowed to win it. The old grid's floor did win regularly, which is why 17-26% of rows
    in the cached tables read exactly 0.5. If every candidate collapses, best_C is None and
    the caller emits NaN with fit_collapsed=True - a hole in the curve, not a chance-level
    measurement.
    """
    n_splits = min(n_folds, int(pd.Series(Y).value_counts().min()))
    folds = list(StratifiedKFold(n_splits=n_splits).split(X, Y))
    best_C, best_score, n_collapsed = None, -np.inf, 0
    for C in c_grid:
        scores, pooled = [], []
        for train_idx, val_idx in folds:
            pred = _make_pipe(C, penalty).fit(X[train_idx], Y[train_idx]).predict(X[val_idx])
            scores.append(balanced_accuracy_score(Y[val_idx], pred))
            pooled.append(pred)
        if len(np.unique(np.concatenate(pooled))) < 2:
            n_collapsed += 1
            continue
        score = float(np.mean(scores))
        if score > best_score:
            best_C, best_score = float(C), score
    return best_C, (best_score if best_C is not None else np.nan), n_collapsed


def _sampler(n_rows, n_pairs, train_pool=None, test_idx=None):
    """(train_idx, test_idx) draw. n_pairs set -> resample TRIALS, not rows.

    A paired design's two rows per trial are one trial's baseline and timebin sample; they
    are not independent, so resampling rows would make the resampling unit the sample and
    the bootstrap spread would come out too narrow.

    train_pool set -> conditioned (CCGP) fit: resample within the training stratum and hold
    the whole opposite stratum out, unchanged on every iteration. Every iteration therefore
    votes on the same rows, and the rows in the training stratum are never predicted at all
    (they stay NaN in `predictions` - the same property the old CCGP rows had).
    """
    if train_pool is not None:
        def draw(rng):
            return rng.choice(train_pool, size=len(train_pool), replace=True), test_idx
    elif n_pairs is None:
        def draw(rng):
            train_idx = rng.choice(n_rows, size=n_rows, replace=True)
            oob = np.ones(n_rows, dtype=bool)
            oob[train_idx] = False
            return train_idx, np.where(oob)[0]
    else:
        def draw(rng):
            boot = rng.choice(n_pairs, size=n_pairs, replace=True)
            oob = np.setdiff1d(np.arange(n_pairs), boot)
            return (np.concatenate([boot, boot + n_pairs]),
                    np.concatenate([oob, oob + n_pairs]))
    return draw


def _bootstrap(X, Y, C, n_iterations, n_pairs, penalty, train_pool=None, test_idx=None,
               seed_offset=0):
    """Bagged out-of-bag decoding: per-iteration scores plus one aggregated prediction."""
    draw = _sampler(len(X), n_pairs, train_pool, test_idx)
    accs, ws = [], []
    pred_sum = np.zeros(len(X))
    pred_cnt = np.zeros(len(X))
    non_converged = 0

    for it in range(n_iterations):
        rng = np.random.default_rng(seed_offset + it)
        train_idx, test_idx = draw(rng)
        if len(np.unique(Y[train_idx])) < 2:
            continue
        if len(test_idx) < 4 or len(np.unique(Y[test_idx])) < 2:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', category=ConvergenceWarning)
            pipe = _make_pipe(C, penalty).fit(X[train_idx], Y[train_idx])
        non_converged += any(issubclass(c.category, ConvergenceWarning) for c in caught)
        y_pred = pipe.predict(X[test_idx])
        accs.append(balanced_accuracy_score(Y[test_idx], y_pred))
        ws.append(pipe.named_steps['svc'].coef_.ravel())
        pred_sum[test_idx] += y_pred
        pred_cnt[test_idx] += 1

    if not accs:
        return None

    # majority vote over the iterations that held each row out. Equivalent to the old
    # DataFrame.mode(axis=0) for binary labels and ~100x cheaper - mode was the single
    # slowest step per fit. Rows never held out stay NaN and are excluded from scoring.
    evaluated = pred_cnt > 0
    pred = np.full(len(X), np.nan)
    pred[evaluated] = (pred_sum[evaluated] / pred_cnt[evaluated] > 0.5).astype(float)

    return dict(
        acc_aggr=float(balanced_accuracy_score(Y[evaluated], pred[evaluated])),
        acc_mean=float(np.mean(accs)),
        acc_ci_lower=float(np.percentile(accs, 10)),
        acc_ci_upper=float(np.percentile(accs, 90)),
        w=np.mean(np.vstack(ws), axis=0),
        predictions=pred,
        n_success_iterations=len(accs),
        n_non_converged=non_converged,
    )


def _fit_point(X, Y, cfg, n_pairs, train_pool=None, test_idx=None, seed_offset=0):
    """One (C selection + bootstrap) fit. None if the labels cannot be fit at all.

    For a conditioned fit the guard applies to the TRAINING stratum's labels - that is the
    set the classifier is actually fit on, and a stratum too small to fit would otherwise
    pass a check made on the pooled vector. C selection stays pooled across both strata,
    because StratifiedKFold cannot run on a stratum with a handful of minority-class trials;
    both directions of a pair therefore land on the same regularization, which is what makes
    their weight vectors comparable to each other and to the naive fit's.
    """
    guard_Y = Y if train_pool is None else Y[train_pool]
    counts = pd.Series(guard_Y).value_counts()
    if (len(guard_Y) < cfg['min_trials'] or len(counts) < 2
            or counts.min() < cfg['min_per_class']):
        return None
    best_C, cv_acc, n_collapsed = _select_C(X, Y, cfg['c_grid'], cfg['n_cv_folds'],
                                             cfg['svm_penalty'])
    if best_C is None:
        return dict(fit_collapsed=True, best_C=np.nan, cv_acc=np.nan, n_C_collapsed=n_collapsed)
    res = _bootstrap(X, Y, best_C, cfg['n_bootstrap'], n_pairs, cfg['svm_penalty'],
                     train_pool=train_pool, test_idx=test_idx, seed_offset=seed_offset)
    if res is None:
        return dict(fit_collapsed=True, best_C=best_C, cv_acc=cv_acc, n_C_collapsed=n_collapsed)
    res.update(fit_collapsed=False, best_C=best_C, cv_acc=cv_acc, n_C_collapsed=n_collapsed)
    return res


# =============================================================================
# One fit, observed + its nulls (this is the unit of parallel work)
# =============================================================================

def _run_point(X, Y, null_Ys, cfg, n_pairs, stamp, train_pool=None, test_idx=None):
    """Fit one row (observed + every null draw). One unit of parallel work."""
    t_row = time.perf_counter()
    observed = _fit_point(X, Y, cfg, n_pairs, train_pool, test_idx)
    if observed is None:
        return None

    row = dict(stamp)
    row.update(
        n_trials=int(len(Y) if n_pairs is None else n_pairs),
        n_pos=int(np.sum(Y)), n_neg=int(len(Y) - np.sum(Y)),
        y_true=np.asarray(Y, dtype=np.int8).tolist(),
        fit_collapsed=bool(observed['fit_collapsed']),
        best_C=observed['best_C'], cv_acc=observed['cv_acc'],
        n_C_collapsed=int(observed['n_C_collapsed']),
    )
    if observed['fit_collapsed']:
        row.update(acc_aggr=np.nan, acc_mean=np.nan, acc_ci_lower=np.nan,
                   acc_ci_upper=np.nan, w=[], predictions=[],
                   n_success_iterations=0, n_non_converged=0)
    else:
        row.update(acc_aggr=observed['acc_aggr'], acc_mean=observed['acc_mean'],
                   acc_ci_lower=observed['acc_ci_lower'],
                   acc_ci_upper=observed['acc_ci_upper'],
                   w=observed['w'].tolist(),
                   predictions=observed['predictions'].tolist(),
                   n_success_iterations=int(observed['n_success_iterations']),
                   n_non_converged=int(observed['n_non_converged']))

    # the null passes: same estimator, same bootstrap count, C re-selected each time
    for null_name in ALL_NULL_NAMES:
        accs = []
        for Y_null in null_Ys.get(null_name, ()):
            res = _fit_point(X, Y_null, cfg, n_pairs, train_pool, test_idx)
            accs.append(np.nan if res is None or res['fit_collapsed']
                        else res['acc_aggr'])
        row[f'null_{null_name}_acc'] = [float(a) for a in accs]
        real = row['acc_aggr']
        finite = np.array([a for a in accs if np.isfinite(a)], dtype=float)
        if finite.size and np.isfinite(real):
            row[f'p_point_{null_name}'] = float((np.sum(finite >= real) + 1) / (finite.size + 1))
        else:
            row[f'p_point_{null_name}'] = np.nan
    # CPU seconds for this row, observed pass + all its null passes. Summed per group at the
    # end of the run to show where the time actually went (see the Timing lines) - with one
    # flat dispatch there is otherwise no way to tell.
    row['fit_seconds'] = round(time.perf_counter() - t_row, 4)
    return row


# =============================================================================
# Entry point
# =============================================================================

def get_SVM_TaskSubspaces(fr_z, t0_events, which_intervals=None, y_names=None,
                          svm_penalty=SVM_PENALTY, n_bootstrap=N_BOOTSTRAP,
                          n_permutations=N_PERMUTATIONS, c_grid=None,
                          n_cv_folds=N_CV_FOLDS, null_names=DEFAULT_NULLS, ccgp=True,
                          match_ccgp_n=True):
    """One row per (interval, timebin, X_modality, predict_y_name, analysis, direction).

    which_intervals / y_names narrow INTERVALS and the trial-label targets, which is how
    a fast iteration run is done (see env_overrides).

    ccgp=True adds, for every target in CONDITIONERS, two conditioned rows per point (train
    on one level of the conditioner, test on the other, and the reverse) next to the
    unconditioned row - same run, same trial set, same X, same C grid. Roughly triples the
    fit count for those targets.

    match_ccgp_n=True adds one `ccgp_control` row per conditioned row: the identical fit on a
    random re-partition of the same trials with the conditioner split's class counts, which is
    what a CCGP score should actually be read against (see the module docstring). Ignored when
    ccgp=False, since a control is defined relative to a conditioned row. Costs roughly another
    +50% wall time on the targets in CONDITIONERS - a control row is cheaper than a naive one
    (it trains on half the trials) but there are two of them per point.
    """
    L = Logger()
    cfg = dict(n_bootstrap=n_bootstrap, c_grid=C_GRID if c_grid is None else np.asarray(c_grid),
               n_cv_folds=n_cv_folds, min_trials=MIN_TRIALS, min_per_class=MIN_PER_CLASS,
               svm_penalty=svm_penalty)

    from_ts = fr_z['from_ephys_timestamp'].values.astype(np.int64)
    units = fr_z.drop(columns=['from_ephys_timestamp', 'to_ephys_timestamp'])
    # modality definitions kept exactly as the old analytic had them
    X_full = {'HP+mPFC': units.fillna(0).values,
              'mPFC': units.iloc[:, 20:].fillna(0).values}
    L.logger.info(f"{units.shape[1]} units, {len(from_ts)} bins; modalities: "
                  f"{ {k: v.shape[1] for k, v in X_full.items()} }")

    intervals = list(which_intervals) if which_intervals else list(INTERVALS)
    unknown = [i for i in intervals if i not in INTERVALS]
    if unknown:
        raise ValueError(f"unknown interval(s) {unknown}; known: {sorted(INTERVALS)}")
    if y_names is not None:
        unknown = [y for y in y_names if y not in TARGETS]
        if unknown:
            raise ValueError(f"unknown target(s) {unknown}; known: {sorted(TARGETS)}")

    tasks = []
    t_build = time.perf_counter()
    interval_task_counts = {}
    for interval_name in intervals:
        if interval_name not in t0_events.columns:
            L.logger.warning(f"{interval_name} absent from t0_events, skipping")
            continue
        t0_rows = t0_events[t0_events[interval_name].notna()].sort_values('trial_id')
        if t0_rows.empty:
            L.logger.warning(f"{interval_name}: no trials with this t0 event, skipping")
            continue

        n_left, n_right, window_bins = _interval_geometry(t0_rows, interval_name)
        bin_idx, valid = _bin_index_matrix(t0_rows, from_ts, interval_name, window_bins)
        n_dropped = int((~valid).sum())
        if valid.sum() < MIN_TRIALS:
            L.logger.warning(f"{interval_name}: only {valid.sum()} usable trials, skipping")
            continue
        trials = t0_rows[valid]
        bin_idx = bin_idx[valid]

        # An interval declares which kinds it supports. y_names replaces the trial-label
        # selection outright rather than filtering the declared tuple, so a target that is
        # not in TRIAL_LABEL_TARGETS (an XOR pair, a conjunction) can be requested. A
        # paired-only interval never gets trial labels regardless of y_names - its labels
        # are constant there. Fitting a target only in SOME intervals is done by narrowing
        # which_intervals, not here: the target list is the same in every visited interval.
        declared = INTERVALS[interval_name]
        paired_targets = [y for y in declared if TARGETS[y]['kind'] == 'paired']
        label_targets = [y for y in declared if TARGETS[y]['kind'] != 'paired']
        if y_names is not None and label_targets:
            label_targets = [y for y in y_names if TARGETS[y]['kind'] != 'paired']
        targets = label_targets + paired_targets
        has_baseline = bool(paired_targets)
        positions = _sweep_positions(window_bins, has_baseline)
        L.logger.info(f"{interval_name}: {window_bins} bins (t0 at {n_left}), "
                      f"{len(positions)} timebins, {len(trials)} trials "
                      f"({n_dropped} dropped), targets {targets}")

        # One fit spec per row-to-be, built once per interval: the label vector, its null
        # draws and (for a conditioned fit) its train/test strata. Built here rather than
        # per timebin so a draw is the same relabelling at every timebin and modality - the
        # coherence sequence_stats depends on. A conditioned direction is another entry in
        # this list, not another branch further down.
        fit_specs = []
        for y_name in targets:
            spec = TARGETS[y_name]
            missing = [c for c in spec['needs'] if c not in trials.columns]
            if missing:
                L.logger.warning(f"{interval_name}/{y_name}: missing {missing}, skipping")
                continue
            if spec['kind'] == 'paired':
                Y = np.concatenate([np.zeros(len(trials), int), np.ones(len(trials), int)])
            else:
                Y = np.asarray(spec['fn'](trials), dtype=int)
            all_rows = [np.arange(len(Y))]
            # Requesting a subset of null kinds only leaves that null's column empty (see
            # _run_point, which always walks ALL_NULL_NAMES) - the schema does not change
            # with it, and the seeds are indexed off ALL_NULL_NAMES so a draw is the same
            # relabelling whether or not the other kind was asked for.
            fit_specs.append(dict(
                y_name=y_name, Y=Y, analysis='baseline_vs_t' if spec['kind'] == 'paired' else 'naive',
                paired=spec['kind'] == 'paired', train_pool=None, test_idx=None,
                nulls=_build_nulls(spec['kind'], Y, all_rows, null_names, n_permutations),
                extra=dict(conditioner=None, train_stratum=None, test_stratum=None,
                           n_train=None, n_test=None, n_train_pos=None, n_train_neg=None,
                           n_test_pos=None, n_test_neg=None)))

            conditioner = CONDITIONERS.get(y_name) if ccgp else None
            if conditioner is None or spec['kind'] != 'trial_label':
                continue
            missing = [c for c in TARGETS[conditioner]['needs'] if c not in trials.columns]
            if missing:
                L.logger.warning(f"{interval_name}/{y_name}|{conditioner}: conditioner "
                                 f"missing {missing}, no conditioned fits")
                continue
            Y_cond = np.asarray(TARGETS[conditioner]['fn'](trials), dtype=int)
            # One random re-partition per (interval, target, conditioner), matched to stratum
            # 0's class counts and then used for BOTH directions exactly as the conditioner's
            # own partition is - so control direction 0->1 matches conditioned direction 0->1
            # in every count, and likewise for the reverse. Seeded off the names, so it is the
            # same partition at every timebin and modality of the row group: the same
            # coherence the null draws need, and for the same reason (sequence_stats reads a
            # control's row group as one curve too).
            ctrl_parts = None
            if match_ccgp_n:
                s0 = np.where(Y_cond == 0)[0]
                ctrl_parts = _matched_split(
                    _split_seed(interval_name, y_name, conditioner), Y,
                    int(Y[s0].sum()), int(len(s0) - Y[s0].sum()))

            for train_stratum, test_stratum in ((0, 1), (1, 0)):
                tr = np.where(Y_cond == train_stratum)[0]
                te = np.where(Y_cond == test_stratum)[0]
                # same two guards the fit itself applies, checked here so the reason is
                # logged once per interval instead of silently dropping a task
                if len(tr) < MIN_TRIALS or len(np.unique(Y[tr])) < 2:
                    L.logger.warning(f"{interval_name}/{y_name}|{conditioner}="
                                     f"{train_stratum}: train stratum unusable "
                                     f"(n={len(tr)}, {len(np.unique(Y[tr]))} classes)")
                    continue
                if len(te) < 4 or len(np.unique(Y[te])) < 2:
                    L.logger.warning(f"{interval_name}/{y_name}|{conditioner}="
                                     f"{test_stratum}: test stratum unusable "
                                     f"(n={len(te)}, {len(np.unique(Y[te]))} classes)")
                    continue
                fit_specs.append(dict(
                    y_name=y_name, Y=Y, analysis='ccgp', paired=False,
                    train_pool=tr, test_idx=te,
                    nulls=_build_nulls('ccgp', Y, [tr, te], null_names, n_permutations),
                    extra=_stratum_extra(conditioner, train_stratum, test_stratum, Y, tr, te)))

                if ctrl_parts is None:
                    continue
                # The matched control for the direction just appended. It needs no guards of
                # its own: its train and test parts carry the same per-class counts as tr/te
                # by construction, so it passes exactly the checks above and the ones in
                # _fit_point whenever its conditioned twin does - which is also why appending
                # it here, inside the loop, keeps the two in 1:1 correspondence when a
                # direction is dropped.
                c_tr, c_te = ctrl_parts[train_stratum], ctrl_parts[test_stratum]
                fit_specs.append(dict(
                    y_name=y_name, Y=Y, analysis='ccgp_control', paired=False,
                    train_pool=c_tr, test_idx=c_te,
                    nulls=_build_nulls('ccgp', Y, [c_tr, c_te], null_names, n_permutations),
                    extra=_stratum_extra(conditioner, train_stratum, test_stratum,
                                         Y, c_tr, c_te)))

        n_ccgp = sum(1 for f in fit_specs if f['analysis'] == 'ccgp')
        n_ctrl = sum(1 for f in fit_specs if f['analysis'] == 'ccgp_control')
        if n_ccgp:
            L.logger.info(f"{interval_name}: {n_ccgp} conditioned direction(s) + {n_ctrl} "
                          f"matched control(s) alongside "
                          f"{len(fit_specs) - n_ccgp - n_ctrl} unconditioned fit(s)")

        base_X = {m: _window_X(mat, bin_idx, 0, BASELINE_BINS) for m, mat in X_full.items()} \
            if has_baseline else None

        for pos in positions:
            for modality, mat in X_full.items():
                X_win = _window_X(mat, bin_idx, pos)
                for fs in fit_specs:
                    if fs['paired']:
                        X_fit = np.concatenate([base_X[modality], X_win], axis=0)
                        n_pairs = len(trials)
                    else:
                        X_fit, n_pairs = X_win, None
                    stamp = dict(
                        interval_name=interval_name, timebin=pos + 1,
                        n_bins_left=n_left, n_bins_right=n_right,
                        window_bins=window_bins, bin_length_us=BIN_LENGTH_US,
                        X_modality=modality, predict_y_name=fs['y_name'],
                        analysis=fs['analysis'],
                        trial_ids=trials['trial_id'].astype(int).tolist(),
                        n_trials_dropped=n_dropped,
                        svm_penalty=svm_penalty, n_bootstrap=n_bootstrap,
                        n_permutations=n_permutations, normalizer='l2_sample',
                        **fs['extra'])
                    tasks.append((X_fit, fs['Y'], fs['nulls'], n_pairs, stamp,
                                  fs['train_pool'], fs['test_idx']))
        interval_task_counts[interval_name] = (
            len(tasks) - sum(interval_task_counts.values()))

    if not tasks:
        L.logger.warning("No fittable (interval, timebin, modality, target) combinations")
        return None

    # Projected cost, logged before anything is dispatched: passes are exact, the SVM-fit
    # count and the ETA use the C grid + bootstrap size and a measured 15.7k fits/s on 32
    # cores. Fits are ~1ms each and scale with trial count and unit count, so treat the ETA
    # as the right order of magnitude, not a promise.
    fits_per_pass = len(cfg['c_grid']) * n_cv_folds + n_bootstrap
    # exact: each row is one observed pass plus one pass per null vector it was given
    n_passes = sum(1 + sum(len(v) for v in nulls.values()) for _, _, nulls, *_ in tasks)
    n_fits = n_passes * fits_per_pass
    L.logger.info(f"{len(tasks)} rows, {n_passes} passes "
                  f"(1 observed + {n_permutations} x {sorted(null_names)} per row), "
                  f"{fits_per_pass} SVM fits per pass "
                  f"({len(cfg['c_grid'])} C x {n_cv_folds} folds + {n_bootstrap} bootstrap), "
                  f"~{n_fits / 1e6:.1f}M SVM fits total, "
                  f"ETA ~{n_fits / ETA_FITS_PER_S / 60:.0f} min on 32 cores")
    L.logger.info(f"rows per interval: {interval_task_counts}")
    L.logger.info(f"designs built in {time.perf_counter() - t_build:.1f}s")

    t_fit = time.perf_counter()
    with parallel_config(backend='loky', inner_max_num_threads=1):
        rows = Parallel(n_jobs=-1, verbose=5)(
            delayed(_run_point)(X, Y, nulls, cfg, n_pairs, stamp, train_pool, test_idx)
            for X, Y, nulls, n_pairs, stamp, train_pool, test_idx in tasks)
    fit_s = time.perf_counter() - t_fit

    rows = [r for r in rows if r is not None]
    L.logger.info(f"Done: {len(rows)}/{len(tasks)} fits produced a row")
    if not rows:
        return None
    out = pd.DataFrame(rows)
    n_deg = int(out.fit_collapsed.sum())
    L.logger.info(f"{n_deg}/{len(out)} fits collapsed (emitted as NaN, not 0.5)")

    # Where the time went. Wall time is what you wait for; fit_seconds sums CPU time across
    # workers, so its total is ~n_workers x wall when the pool is saturated - that ratio IS
    # the parallel efficiency, and a low one means the tail of long rows is running alone.
    # The per-group breakdown is not guessable in advance: a conditioned direction trains on
    # roughly half the trials so each of its passes is cheaper, while a wide interval
    # contributes more rows, so the biggest consumer is rarely the slowest single row.
    cpu_s = float(out.fit_seconds.sum())
    L.logger.info(
        f"Timing: {fit_s / 60:.1f} min wall fitting, {cpu_s / 60:.0f} min CPU "
        f"({cpu_s / max(fit_s, 1e-9):.1f}x parallel), "
        f"{n_fits / max(fit_s, 1e-9) / 1000:.1f}k SVM fits/s actual vs "
        f"{ETA_FITS_PER_S / 1000:.1f}k assumed")
    for keys in (['interval_name'], ['predict_y_name', 'analysis']):
        grp = (out.groupby(keys).fit_seconds.agg(['sum', 'mean', 'size'])
               .sort_values('sum', ascending=False))
        grp['pct'] = (100 * grp['sum'] / cpu_s).round(1)
        L.logger.info(f"CPU time by {'+'.join(keys)} (sum s / mean s per row / rows / %):\n"
                      + grp.round(3).to_string())
    L.spacer()
    return out


# =============================================================================
# Reading the output: the statistic that belongs to a whole row group
# =============================================================================
# The per-row `p_point_*` answers "is this timebin's score high", which is only the right
# question for a timebin chosen before looking. Scanning a panel for its highest point is a
# different question, and with 23-38 overlapping timebins per group the highest point clears
# a per-timebin 95th percentile 35-50% of the time under the null.
#
# Because every draw's relabelling is held fixed across a group's timebins (see
# _build_nulls), draw d's scores along the group are one complete null panel: taking its
# maximum gives the distribution the observed maximum has to be ranked against. This is a
# reader-side function rather than a stored column so it also applies to tables computed
# before it existed.

# `analysis` is load-bearing and not redundant with `conditioner`: a conditioned row and its
# matched control deliberately carry the SAME conditioner and train_stratum (that is the join
# key between them), so without it the two curves would be pooled into one group here.
SEQUENCE_KEYS = ('session_id', 'interval_name', 'X_modality', 'predict_y_name', 'analysis',
                 'conditioner', 'train_stratum')


def sequence_stats(df, null_names=ALL_NULL_NAMES, alpha=0.05):
    """One row per row group, with its observed maximum, threshold and p.

    Groups on whatever of SEQUENCE_KEYS the frame carries, as columns or index levels, so a
    single-session frame and a multi-session one both work. `thr` is the (1-alpha) quantile
    of the per-draw maxima; `p` is the observed maximum's rank among them.
    """
    df = df.reset_index()
    keys = [k for k in SEQUENCE_KEYS if k in df.columns]
    out = []
    for key_vals, grp in df.groupby(keys, dropna=False, sort=False):
        grp = grp.sort_values('timebin')
        obs = grp.acc_aggr.to_numpy(dtype=float)
        rec = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        # tables written before the rename carry `degenerate` for the same flag
        collapsed_col = 'fit_collapsed' if 'fit_collapsed' in grp else 'degenerate'
        rec.update(n_timebins=len(grp),
                   n_collapsed_timebins=int(grp[collapsed_col].sum()),
                   n_trials=int(grp.n_trials.iloc[0]),
                   obs_max=float(np.nanmax(obs)) if np.isfinite(obs).any() else np.nan)
        # n_trials is the whole trial set on every row kind, including the two-stratum ones
        # that only TRAIN on half of it and only SCORE the other half - so carry n_train and
        # n_test through as well, since they are the numbers that answer whether two groups
        # being compared saw the same amount of data. Absent on tables written before them.
        for col in ('n_train', 'n_test'):
            if col in grp:
                val = grp[col].iloc[0]
                rec[col] = int(val) if pd.notna(val) else np.nan
        for null_name in null_names:
            col = f'null_{null_name}_acc'
            draws = [np.asarray(v, dtype=float) for v in grp[col]] if col in grp else []
            n_draws = min((len(d) for d in draws), default=0)
            if n_draws == 0:
                rec[f'thr_{null_name}'] = np.nan
                rec[f'p_{null_name}'] = np.nan
                rec[f'n_draws_{null_name}'] = 0
                continue
            # (n_timebins, n_draws) -> one maximum per draw
            null_max = np.nanmax(np.vstack([d[:n_draws] for d in draws]), axis=0)
            rec[f'thr_{null_name}'] = float(np.nanquantile(null_max, 1 - alpha))
            rec[f'p_{null_name}'] = float((np.sum(null_max >= rec['obs_max']) + 1)
                                          / (len(null_max) + 1))
            rec[f'n_draws_{null_name}'] = int(n_draws)
        out.append(rec)
    return pd.DataFrame(out)
