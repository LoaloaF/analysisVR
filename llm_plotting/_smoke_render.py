"""Render a fixed set of figures from a given lib directory, for refactor regression tests.

Deliberately self-contained: it does its own data loading with plain `analytics` calls
rather than going through `plotlib.session_data`, so the exact same script can be pointed
at the pre-refactor `scripted_plotting/` (which has no plotlib) and at the refactored
`llm_plotting/`, and the two output sets diffed.

    python _smoke_render.py --libdir ../scripted_plotting --out /tmp/base
    python _smoke_render.py --libdir .                    --out /tmp/stage1
    python _smoke_render.py --compare /tmp/base /tmp/stage1

The figure set is fixed and small (2 sessions) so a run takes well under a minute. Note
that the SVM figures are rendered with the SAME analytic pairing the notebooks used
before the refactor - including the mismatched t0 table - because the point is to detect
*unintended* changes from moving code, not to be correct. Stages that intentionally
change output (the geometry fix) are expected to differ here; that is what
--compare's report is read against.
"""

import argparse
import os
import sys

SESSIONS = ['2024-12-04_18-06', '2024-12-06_16-49']
X_MODALITY = 'mPFC'
Y_NAMES = ['choice_R1']
INTERVALS = ['R1_entry_interval']
SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
T0_FOR_SVM = 'TrialWiseT0Events40ms2'   # matched to SVML1CueOutcomeChoicePred2 (post-refactor)
T0_FOR_DIM = 'TrialWiseT0Events40ms2'   # what dim_analysis loaded pre-refactor


def _bootstrap(libdir):
    sys.path.insert(0, os.path.abspath(libdir))
    sys.path.append('/home/vrmaster/Projects/VirtualReality')
    from baseVR.base_functionality import init_import_paths
    init_import_paths()
    import matplotlib
    matplotlib.use('Agg')
    from CustomLogger import CustomLogger as Logger
    Logger().init_logger(None, None, logging_level="ERROR")


def _load():
    import numpy as np, pandas as pd
    from analytics_processing import analytics
    from analytics_processing.sessions_from_nas_parsing import (
        sessionlist_fullfnames_from_args, fullfnames2snames)

    session_dirs, _ = sessionlist_fullfnames_from_args([1100], [6], None)
    session_names = fullfnames2snames(session_dirs)
    ith_to_dt = {sn[:16]: i for i, sn in enumerate(session_names)}

    def t0(name):
        t = analytics.get_analytics(name, session_names=session_names)
        t.index = t.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
        t.set_index('trial_id', append=True, inplace=True)
        return t

    behavior = analytics.get_analytics('Behavior40msAligned', session_names=session_names)
    behavior = behavior.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
    behavior.index = pd.IntervalIndex.from_arrays(
        left=behavior.to_ephys_timestamp - 40_000, right=behavior.to_ephys_timestamp, closed='right')
    behavior.set_index('session_id', append=True, inplace=True)

    fr_z = analytics.get_analytics('FiringRate40msZ', session_names=session_names)
    fr_z = fr_z.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
    fr_z['trial_id'] = behavior['trial_id'].values
    fr_z.index = pd.IntervalIndex.from_arrays(
        left=fr_z.to_ephys_timestamp - 40_000, right=fr_z.to_ephys_timestamp, closed='right')
    fr_z.set_index('session_id', append=True, inplace=True)
    unit_cols = [c for c in fr_z.columns if c.startswith('Unit')][20:]  # mPFC

    return dict(session_names=session_names, ith_to_dt=ith_to_dt,
                t0_svm=t0(T0_FOR_SVM), t0_dim=t0(T0_FOR_DIM), behavior=behavior,
                svm_results=analytics.get_analytics(SVM_ANALYTIC, session_names=session_names),
                fr_z=fr_z, unit_cols=unit_cols)


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def render(out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    d = _load()
    done, failed = [], []

    def attempt(name, fn):
        try:
            fn()
            done.append(name)
            print(f"  ok    {name}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            plt.close('all')

    import svm_plotting_lib as spl
    import svm_heatmap_plotting_lib as shl
    import dim_plotting_lib as dpl

    for s_id in SESSIONS:
        attempt(f'session_analysis_{s_id}', lambda s=s_id: spl.plot_session_analysis(
            s, d['t0_svm'], d['behavior'], d['svm_results'],
            x_modalities=[X_MODALITY], y_predicted_names=Y_NAMES, which_intervals=INTERVALS,
            show_covariate_plot=True, show_w_comparison=True, metric='balanced_accuracy',
            min_metric=.55, smooth=2, ith_to_dt_session=d['ith_to_dt'],
            save_dir=out_dir, save_markdown=False))

    # Two targets over two intervals: the only configuration that exercises
    # plot_w_comparison (guarded on ncols > 1) and the multi-interval concatenation
    # layout - both untested by the single-column case above.
    attempt('session_analysis_2col', lambda: spl.plot_session_analysis(
        SESSIONS[0], d['t0_svm'], d['behavior'], d['svm_results'],
        x_modalities=[X_MODALITY], y_predicted_names=['cue', 'choice_R1'],
        which_intervals=['cue_entry_interval', 'R1_entry_interval'],
        show_covariate_plot=True, show_w_comparison=True, metric='balanced_accuracy',
        min_metric=.55, smooth=2, ith_to_dt_session=d['ith_to_dt'],
        # plot_session_analysis does not create save_dir itself (the notebooks always
        # os.makedirs it first) - see plotlib.export.save_figure, which does.
        save_dir=_ensure(os.path.join(out_dir, '2col')), save_markdown=False))

    attempt('session_sequence_w', lambda: spl.plot_session_sequence_w_comparison(
        SESSIONS, d['svm_results'], d['t0_svm'], X_MODALITY, Y_NAMES[0], INTERVALS,
        metric='balanced_accuracy', min_metric=.55, ith_to_dt_session=d['ith_to_dt'],
        save_path=os.path.join(out_dir, 'session_sequence_w.png')))

    attempt('metric_heatmap', lambda: shl.plot_metric_heatmap_sessions(
        d['svm_results'], d['t0_svm'], which_intervals=INTERVALS, y_predicted_names=Y_NAMES,
        x_modality=X_MODALITY, session_ids=SESSIONS, ith_to_dt_session=d['ith_to_dt'],
        save_path=os.path.join(out_dir, 'metric_heatmap.png')))

    attempt('metric_distribution', lambda: shl.plot_metric_distribution(
        {'all': d['svm_results']}, which_intervals=INTERVALS, y_predicted_names=Y_NAMES,
        x_modality=X_MODALITY, session_ids=SESSIONS,
        save_path=os.path.join(out_dir, 'metric_distribution.png')))

    def _dim(s_id):
        fig, _ = dpl.plot_session_kinematics_neural_and_pca(
            d['t0_dim'], d['behavior'], d['fr_z'], s_id,
            ['R1_entry_interval', 'R2_entry_interval'], unit_cols=d['unit_cols'],
            smooth=3, pca_smooth=3, pca_n_pcs=6, angle_range=(30, 70),
            metric_range=(.7, 1.1), mean_z_range=(-.1, .1), var_explained_display='off',
            show_neuron_contrib=True, neuron_contrib_max_rows=12)
        fig.savefig(os.path.join(out_dir, f'dim_{s_id}.png'))
    attempt(f'dim_kinematics_pca_{SESSIONS[0]}', lambda: _dim(SESSIONS[0]))

    print(f"\n{len(done)} ok, {len(failed)} failed -> {out_dir}")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
    return len(failed)


def compare(dir_a, dir_b):
    """Report which rendered PNGs differ between two runs, and by how much."""
    import numpy as np
    from PIL import Image
    def pngs(root):
        return {os.path.relpath(os.path.join(dirpath, f), root)
                for dirpath, _, files in os.walk(root) for f in files if f.endswith('.png')}
    names = sorted(pngs(dir_a) | pngs(dir_b))
    identical, differing, missing = [], [], []
    for name in names:
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            missing.append(name)
            continue
        a, b = np.asarray(Image.open(pa).convert('RGB')), np.asarray(Image.open(pb).convert('RGB'))
        if a.shape != b.shape:
            differing.append((name, f'shape {a.shape} vs {b.shape}'))
        elif (a == b).all():
            identical.append(name)
        else:
            diff = (a.astype(int) - b.astype(int))
            pct = 100 * (np.abs(diff).sum(axis=2) > 0).mean()
            differing.append((name, f'{pct:.3f}% of pixels, max channel delta {np.abs(diff).max()}'))

    print(f"identical: {len(identical)}")
    for n in identical:
        print(f"    = {n}")
    print(f"differing: {len(differing)}")
    for n, how in differing:
        print(f"    ! {n}  ({how})")
    print(f"missing on one side: {len(missing)}")
    for n in missing:
        print(f"    ? {n}")
    return len(differing) + len(missing)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--libdir')
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2)
    args = ap.parse_args()
    if args.compare:
        sys.exit(1 if compare(*args.compare) else 0)
    _bootstrap(args.libdir)
    sys.exit(1 if render(args.out) else 0)
