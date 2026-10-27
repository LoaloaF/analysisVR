"""Regression harness for the code moved out of svm_analysis_2026.py into
w_plotting_lib.py / plotlib / dim_plotting_lib.py.

Unlike _smoke_render.py, the pre-refactor side of this comparison never existed as an
importable module - it was ~900 lines of function definitions sitting between notebook
cells. So for a baseline this script *extracts* those definitions from the original
notebook's AST and execs them in a namespace with the globals they used to close over
(`analytics`, `ith_to_dt_session`), which is precisely the coupling the move removed.

    python _smoke_render_w.py --libdir ../scripted_plotting --out /tmp/wbase
    python _smoke_render_w.py --libdir .                    --out /tmp/wnew
    python _smoke_render.py --compare /tmp/wbase /tmp/wnew

Every figure left open by a call is saved by this script rather than relying on each
function's own save_dir/save_path argument, since those differ per function (and some
write SVG only).
"""

import argparse
import ast
import inspect
import os
import sys

SESSION = '2024-12-04_18-06'
X_MODALITY = 'mPFC'
Y_NAME = 'choice_R1'
INTERVAL = 'R1_entry_interval'
WINDOW = (5, 15)
SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
T0_ANALYTIC = 'TrialWiseT0Events40ms2'  # matched to SVML1CueOutcomeChoicePred2

MOVED = ['_desaturate', 'REGION_LABELS', 'REGION_COLORS', 'REGION_FRACTIONS',
         '_region_gradient_cmap', '_plot_region_gradient', 'plot_W_list',
         '_x_modality_fr', 'get_trial_w_projection', 'get_trial_behavior_trace',
         '_trial_covariate_levels', '_interval_x0', '_shared_ylim', 'plot_trial_velocity',
         'plot_w_projection', '_NESTED_SPLIT_PARTNER', '_covariate_panel_specs',
         'plot_w_projection_covariate_single', 'plot_w_projection_covariates',
         'plot_w_projection_session', '_pca_eigen', 'plot_pca_eigenvalues',
         'get_interval_window_activity', 'plot_w_variance_explained',
         '_window_trial_bin_values', 'KINEMATIC_COLUMNS', 'plot_w_projection_kinematics',
         'plot_session_projection_kinematics', 'plot_session_pca', '_canonical_angles_deg',
         'plot_session_pca_similarity', 'plot_pca_reference_drift']


def _from_notebook(libdir, ith_to_dt_session):
    """Exec the moved definitions straight out of the original notebook."""
    import numpy as np, pandas as pd, textwrap
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from matplotlib.colors import LinearSegmentedColormap, to_rgb, rgb_to_hsv, hsv_to_rgb
    from analytics_processing import analytics
    import svm_plotting_lib as spl

    path = os.path.join(libdir, 'svm_analysis_2026.py')
    src = open(path).read()
    lines = src.split('\n')
    tree = ast.parse(src)
    wanted, chunks = set(MOVED), []
    for node in tree.body:
        name = (node.name if isinstance(node, ast.FunctionDef) else
                next((t.id for t in getattr(node, 'targets', []) if isinstance(t, ast.Name)), None))
        if name in wanted:
            chunks.append('\n'.join(lines[node.lineno - 1:node.end_lineno]))
            wanted.discard(name)
    if wanted:
        print(f"  note: not found in notebook (already moved out?): {sorted(wanted)}")

    ns = dict(os=os, np=np, pd=pd, plt=plt, stats=stats, textwrap=textwrap,
              LinearSegmentedColormap=LinearSegmentedColormap, to_rgb=to_rgb,
              rgb_to_hsv=rgb_to_hsv, hsv_to_rgb=hsv_to_rgb, analytics=analytics,
              ith_to_dt_session=ith_to_dt_session,
              ALL_Y_PREDICTED_NAMES=spl.ALL_Y_PREDICTED_NAMES,
              CORRECTNESS_SPECS=spl.CORRECTNESS_SPECS, _covariate_color=spl._covariate_color,
              _level_label=spl._level_label, _interval_short_label=spl._interval_short_label,
              _smooth=spl._smooth)
    exec('\n\n'.join(chunks), ns)
    return ns


def get_fns(libdir, ith_to_dt_session, source='auto'):
    # The script's own directory is always on sys.path, so a plain `import w_plotting_lib`
    # would find the refactored module even when --libdir points at the pre-refactor tree.
    # Default to the notebook extraction unless --libdir IS this directory.
    if source == 'auto':
        source = 'lib' if os.path.abspath(libdir) == os.path.dirname(os.path.abspath(__file__)) else 'notebook'
    if source == 'notebook':
        print("  using: definitions extracted from the original notebook")
        return _from_notebook(libdir, ith_to_dt_session)
    try:
        import w_plotting_lib as wpl
        import dim_plotting_lib as dpl
        from plotlib.pca import pca_eigen, plot_pca_eigenvalues
        ns = {n: getattr(wpl, n) for n in dir(wpl) if not n.startswith('__')}
        for n in ['plot_session_pca', 'plot_session_pca_similarity', 'plot_pca_reference_drift']:
            ns[n] = getattr(dpl, n)
        ns['_pca_eigen'], ns['plot_pca_eigenvalues'] = pca_eigen, plot_pca_eigenvalues
        print("  using: w_plotting_lib + dim_plotting_lib + plotlib.pca")
        return ns
    except ImportError as e:
        print(f"  using: definitions extracted from the original notebook ({e})")
        return _from_notebook(libdir, ith_to_dt_session)


def render(libdir, out_dir, source='auto'):
    import numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    from analytics_processing import analytics
    from analytics_processing.sessions_from_nas_parsing import (
        sessionlist_fullfnames_from_args, fullfnames2snames)
    import svm_plotting_lib as spl

    os.makedirs(out_dir, exist_ok=True)
    session_dirs, _ = sessionlist_fullfnames_from_args([1100], [6], None)
    session_names = fullfnames2snames(session_dirs)
    ith = {sn[:16]: i for i, sn in enumerate(session_names)}

    t0 = analytics.get_analytics(T0_ANALYTIC, session_names=session_names)
    t0.index = t0.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
    t0.set_index('trial_id', append=True, inplace=True)
    behavior = analytics.get_analytics('Behavior40msAligned', session_names=session_names)
    behavior = behavior.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
    behavior.index = pd.IntervalIndex.from_arrays(
        left=behavior.to_ephys_timestamp - 40_000, right=behavior.to_ephys_timestamp, closed='right')
    behavior.set_index('session_id', append=True, inplace=True)
    svm_results = analytics.get_analytics(SVM_ANALYTIC, session_names=session_names)

    # a real hand-picked W, via the same path the notebook uses
    highlighted = spl.resolve_w_selections(
        [{'session_id': SESSION, 'y_name': Y_NAME, 'interval': INTERVAL,
          'timebin': 22, 'name': 'test_w'}],
        svm_results, x_modality=X_MODALITY)
    assert len(highlighted) == 1, f"expected 1 highlighted W, got {len(highlighted)}"
    w_row = highlighted.iloc[0]

    fns = get_fns(libdir, ith, source)
    done, failed = [], []

    def attempt(name, call):
        plt.close('all')
        try:
            fn = fns[name.split('|')[0]]
            kw = {}
            if 'ith_to_dt_session' in inspect.signature(fn).parameters:
                kw['ith_to_dt_session'] = ith
            call(fn, kw)
            for i, num in enumerate(plt.get_fignums()):
                plt.figure(num).savefig(os.path.join(out_dir, f'{name.replace("|", "_")}_{i}.png'), dpi=100)
            done.append(name)
            print(f"  ok    {name}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            plt.close('all')

    attempt('plot_W_list', lambda f, kw: f(highlighted, plot_type='stem', **kw))
    attempt('plot_w_projection_session', lambda f, kw: f(
        SESSION, w_row, t0, behavior, session_names, [INTERVAL], smooth=2, **kw))
    attempt('plot_w_variance_explained', lambda f, kw: f(
        {'session_id': SESSION, 'interval': INTERVAL, 'timebin_range': WINDOW,
         'x_modality': X_MODALITY, 'name': 'test_w'}, w_row, t0, session_names, **kw))
    attempt('plot_w_projection_kinematics', lambda f, kw: f(
        SESSION, w_row, t0, behavior, session_names, INTERVAL, WINDOW, **kw))
    attempt('plot_session_projection_kinematics', lambda f, kw: f(
        SESSION, w_row, behavior, session_names, max_points=5000, **kw))
    attempt('plot_session_pca', lambda f, kw: f(SESSION, session_names, X_MODALITY, **kw))

    print(f"\n{len(done)} ok, {len(failed)} failed -> {out_dir}")
    for n, e in failed:
        print(f"  FAILED {n}: {e}")
    return len(failed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--libdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='auto', choices=['auto', 'lib', 'notebook'])
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.libdir))
    sys.path.append('/home/vrmaster/Projects/VirtualReality')
    from baseVR.base_functionality import init_import_paths
    init_import_paths()
    import matplotlib
    matplotlib.use('Agg')
    from CustomLogger import CustomLogger as Logger
    Logger().init_logger(None, None, logging_level="ERROR")
    sys.exit(1 if render(args.libdir, args.out, args.source) else 0)
