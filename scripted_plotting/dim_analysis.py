#!/usr/bin/env python
# coding: utf-8

# %%
# to setup import paths add project root dirs to sys.path
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'widget')

from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args
from analytics_processing.sessions_from_nas_parsing import fullfnames2snames

# %%
nas_dir = C.device_paths()[0]
output_dir = os.path.join(nas_dir, 'Simon', 'mPFC_timealigned_PCA_dims',)
os.makedirs(output_dir, exist_ok=True)
Logger().init_logger(None, None, logging_level="INFO")

# %%
animal_ids = [6]
paradigm_ids = [1100]
session_ids = None

# %%
session_dirs, _ = sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, session_ids)
session_names = fullfnames2snames(session_dirs)
session_to_dt_session = {sn[:16]: i for i, sn in enumerate(session_names)}
session_names

# %%
# ---- data for the first plot: single-trial position kinematics across t0 intervals ----
t0_events = analytics.get_analytics('TrialWiseT0Events40ms2', session_names=session_names)
t0_events.index = t0_events.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
t0_events.set_index('trial_id', append=True, inplace=True)

behavior = analytics.get_analytics(
    'Behavior40msAligned', session_names=session_names,
    columns=['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
             'to_ephys_timestamp', 'frame_position'])
behavior = behavior.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
behavior.index = pd.IntervalIndex.from_arrays(
    left=behavior.to_ephys_timestamp - 40_000,
    right=behavior.to_ephys_timestamp,
    closed='right')
behavior.set_index('session_id', append=True, inplace=True)
t0_events.t0_event_name.unique()

# %%
# ---- ephys: population activity magnitude, aligned to the same t0 intervals ----
# Alternative A: raw (non-Z) firing rate, then Z-score per unit column pooled *across all
# sessions* instead of per-session (FiringRate40msZ's own approach - see get_FiringRate40msZ
# in postproc_mea1k_ephys.py). NB: 'UnitNNNN' is just that session's own sequential sorted-
# cluster index (see extract_jrc_spikes) - the same name in two different sessions is a
# different physical neuron, so this pools different neurons' distributions together under
# one column. Simon confirmed this tradeoff and wants it anyway, for a first pass.
fr_hz_all = analytics.get_analytics('FiringRate40msHz', session_names=session_names)
unit_cols = [c for c in fr_hz_all.columns if c.startswith('Unit')]

global_mean = fr_hz_all[unit_cols].mean()
global_std = fr_hz_all[unit_cols].std().replace(0, np.nan)  # avoid inf for an always-silent unit
fr_z_global = fr_hz_all.copy()
fr_z_global[unit_cols] = (fr_hz_all[unit_cols] - global_mean) / global_std

# %%
# Alternative B: the existing per-session Z-scored analytic, straight from disk
fr_z = analytics.get_analytics('FiringRate40msZ', session_names=session_names)

# %%
# pick which Z-scoring feeds the plot below (defaults to the plain per-session analytic -
# swap in fr_z_global to use the pooled-across-sessions version instead), then align it to
# behavior's own trial_id/IntervalIndex: fr_z/fr_z_global share the exact same 40ms ephys
# bin grid as behavior, row-for-row (confirmed on real data - both derive from the same
# FiringRate40msHz timestamp columns, see get_Behavior40msAligned), so behavior's trial_id
# can be copied across directly rather than needing its own trial_id column.
fr_z_plot = fr_z  # or fr_z_global
# fr_z_plot = fr_z_global

fr_z_plot = fr_z_plot.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
fr_z_plot['trial_id'] = behavior['trial_id'].values
fr_z_plot.index = pd.IntervalIndex.from_arrays(
    left=fr_z_plot.to_ephys_timestamp - 40_000,
    right=fr_z_plot.to_ephys_timestamp,
    closed='right')
fr_z_plot.set_index('session_id', append=True, inplace=True)

# slice to mPFC units:
fr_z_plot = fr_z_plot.iloc[:, 20:] # mPFC
# fr_z_plot = fr_z_plot.drop(columns=[f"Unit{i:04d}" for i in range(20, 78)]) # HP
fr_z_plot

# %%
import time
import dim_plotting_lib as dpl
from dim_plotting_lib import plot_session_kinematics_neural_and_pca

which_intervals = [
                #    'track_start_interval',
                #    'cue_entry_interval', 'cue_exit_interval',
                    'R1_entry_interval', 
                    # 'reward1_sound_interval',
                    'R2_entry_interval', 
                    # 'reward2_sound_interval',
                    # 'ITI_interval'
                    ]

saved_png_paths, saved_svg_paths = [], []
for i, s_id in enumerate(session_names[:20]):
# for i, s_id in enumerate(session_names[14:15]):
    s_id = s_id[:16]  # strip off the paradigm/animal suffix
    if s_id not in t0_events.index.unique('session_id'):
        print(f"Session {s_id} has no t0 events, skipping")
        continue
    if s_id not in fr_z_plot.index.unique('session_id'):
        print(f"Session {s_id} has no ephys, skipping")
        continue
    # every plot_session_kinematics_neural_and_pca kwarg, spelled out with its own default
    # value (see the function's docstring in dim_plotting_lib.py for what each one does) -
    # so tweaking one just means editing the value already sitting here, not looking up
    # what else there is to override. The fine-grained figure-geometry knobs (row heights,
    # gap_bins, colorbar sizing, ...) got cut from the function entirely - nothing here
    # ever changed them - so this list is now just the parameters that actually matter.
    fig, (ax0, ax1, ax2) = plot_session_kinematics_neural_and_pca(
        t0_events, behavior, fr_z_plot, s_id, which_intervals,
        unit_cols=None,
        smooth=3,                              # rows 0-1 (position/RMS) display smoothing
        pca_smooth=3,                          # smoothing applied before PCA
        pca_n_pcs=6,                           # fixed number of top PCs kept at every timebin
        angle_range=(30, 70),                  # canonical-angle heatmap color scale (true range is 0-90)
        metric_range=(.7, 1.1),                # fixed y-lim for row 1 (population RMS)
        mean_z_range=(-.1, .1),                # None = no line; else draws a dotted population-mean line on a twin y-axis
        draw_colorbar=True,                    # master switch for every colorbar
        var_explained_display='off',           # 'scatter' (halo on the RMS line) | 'off'; the old dedicated 'row' is gone,
                                               #   its curve is now the ceiling line of the summary row below the neuron heatmap
        var_explained_center=0.35,             # diverging colormap center (40% explained)
        var_explained_range=(0.1, 0.7),        # fixed color range, +/-0.3 around the center
        show_neuron_contrib=True,              # extra heatmap row: which neurons carry the subspace
        show_neuron_contrib_summary=True,      # thin line row below it: what the selection settings actually capture
        neuron_contrib_select='per_pc',        # 'per_pc' (top-|loading| per eigenvector) | 'top_contrib' (largest contributors)
        neuron_contrib_top_per_pc=2,           # 'per_pc' only: neurons drawn per PC per timebin
        neuron_contrib_top_n=12,               # 'top_contrib' only: neurons drawn per timebin
        neuron_contrib_fill='dense',           # 'dense' (every shown neuron at every timebin) | 'selected' (only nominated cells)
        neuron_contrib_max_rows=12,            # row cap, kept by largest total contribution
        neuron_contrib_range=(0, 0.10),        # cell color scale, fraction of total variance (None = autoscale to this session)
    )
    fig.suptitle(f"Session {i:02d} - {s_id}")
    plt.show()

    png_path = os.path.join(output_dir, f"{i:02d}_{s_id}_mPFC_pca.png")
    svg_path = os.path.join(output_dir, f"{i:02d}_{s_id}_mPFC_pca.svg")
    fig.savefig(png_path)
    fig.savefig(svg_path)
    saved_png_paths.append(png_path)
    saved_svg_paths.append(svg_path)
    break

# %% [markdown]
"""
Done — the dedicated variance-explained row is gone, its curve is now labelled 6 PCs explained var (all units) in the summary row, and the color scale is a fixed parameter. Your figure is now: position → RMS → neuron heatmap → summary → angle heatmap. Passing var_explained_display='row' raises with a pointer instead of silently doing nothing; I set your call site to 'off' ('scatter' still available for the halo on the RMS line).

Red vs orange — yes, both depend on select, but differently:

Orange = the selection itself, summed at each timebin. With top_contrib: exactly the top_n=12 largest contributors. With per_pc: the union of each of the 6 eigenvectors' top-3 leaders — so ≤ 18, and your instinct is right that it's usually fewer, because eigenvectors share leaders. On real data it's median 13, range 8–18. That's exactly what the dashed blue count line shows in per_pc mode (and why there's no such line in top_contrib mode — it would be flat at 12).
Red = the neurons that have a row in the heatmap above, at every timebin (dense fill). Its dependence on select is only indirect: the row ranking sums each neuron's contribution over the bins where it was selected, so per_pc favours transient specialists and top_contrib steady carriers. Once the 35 rows are chosen, red is just their total.
That's why red can sit above orange: a shown neuron that wasn't in this timebin's top-12 still contributes, and dense fill counts it. Both are always ≤ the gray ceiling.

And yes, your reading of the color is exactly right. A cell is that neuron's share of the timebin's total across-trial variance — not of the subspace's variance. Same quantity and same units as the gray line, which is why summing a column's cells over all neurons reproduces the gray line exactly (verified to 1e-15). So a cell reading 8% and a ceiling of 40% means that one neuron accounts for a fifth of what the whole 6-PC subspace captures.

Scale now fixed by parameter: neuron_contrib_range=(0, 0.10) is the new default (fractions of total variance; None restores per-session autoscale). I measured before picking it — across 6 sessions the 98th percentile of drawn cell values sat between 7.9% and 11.0% under both select rules (pooled median 0.51%, p90 3.4%, max 73%), so 0–10% loses essentially no contrast versus autoscaling while making colors mean the same thing in every session. About 1.9% of cells clip to the darkest red.

Verified: all 24 flag combinations (var_explained_display × select × fill × summary × heatmap) render with nothing off-canvas, and the real-session render in your exact config is 10.1×19.0in with the lowest colorbar at the 0.05in buffer.
"""


# %%
# one PNG canvas (build_canvas, rasterized via PIL) and one SVG canvas (build_canvas_svg,
# stays vector/editable - see dim_plotting_lib.py) with every session above placed side by
# side - each session here is a single image (no main+secondary stack), so each path gets
# wrapped in its own one-element list; ported from the same build_canvas/build_canvas_svg
# used for svm_analysis_2026.py's per-session posters.
if saved_png_paths:
    canvas_path = os.path.join(output_dir, 'mPFC_pca_canvas.png')
    dpl.build_canvas([[p] for p in saved_png_paths], canvas_path)
    print(f"Saved combined PNG canvas of {len(saved_png_paths)} sessions to {canvas_path}")

    canvas_path_svg = os.path.join(output_dir, 'mPFC_pca_canvas.svg')
    dpl.build_canvas_svg([[p] for p in saved_svg_paths], canvas_path_svg)
    print(f"Saved combined SVG canvas to {canvas_path_svg}")

# %%
# ==========================================================================================
# EXPERIMENTAL - delete this cell (and the next) once done exploring. Quick, minimal-effort
# hack: concatenate every session's own PC-subspace-per-timebin sequence (see
# dim_plotting_lib._subspace_block_with_separators - same per-timebin PCA the single-session
# heatmap above uses) into ONE long sequence, insert a small blank-separator gap between
# sessions (same zero-basis trick used between intervals within a session), then run the
# same canonical-angle comparison over the whole thing - i.e. the single-session heatmap
# above, scaled up to "does session A's subspace at time t look like session B's at time t'".
# Reuses dim_plotting_lib's own private helpers directly rather than adding anything new to
# that file, since this is meant to be thrown away, not become a permanent library function.
#
# Cost is O(n^2) in total bins across all included sessions (~50us/pair on this machine,
# measured directly) - full 29 sessions (~6400 bins incl. gaps) would take ~15-20 min.
# N_SESSIONS below caps it to a fast, iterable subset; raise it once you know this is worth
# waiting longer for.
N_SESSIONS = 17

valid_session_ids = [s_id for s_id in (sn[:16] for sn in session_names)
                     if s_id in t0_events.index.unique('session_id')
                     and s_id in fr_z_plot.index.unique('session_id')]
pca_session_ids = valid_session_ids[:N_SESSIONS]

SESSION_GAP = 5  # blank separator slots between sessions, same idea as gap_bins between intervals

all_bases = []
session_bounds = []  # (start_idx, end_idx, s_id) per session, for boundary lines/labels
for s_id in pca_session_ids:
    x_coords_s, bases_s, _, _, _ = dpl._subspace_block_with_separators(
        t0_events, fr_z_plot, s_id, which_intervals, unit_cols, gap_bins=2, smooth=5, n_pcs=6)
    if len(bases_s) == 0:
        print(f"Session {s_id}: no PCA bases, skipping")
        continue
    start = len(all_bases)
    all_bases.extend(bases_s)
    session_bounds.append((start, len(all_bases), s_id))
    all_bases.extend([np.zeros((0, len(unit_cols)))] * SESSION_GAP)

print(f"Total bins across {len(session_bounds)} sessions (incl. gaps): {len(all_bases)}")

# %%
n = len(all_bases)
angles = np.full((n, n), np.nan)
t_start = time.time()
for i in range(n):
    if all_bases[i].shape[0] == 0:
        continue
    if i % max(1, n // 10) == 0:
        print(f"  row {i}/{n} ({time.time() - t_start:.0f}s elapsed)")
    for j in range(i, n):
        if all_bases[j].shape[0] == 0:
            continue
        theta = dpl.subspace_angles(all_bases[i].T, all_bases[j].T)
        deg = np.mean(np.degrees(theta))
        angles[i, j] = angles[j, i] = deg
print(f"done in {time.time() - t_start:.0f}s")

fig, ax = plt.subplots(figsize=(12, 12))
cmap = plt.cm.viridis.with_extremes(bad='white')
mesh = ax.pcolormesh(np.ma.masked_invalid(angles), cmap=cmap, vmin=0, vmax=90)
for start, end, s_id in session_bounds:
    ax.axvline(x=start - 0.5, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axhline(y=start - 0.5, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
centers = [(start + end) / 2 for start, end, _ in session_bounds]
labels = [s_id for _, _, s_id in session_bounds]
ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticks(centers)
ax.set_yticklabels(labels, fontsize=7)
ax.set_aspect('equal', adjustable='datalim')
fig.colorbar(mesh, ax=ax, label='Canonical angle (deg)', shrink=0.8)
ax.set_title(f'Cross-session PC-subspace canonical angle ({len(session_bounds)} sessions)')
fig.tight_layout()
plt.show()

# %%
