#!/usr/bin/env python3
"""
Three supervisor figures for E07 x cue_visible and E23 x upcoming_choice.

Figure 1 — Data distributions: per-session Cohen's d bar chart + example
           session violin plots showing the real between-condition signal.

Figure 2 — Attribution performance: where GPV/IG work (E07) and fail (E23)
           when compared directly to Cohen's d.

Figure 3 — Collinearity experiment: feature collinearity bar chart, then
           original vs deconfounded GPV vs Cohen's d for E23.
"""

import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots
plt.style.use(['science', 'no-latex'])
import matplotlib.font_manager as _mpl_fm
for _fp in ['/mnt/c/Windows/Fonts/arial.ttf', '/mnt/c/Windows/Fonts/arialbd.ttf']:
    if os.path.exists(_fp): _mpl_fm.fontManager.addfont(_fp)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans']})
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

out_dir  = '../outputs/supervisor_figures'
desk_dir = '/mnt/c/Users/amits/Desktop/supervisor_figures'
for d in (out_dir, desk_dir):
    os.makedirs(d, exist_ok=True)

R2_THRESH = 0.01

# ── load shared data ──────────────────────────────────────────────────────────
with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names       = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

mlp_r2_all  = np.load('../outputs/mlps/ensembles_multiseed/all_r2.npy')
ceb_r2_all  = np.load('../outputs/cebra_eval/ensembles/all_r2.npy')
pred_r2_all = np.load('../outputs/cebra_pred_eval/ensembles/all_r2.npy')
mlp_valid   = (~np.all(np.isnan(mlp_r2_all),  axis=0)) & (np.nanmean(mlp_r2_all,  axis=0) >= R2_THRESH)
ceb_valid   = (~np.all(np.isnan(ceb_r2_all),  axis=0)) & (np.nanmean(ceb_r2_all,  axis=0) >= R2_THRESH)
pred_valid  = (~np.all(np.isnan(pred_r2_all), axis=0)) & (np.nanmean(pred_r2_all, axis=0) >= R2_THRESH)

mlp_gpv  = np.load('../outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy')
mlp_ig   = np.load('../outputs/mlps/ensembles_multiseed/importance_ig_semantic.npy')
ceb_ig   = np.load('../outputs/cebra_eval/ensembles/importance_ig_semantic.npy')
pred_gpv = np.load('../outputs/cebra_pred_eval/ensembles/importance_global_pv_semantic.npy')
pred_ig  = np.load('../outputs/cebra_pred_eval/ensembles/importance_ig_semantic.npy')

PAIRS = [
    dict(name='E07', ens_idx=6,  g_name='cue_visible',
         cond_labels=['No cue', 'Cue 1', 'Cue 2'],      color='#4CAF50'),
    dict(name='E23', ens_idx=22, g_name='upcoming_choice',
         cond_labels=['Skip', 'Baseline', 'Stop'],        color='#2196F3'),
]


def cohens_d_max(groups):
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2: return np.nan
    best = 0.0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            a, b = groups[i], groups[j]
            s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            if s > 1e-10: best = max(best, abs(a.mean() - b.mean()) / s)
    return best


def get_session_cd(sess, feat_cols, ens_idx):
    sd = ds[sess]
    X = np.concatenate([sd['data'][t][:, feat_cols] for t in sd['data']], axis=0)
    y = np.concatenate([sd['labels'][t][:, ens_idx] for t in sd['data']], axis=0)
    cond = np.argmax(X, axis=1)
    return cohens_d_max([y[cond == c] for c in range(X.shape[1])]), X, y, cond


def savefig(fname):
    for root in (out_dir, desk_dir):
        plt.savefig(os.path.join(root, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Data distributions
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], hspace=0.45, wspace=0.35)

for row, pair in enumerate(PAIRS):
    ens_idx   = pair['ens_idx']
    feat_cols = feat_idx_by_group[pair['g_name']]
    clabels   = pair['cond_labels']
    color     = pair['color']
    valid_s   = [s for s in range(len(sessions)) if mlp_valid[s, ens_idx]]

    # compute Cohen's d per session
    cds, sidxs = [], []
    for s_idx in valid_s:
        cd, _, _, _ = get_session_cd(sessions[s_idx], feat_cols, ens_idx)
        if not np.isnan(cd):
            cds.append(cd); sidxs.append(s_idx)
    cds, sidxs = np.array(cds), np.array(sidxs)
    order = np.argsort(cds)[::-1]
    cds, sidxs = cds[order], sidxs[order]

    # bar chart
    ax_bar = fig.add_subplot(gs[row, 0])
    thresh = np.percentile(cds, 60)   # top 40% highlighted
    bar_colors = [color if d >= thresh else '#BDBDBD' for d in cds]
    bars = ax_bar.bar(range(len(cds)), cds, color=bar_colors, edgecolor='none', width=0.8)
    ax_bar.axhline(thresh, color='#E53935', lw=1.2, ls='--', label=f'Threshold (d={thresh:.2f})')
    ax_bar.set_xticks(range(len(cds)))
    ax_bar.set_xticklabels([f'S{s:02d}' for s in sidxs], fontsize=6, rotation=60, ha='right')
    ax_bar.set_ylabel("Cohen's d", fontsize=10)
    ax_bar.set_title(f'{pair["name"]} × {pair["g_name"]} — per-session Cohen\'s d '
                     f'(n={len(cds)} sessions)', fontsize=10)
    ax_bar.legend(fontsize=8)

    # violin for top session
    ax_vio = fig.add_subplot(gs[row, 1])
    top_s   = sidxs[0]
    cd_top, X_top, y_top, cond_top = get_session_cd(sessions[top_s], feat_cols, ens_idx)
    vdata = [y_top[cond_top == c] for c in range(len(clabels))]
    vdata = [v for v in vdata if len(v) > 1]
    parts = ax_vio.violinplot(vdata, positions=range(len(vdata)),
                               showmedians=True, showextrema=False)
    cond_colors = ['#FF7043', '#42A5F5', '#66BB6A']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cond_colors[i % 3])
        pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
    ax_vio.set_xticks(range(len(vdata)))
    ax_vio.set_xticklabels(clabels[:len(vdata)], fontsize=9)
    ax_vio.set_ylabel('Neural activity', fontsize=9)
    ax_vio.set_title(f'S{top_s:02d} (d={cd_top:.2f}) — top session', fontsize=9)

plt.suptitle('Figure 1 — Between-condition neural activity differences vary across sessions',
             fontsize=12, fontweight='bold')
savefig('fig1_data_distributions.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Attribution performance
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# E07: three methods that work
e07_methods = [
    ('MLP IG',        mlp_ig,  mlp_valid,  '#4CAF50', 'o'),
    ('MLP GPV',       mlp_gpv, mlp_valid,  '#1B5E20', 's'),
    ('TempConv-Pred GPV',pred_gpv,pred_valid, '#E65100', 's'),
]
e07_ens, e07_fcols = 6, feat_idx_by_group['cue_visible']
e07_gidx = group_names.index('cue_visible')
e07_valid = [s for s in range(len(sessions)) if mlp_valid[s, e07_ens]]

e07_cd = {}
for s_idx in e07_valid:
    cd, _, _, _ = get_session_cd(sessions[s_idx], e07_fcols, e07_ens)
    e07_cd[s_idx] = cd

for col_idx, (label, arr, valid_arr, color, marker) in enumerate(e07_methods):
    ax = axes[0, col_idx]
    xs, ys, sidxs = [], [], []
    for s_idx in e07_valid:
        if not valid_arr[s_idx, e07_ens]: continue
        cd = e07_cd.get(s_idx, np.nan)
        if np.isnan(cd): continue
        xs.append(arr[s_idx, e07_ens, e07_gidx])
        ys.append(cd)
        sidxs.append(s_idx)
    xs, ys = np.array(xs), np.array(ys)
    rho, p = spearmanr(xs, ys)
    ax.scatter(xs, ys, color=color, marker=marker, s=65, alpha=0.85, zorder=3)
    for x, y, s in zip(xs, ys, sidxs):
        ax.annotate(f'S{s:02d}', (x, y), fontsize=6, xytext=(3,2), textcoords='offset points')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Cohen's d" if col_idx == 0 else '', fontsize=10)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.set_title(f'E07 × cue_visible\nρ={rho:+.3f}  p={p:.3f}  {sig}', fontsize=10)

# E23: three methods that fail
e23_methods = [
    ('MLP IG',         mlp_ig,  mlp_valid,  '#4CAF50', 'o'),
    ('TempConv-Cont IG',  ceb_ig,  ceb_valid,  '#2196F3', 'o'),
    ('TempConv-Pred GPV', pred_gpv,pred_valid, '#E65100', 's'),
]
e23_ens, e23_fcols = 22, feat_idx_by_group['upcoming_choice']
e23_gidx = group_names.index('upcoming_choice')
e23_valid = [s for s in range(len(sessions)) if mlp_valid[s, e23_ens]]

e23_cd = {}
for s_idx in e23_valid:
    cd, _, _, _ = get_session_cd(sessions[s_idx], e23_fcols, e23_ens)
    e23_cd[s_idx] = cd

for col_idx, (label, arr, valid_arr, color, marker) in enumerate(e23_methods):
    ax = axes[1, col_idx]
    xs, ys, sidxs = [], [], []
    for s_idx in e23_valid:
        if not valid_arr[s_idx, e23_ens]: continue
        cd = e23_cd.get(s_idx, np.nan)
        if np.isnan(cd): continue
        xs.append(arr[s_idx, e23_ens, e23_gidx])
        ys.append(cd)
        sidxs.append(s_idx)
    xs, ys = np.array(xs), np.array(ys)
    rho, p = spearmanr(xs, ys)
    ax.scatter(xs, ys, color=color, marker=marker, s=65, alpha=0.85, zorder=3)
    for x, y, s in zip(xs, ys, sidxs):
        ax.annotate(f'S{s:02d}', (x, y), fontsize=6, xytext=(3,2), textcoords='offset points')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Cohen's d" if col_idx == 0 else '', fontsize=10)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.set_title(f'E23 × upcoming_choice\nρ={rho:+.3f}  p={p:.3f}  {sig}', fontsize=10)

axes[0, 0].annotate('Attribution works (E07)', xy=(0.5, 1.18), xycoords='axes fraction',
                    ha='center', fontsize=11, color='#2E7D32', fontweight='bold')
axes[1, 0].annotate('Attribution fails without deconfounding (E23)', xy=(0.5, 1.18),
                    xycoords='axes fraction',
                    ha='center', fontsize=11, color='#C62828', fontweight='bold')

plt.suptitle('Figure 2 — Attribution scores predict condition coding in E07 but not E23',
             fontsize=12, fontweight='bold')
savefig('fig2_attribution_performance.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Collinearity experiment
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# Panel A: feature collinearity bar chart
COLLINEAR_THRESH = 0.5
col_groups = [(g, cols) for g, cols in sg if g != 'upcoming_choice']
feat_cds = {}
for g_name, g_cols in col_groups:
    cds = []
    for s_idx in e23_valid:
        sd = ds[sessions[s_idx]]
        X_ch = np.concatenate([sd['data'][t][:, e23_fcols] for t in sd['data']], axis=0)
        X_ft = np.concatenate([sd['data'][t][:, g_cols]   for t in sd['data']], axis=0)
        cond = np.argmax(X_ch, axis=1)
        fm   = X_ft.mean(axis=1)
        cds.append(cohens_d_max([fm[cond == c] for c in range(3)]))
    feat_cds[g_name] = float(np.nanmean(cds))

short_names = {
    'frame_raw_500msMedian':                    'frame_raw',
    'frame_raw_abs_acc_500msMedian':            'frame_acc',
    'frame_YawPitch_abs_vel_sum_500msMedian':   'yaw_vel',
    'frame_YawPitch_abs_acc_sum_500msMedian':   'yaw_acc',
    'head_angle_vel':                           'head_vel',
    'head_angle':                               'head_ang',
    'frame_position':                           'position',
    'cue_visible':                              'cue_vis',
    'reward_window':                            'reward',
    'lick_detected':                            'lick',
}

sorted_groups = sorted(feat_cds.items(), key=lambda x: x[1], reverse=True)
names_s  = [short_names.get(g, g) for g, _ in sorted_groups]
vals_s   = [v for _, v in sorted_groups]
colors_s = ['#E53935' if v >= COLLINEAR_THRESH else '#90A4AE' for v in vals_s]

ax_a = fig.add_subplot(gs[0])
ax_a.barh(range(len(vals_s)), vals_s, color=colors_s, edgecolor='none')
ax_a.axvline(COLLINEAR_THRESH, color='#E53935', lw=1.5, ls='--',
             label=f'Threshold (d={COLLINEAR_THRESH})')
ax_a.set_yticks(range(len(names_s)))
ax_a.set_yticklabels(names_s, fontsize=9)
ax_a.set_xlabel("Cohen's d (feature split by upcoming_choice)", fontsize=9)
ax_a.set_title('A — Feature collinearity\nwith upcoming_choice', fontsize=10)
ax_a.legend(fontsize=8)
ax_a.invert_yaxis()

# Panel B & C: original vs deconfounded GPV vs Cohen's d
df_deconf = pd.read_csv('../outputs/ablation_vs_attribution/E23_deconfounded_results.csv')
df_deconf['cohen_d'] = [e23_cd.get(r.s_idx, np.nan) for _, r in df_deconf.iterrows()]
df_deconf['orig_gpv'] = [mlp_gpv[r.s_idx, e23_ens, e23_gidx] for _, r in df_deconf.iterrows()]

for panel_idx, (col, label, color, panel_label) in enumerate([
    ('orig_gpv', 'Original GPV\n(all 10 feature groups)',       '#1B5E20', 'B'),
    ('gpv',      'Deconfounded GPV\n(5 collinear groups removed)', '#E65100', 'C'),
]):
    ax = fig.add_subplot(gs[panel_idx + 1])
    sub = df_deconf[[col, 'cohen_d']].dropna()
    rho, p = spearmanr(sub[col], sub['cohen_d'])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else f'p={p:.2f}'))
    ax.scatter(sub[col], sub['cohen_d'], color=color, s=65, alpha=0.85, zorder=3)
    for _, row in sub.iterrows():
        s = int(df_deconf.loc[row.name, 's_idx'])
        ax.annotate(f'S{s:02d}', (row[col], row['cohen_d']),
                    fontsize=6, xytext=(3,2), textcoords='offset points')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Cohen's d (upcoming_choice)" if panel_idx == 0 else '', fontsize=10)
    ax.set_title(f'{panel_label} — E23 × upcoming_choice\nρ={rho:+.3f}  {sig}  n={len(sub)}',
                 fontsize=10)

plt.suptitle('Figure 3 — Collinearity masks upcoming_choice attribution; '
             'deconfounding reveals the signal',
             fontsize=12, fontweight='bold')
savefig('fig3_collinearity_experiment.png')

print('\nAll figures saved.')
