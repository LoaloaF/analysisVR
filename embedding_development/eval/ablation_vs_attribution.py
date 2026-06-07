#!/usr/bin/env python3
"""
ablation_vs_attribution.py

For E07 x cue_visible and E23 x upcoming_choice:

  1. Train a full MLP (2 hidden layers, 64 units) using ONLY the 3 one-hot
     columns of that categorical variable as input — across all valid sessions.
     5 seeds, same train/test splits as the full model.

  2. Compare per-session ablation R² against:
       - MLP IG
       - MLP GPV
       - TempConv-Cont IG
       - TempConv-Cont GPV

  If attribution scores are useful, they should rank sessions by ablation R²:
  high-attribution sessions should have higher ablation R² than low-attribution
  ones.

Outputs:
  outputs/ablation_vs_attribution/   + Desktop copy
"""

import os, sys, shutil, pickle, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

# ── config ────────────────────────────────────────────────────────────────────
SEEDS      = [42, 43, 44, 45, 46]
N_EPOCHS   = 150
LR         = 1e-3
HIDDEN     = 64
N_LAYERS   = 2
BATCH_SIZE = 512
R2_THRESH  = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

output_dir  = '../outputs/ablation_vs_attribution'
desktop_dir = '/mnt/c/Users/amits/Desktop/ablation_vs_attribution'
model_dir   = '../outputs/ablation_vs_attribution/models'
# clear plots/CSVs but preserve saved models
for d in (output_dir, desktop_dir):
    if os.path.exists(d):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                os.remove(p)
    else:
        os.makedirs(d)
os.makedirs(model_dir, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names       = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

mlp_gpv  = np.load('../outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy')
mlp_ig   = np.load('../outputs/mlps/ensembles_multiseed/importance_ig_semantic.npy')
ceb_gpv  = np.load('../outputs/cebra_eval/ensembles/importance_global_pv_semantic.npy')
ceb_ig   = np.load('../outputs/cebra_eval/ensembles/importance_ig_semantic.npy')
pred_gpv = np.load('../outputs/cebra_pred_eval/ensembles/importance_global_pv_semantic.npy')
pred_ig  = np.load('../outputs/cebra_pred_eval/ensembles/importance_ig_semantic.npy')

mlp_r2_all  = np.load('../outputs/mlps/ensembles_multiseed/all_r2.npy')
ceb_r2_all  = np.load('../outputs/cebra_eval/ensembles/all_r2.npy')
pred_r2_all = np.load('../outputs/cebra_pred_eval/ensembles/all_r2.npy')

mlp_valid  = (~np.all(np.isnan(mlp_r2_all),  axis=0)) & (np.nanmean(mlp_r2_all,  axis=0) >= R2_THRESH)
ceb_valid  = (~np.all(np.isnan(ceb_r2_all),  axis=0)) & (np.nanmean(ceb_r2_all,  axis=0) >= R2_THRESH)
pred_valid = (~np.all(np.isnan(pred_r2_all), axis=0)) & (np.nanmean(pred_r2_all, axis=0) >= R2_THRESH)

PAIRS = [
    dict(name='E07_cue_visible',     ens_idx=6,  g_name='cue_visible',
         feat_cols=feat_idx_by_group['cue_visible'],
         class_names=['no_cue', 'cue1', 'cue2']),
    dict(name='E23_upcoming_choice',  ens_idx=22, g_name='upcoming_choice',
         feat_cols=feat_idx_by_group['upcoming_choice'],
         class_names=['skip', 'baseline', 'stop']),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def cohens_d_max(arrays):
    """Max pairwise Cohen's d across a list of 1-D arrays."""
    arrays = [a for a in arrays if len(a) > 1]
    if len(arrays) < 2:
        return np.nan
    best = 0.0
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            a, b = arrays[i], arrays[j]
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            if pooled < 1e-10:
                continue
            best = max(best, abs(np.mean(a) - np.mean(b)) / pooled)
    return best


# ── model ─────────────────────────────────────────────────────────────────────
def make_mlp(in_size):
    layers, prev = [], in_size
    for _ in range(N_LAYERS):
        layers += [nn.Linear(prev, HIDDEN), nn.ReLU()]
        prev = HIDDEN
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers).to(device)


def train_and_eval(X_tr, y_tr, X_te, y_te, seed, model_path=None):
    if model_path and os.path.exists(model_path):
        net = make_mlp(X_tr.shape[1])
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        torch.manual_seed(seed)
        net     = make_mlp(X_tr.shape[1])
        opt     = torch.optim.Adam(net.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
        for _ in range(N_EPOCHS):
            perm = torch.randperm(len(Xt), device=device)
            for b in range(0, len(Xt), BATCH_SIZE):
                idx = perm[b:b + BATCH_SIZE]
                opt.zero_grad()
                loss_fn(net(Xt[idx]), yt[idx]).backward()
                opt.step()
        if model_path:
            torch.save(net.state_dict(), model_path)

    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(X_te, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
    r2 = float(r2_score(y_te, pred)) if np.var(y_te) > 1e-8 else np.nan

    # Cohen's d: predicted condition means / actual neural activity SD
    cond = np.argmax(X_te, axis=1)
    n_conds = X_te.shape[1]
    pred_means = [pred[cond == c].mean() if (cond == c).any() else np.nan
                  for c in range(n_conds)]
    y_by_cond  = [y_te[cond == c] for c in range(n_conds)]
    pooled_sd  = np.sqrt(np.mean([np.var(y, ddof=1) for y in y_by_cond if len(y) > 1]))
    cd = np.nan
    if pooled_sd > 1e-10:
        diffs = [abs(pred_means[i] - pred_means[j])
                 for i in range(n_conds) for j in range(i+1, n_conds)
                 if not (np.isnan(pred_means[i]) or np.isnan(pred_means[j]))]
        if diffs:
            cd = float(max(diffs) / pooled_sd)

    # linear baseline on same features
    lr = LinearRegression().fit(X_tr, y_tr)
    r2_lin = float(r2_score(y_te, lr.predict(X_te))) if np.var(y_te) > 1e-8 else np.nan
    return r2, r2_lin, cd


def get_arrays(sess, trial_ids, feat_cols, n_idx):
    sd = ds[sess]
    vt = [t for t in trial_ids if t in sd['data']]
    X  = np.concatenate([sd['data'][t][:, feat_cols]   for t in vt], axis=0).astype(np.float32)
    y  = np.concatenate([sd['labels'][t][:, n_idx]     for t in vt], axis=0).astype(np.float32)
    return X, y


def savefig(fname):
    for root in (output_dir, desktop_dir):
        plt.savefig(os.path.join(root, fname), dpi=150, bbox_inches='tight')
    plt.close()


# ── main loop ─────────────────────────────────────────────────────────────────
all_dfs = {}

for pair in PAIRS:
    name      = pair['name']
    ens_idx   = pair['ens_idx']
    g_name    = pair['g_name']
    feat_cols = pair['feat_cols']
    g_idx     = group_names.index(g_name)

    print(f'\n{"="*60}')
    print(f'  {name}  (ensemble {ens_idx}, group {g_name})')
    print(f'{"="*60}')

    rows = []
    valid_sessions = [s for s in range(len(sessions)) if mlp_valid[s, ens_idx]]
    print(f'  MLP-valid sessions: {len(valid_sessions)}')

    for s_idx in valid_sessions:
        sess = sessions[s_idx]
        r2s_mlp, r2s_lin, cds_abl = [], [], []

        for seed in SEEDS:
            split_map    = np.load(f'../splits/split_seed{seed}.npy', allow_pickle=True).item()
            test_trials  = [int(i) for i in split_map[sess]]
            all_trials   = list(ds[sess]['data'].keys())
            train_trials = [t for t in all_trials if t not in test_trials]

            X_tr, y_tr = get_arrays(sess, train_trials, feat_cols, ens_idx)
            X_te, y_te = get_arrays(sess, test_trials,  feat_cols, ens_idx)

            if len(X_tr) < 50 or len(X_te) < 20:
                continue

            mpath = os.path.join(model_dir, f'{name}_S{s_idx:02d}_seed{seed}.pt')
            r2_mlp, r2_lin, cd = train_and_eval(X_tr, y_tr, X_te, y_te, seed, mpath)
            if not np.isnan(r2_mlp):
                r2s_mlp.append(r2_mlp)
                r2s_lin.append(r2_lin)
                if not np.isnan(cd):
                    cds_abl.append(cd)

        if not r2s_mlp:
            continue

        abl_r2      = float(np.median(r2s_mlp))
        abl_lin_r2  = float(np.median(r2s_lin))
        abl_cohen_d = float(np.median(cds_abl)) if cds_abl else np.nan
        full_r2    = float(np.nanmean(mlp_r2_all[:, s_idx, ens_idx]))
        mi_val     = mlp_ig[s_idx, ens_idx, g_idx]
        mg_val     = mlp_gpv[s_idx, ens_idx, g_idx]
        ci_val     = ceb_ig[s_idx, ens_idx, g_idx]    if ceb_valid[s_idx, ens_idx]  else np.nan
        cg_val     = ceb_gpv[s_idx, ens_idx, g_idx]   if ceb_valid[s_idx, ens_idx]  else np.nan
        pi_val     = pred_ig[s_idx, ens_idx, g_idx]   if pred_valid[s_idx, ens_idx] else np.nan
        pg_val     = pred_gpv[s_idx, ens_idx, g_idx]  if pred_valid[s_idx, ens_idx] else np.nan

        rows.append(dict(
            s_idx=s_idx, session=sess,
            abl_r2=abl_r2, abl_cohen_d=abl_cohen_d, abl_lin_r2=abl_lin_r2, full_r2=full_r2,
            mlp_ig=mi_val, mlp_gpv=mg_val,
            ceb_ig=ci_val, ceb_gpv=cg_val,
            pred_ig=pi_val, pred_gpv=pg_val,
            n_seeds=len(r2s_mlp),
        ))
        print(f'  S{s_idx:02d}  abl_r2={abl_r2:.4f}  abl_cd={abl_cohen_d:.3f}  '
              f'mlp_ig={mi_val:.4f}  ceb_ig={ci_val:.4f}  pred_ig={pi_val:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f'{name}_ablation.csv'), index=False)
    all_dfs[name] = df
    print(f'\n  Saved {name}_ablation.csv  ({len(df)} sessions)')

    if len(df) < 4:
        continue

    methods = [
        ('mlp_ig',   'MLP IG',           '#4CAF50', 'o'),
        ('mlp_gpv',  'MLP GPV',          '#1B5E20', 's'),
        ('ceb_ig',   'TempConv-Cont IG',    '#2196F3', 'o'),
        ('ceb_gpv',  'TempConv-Cont GPV',   '#0D47A1', 's'),
        ('pred_ig',  'TempConv-Pred IG',    '#FF9800', 'o'),
        ('pred_gpv', 'TempConv-Pred GPV',   '#E65100', 's'),
    ]

    # internal check: how well does abl_cohen_d track abl_r2?
    sub_check = df[['abl_r2', 'abl_cohen_d']].dropna()
    rho_cd_r2, p_cd_r2 = spearmanr(sub_check['abl_r2'], sub_check['abl_cohen_d'])
    print(f'\n  abl_cohen_d vs abl_r2: ρ={rho_cd_r2:+.3f}  p={p_cd_r2:.3f}  n={len(sub_check)}')

    for target_col, target_label, fname_suffix in [
        ('abl_r2',      'Ablation R²',       'ablation_r2'),
        ('abl_cohen_d', "Ablation Cohen's d", 'ablation_cohend'),
    ]:
        # ── scatter ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 6, figsize=(26, 4.5))
        for ax, (col, label, color, marker) in zip(axes, methods):
            sub = df[[target_col, col]].dropna()
            if len(sub) < 4:
                ax.set_title(f'{label}\n(insufficient data)')
                continue
            rho, p = spearmanr(sub[col], sub[target_col])
            ax.scatter(sub[col], sub[target_col], color=color, marker=marker,
                       s=60, alpha=0.8, zorder=3)
            for _, row in sub.iterrows():
                ax.annotate(f"S{int(df.loc[row.name,'s_idx']):02d}",
                            (row[col], row[target_col]),
                            fontsize=6, xytext=(3, 2), textcoords='offset points')
            if target_col == 'abl_r2':
                ax.axhline(0, color='gray', lw=0.8, ls='--')
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel(target_label, fontsize=9)
            ax.set_title(f'{label}\nρ={rho:.3f}  p={p:.3f}', fontsize=10)

        plt.suptitle(f'{name.replace("_", " ")} — attribution vs {target_label}', fontsize=11)
        plt.tight_layout()
        savefig(f'{name}_attribution_vs_{fname_suffix}.png')
        print(f'  Saved {name}_attribution_vs_{fname_suffix}.png')

        # ── rank correlations ─────────────────────────────────────────────────
        print(f'\n  Rank correlations (Spearman ρ) with {target_label}:')
        for col, label, _, _ in methods:
            sub = df[[target_col, col]].dropna()
            if len(sub) < 4:
                print(f'    {label:20s}: n<4')
                continue
            rho, p = spearmanr(sub[col], sub[target_col])
            print(f'    {label:20s}: ρ={rho:+.3f}  p={p:.3f}  n={len(sub)}')


# ── combined plots: R² and Cohen's d side by side ────────────────────────────
titles = {
    'E07_cue_visible':    'E07 × cue_visible',
    'E23_upcoming_choice':'E23 × upcoming_choice',
}
model_styles = [
    ('mlp',  'MLP',          '#4CAF50', 'o', 0.00),
    ('ceb',  'TempConv-Cont',   '#2196F3', 's', 0.12),
    ('pred', 'TempConv-Pred',   '#FF9800', '^', 0.24),
]

for target_col, target_label, out_name in [
    ('abl_r2',      'Ablation R²',       'combined_attribution_vs_ablation_r2.png'),
    ('abl_cohen_d', "Ablation Cohen's d", 'combined_attribution_vs_ablation_cohend.png'),
]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row_idx, (pair_name, df) in enumerate(all_dfs.items()):
        for col_idx, (attr_type, attr_label) in enumerate([('ig', 'IG'), ('gpv', 'GPV')]):
            ax = axes[row_idx, col_idx]
            for model, label, mcolor, mk, yoff in model_styles:
                col = f'{model}_{attr_type}'
                if col not in df.columns:
                    continue
                sub = df[[target_col, col]].dropna()
                if len(sub) < 4:
                    continue
                ax.scatter(sub[col], sub[target_col], color=mcolor,
                           marker=mk, s=55, alpha=0.85, label=label, zorder=3)
                rho, p = spearmanr(sub[col], sub[target_col])
                ax.text(0.05, 0.95 - yoff,
                        f'{label}: ρ={rho:+.3f} p={p:.2f}',
                        transform=ax.transAxes, fontsize=7,
                        color=mcolor, va='top')

            if target_col == 'abl_r2':
                ax.axhline(0, color='gray', lw=0.8, ls='--')
            ax.set_xlabel(f'{attr_label} attribution', fontsize=10)
            ax.set_ylabel(target_label if col_idx == 0 else '')
            ax.set_title(f'{titles[pair_name]} — {attr_label}', fontsize=10)
            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(f'Do attribution scores predict per-session {target_label}?\n'
                 '(ablation MLP trained on 3 one-hot inputs only)', fontsize=11)
    plt.tight_layout()
    savefig(out_name)
    print(f'Saved {out_name}')

print('Done.')
