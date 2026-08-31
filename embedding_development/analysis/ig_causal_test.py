#!/usr/bin/env python3
"""
ig_causal_test.py

Tests whether the baseline-distance mechanism is the actual cause of high IG
attribution in "genuine over-attribution" pairs.

IG = x_HA × mean_gradient_along_path
           ^^^           ^^^^^^^^^^^
      path length     gradient component

Baseline-distance hypothesis:
  genuine_overattr: large |x_HA|, small mean_gradient → IG inflated by path
  nonmono_correct:  moderate |x_HA|, large mean_gradient → IG earned by gradient

Three tests (in increasing compute):
  Test 1: Statistical — mean |x_HA| per session across categories
  Test 2: IG decomposition — ratio IG / mean(|x_HA|) per pair (no model loading)
  Test 3: Permutation test — shuffle x_HA in test data, compare real vs shuffled IG
           (loads models; N_SAMPLE pairs randomly sampled per category)
  Test 4: Gradient-function shape — plot dF/dx_HA vs x_HA for each sampled pair
"""
import os, sys, pickle, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from scipy.stats import mannwhitneyu, kruskal
from sklearn.metrics import r2_score
from datetime import datetime

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP

# ─────────────────────────────── CONFIG ─────────────────────────────────────
TUNING_CSV  = "outputs/mlps/head_angle_tuning_20260517_1634/head_angle_tuning_table.csv"
ATTR_DIR    = "outputs/mlps/ensembles_multiseed"
DATASET_PATH = "outputs/session_dataset_ensembles.pkl"
MODELS_ROOT  = "models/mlps/ensembles"
SPLITS_DIR   = "splits"
SEED         = 42
IG_STEPS     = 50
HA_FEAT_IDX  = 5
INPUT_SIZE   = 26
HIDDEN_SIZE  = 64
NUM_HIDDEN   = 2
OUTPUT_SIZE  = 1
N_SAMPLE     = 6     # pairs per category for permutation / gradient shape tests

random.seed(0)
np.random.seed(0)

ts        = datetime.now().strftime('%Y%m%d_%H%M')
OUT_DIR   = f"outputs/mlps/ig_causal_test_{ts}"
DESK_DIR  = f"/mnt/c/Users/amits/Desktop/ig_causal_test_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DESK_DIR, exist_ok=True)
print(f"Output: {OUT_DIR}")

PALETTE = {
    'mono_correct':    '#2196F3',
    'nonmono_correct': '#FF9800',
    'genuine_overattr':'#F44336',
    'under_attr':      '#9E9E9E',
}
LABELS = {
    'mono_correct':    'Monotonic\ncorrect',
    'nonmono_correct': 'Non-monotonic\ncorrect',
    'genuine_overattr':'Genuine\nover-attr',
    'under_attr':      'Under-\nattr',
}
CATS = ['mono_correct', 'nonmono_correct', 'genuine_overattr', 'under_attr']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")


def savefig(name):
    for d in (OUT_DIR, DESK_DIR):
        plt.savefig(os.path.join(d, name), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}")


def load_model(s_i, n_i):
    path = os.path.join(MODELS_ROOT, f"seed{SEED}",
                        f"session_{s_i:02d}_neuron_{n_i:02d}.pt")
    m = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_HIDDEN, OUTPUT_SIZE).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


# ──────────────────────────── LOAD BASE DATA ────────────────────────────────
print("Loading data...")
df_tuning = pd.read_csv(TUNING_CSV)
# session column is 1-indexed in the CSV (s_i + 1), ensemble is 1-indexed too
df_tuning['s_i'] = df_tuning['session'] - 1
df_tuning['n_i'] = df_tuning['ensemble'].str.replace('E', '').astype(int) - 1
df_valid = df_tuning[df_tuning['category'].isin(CATS)].copy()

attr_r2 = np.load(os.path.join(ATTR_DIR, "all_r2.npy"))          # (seeds, S, N)
ig_mat  = np.load(os.path.join(ATTR_DIR, "importance_ig_semantic.npy"))  # (S, N, G)
# IG for head_angle group = group index 5
HA_GRP  = 5

with open(DATASET_PATH, 'rb') as f:
    dataset = pickle.load(f)
session_keys = sorted(dataset.keys())

# Load test-split indices for seed=42
split_map = np.load(os.path.join(SPLITS_DIR, f"split_seed{SEED}.npy"),
                    allow_pickle=True).item()


def get_test_X(s_i):
    """Return test-split X array for a session."""
    sk   = session_keys[s_i]
    tidx = split_map[sk]
    X    = np.concatenate([dataset[sk]['data'][t] for t in tidx], axis=0).astype(np.float32)
    return X


def get_full_X(s_i):
    """Return all-trials X array for a session."""
    sk = session_keys[s_i]
    X  = np.concatenate([dataset[sk]['data'][t]
                         for t in sorted(dataset[sk]['data'])], axis=0).astype(np.float32)
    return X


# Per-session statistics: mean |x_HA| computed on ALL trials (same as head angle σ proxy)
print("Computing session HA statistics...")
session_ha_stats = {}
for s_i in range(len(session_keys)):
    X = get_full_X(s_i)
    ha = X[:, HA_FEAT_IDX]
    session_ha_stats[s_i] = {
        'mean_abs_ha': float(np.mean(np.abs(ha))),
        'std_ha':      float(np.std(ha)),
        'range_ha':    float(np.ptp(ha)),
    }

# Attach per-session stats to each pair
df_valid['mean_abs_ha'] = df_valid['s_i'].map(lambda s: session_ha_stats[s]['mean_abs_ha'])
df_valid['std_ha']      = df_valid['s_i'].map(lambda s: session_ha_stats[s]['std_ha'])

# Attach stored IG per pair  (already in CSV as ig_attr — averaged over seeds)
# Compute gradient proxy = ig_attr / mean_abs_ha
df_valid['gradient_proxy'] = df_valid['ig_attr'] / (df_valid['mean_abs_ha'] + 1e-8)

print(f"\nCategory counts: {df_valid['category'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — Statistical comparison of path-length component (mean |x_HA|)
# ════════════════════════════════════════════════════════════════════════════
print("\n══ TEST 1: Path-length comparison across categories ══")

for metric in ['mean_abs_ha', 'std_ha', 'ig_attr', 'eta2']:
    groups = [df_valid[df_valid['category'] == c][metric].dropna().values for c in CATS]
    stat, pval = kruskal(*groups)
    print(f"  {metric}: Kruskal-Wallis H={stat:.2f}, p={pval:.4f}")
    for ci, c1 in enumerate(CATS):
        for c2 in CATS[ci+1:]:
            g1 = df_valid[df_valid['category'] == c1][metric].dropna()
            g2 = df_valid[df_valid['category'] == c2][metric].dropna()
            u, p = mannwhitneyu(g1, g2, alternative='two-sided')
            if p < 0.05:
                print(f"    {c1} vs {c2}: U={u:.0f}, p={p:.4f}  *")

# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — IG gradient decomposition (no model loading)
# ════════════════════════════════════════════════════════════════════════════
print("\n══ TEST 2: IG decomposition (gradient proxy = IG / mean|x_HA|) ══")

for c in CATS:
    sub = df_valid[df_valid['category'] == c]
    print(f"  {c:20s}  n={len(sub):3d}  "
          f"mean|x_HA|={sub['mean_abs_ha'].mean():.3f}±{sub['mean_abs_ha'].std():.3f}  "
          f"ig={sub['ig_attr'].mean():.3f}±{sub['ig_attr'].std():.3f}  "
          f"grad_proxy={sub['gradient_proxy'].mean():.3f}±{sub['gradient_proxy'].std():.3f}")

gp_by_cat = {c: df_valid[df_valid['category']==c]['gradient_proxy'].values for c in CATS}
stat, p = kruskal(*[gp_by_cat[c] for c in CATS])
print(f"\n  gradient_proxy Kruskal-Wallis: H={stat:.2f}, p={p:.4f}")

# Key pairwise: genuine_overattr vs nonmono_correct
g1 = gp_by_cat['genuine_overattr']
g2 = gp_by_cat['nonmono_correct']
u, p = mannwhitneyu(g1, g2, alternative='less')  # H1: genuine_overattr gradient < nonmono
print(f"  gradient_proxy: genuine_overattr < nonmono_correct: U={u:.0f}, p={p:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — Permutation test (model loading, N_SAMPLE per category)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n══ TEST 3: Permutation test (N={N_SAMPLE} per category, seed={SEED}) ══")

N_PERMS = 10

def compute_ig_ha(model, Xnp, steps=IG_STEPS):
    """Returns mean |IG| for the head_angle feature."""
    n_samp = len(Xnp)
    test_t = torch.tensor(Xnp, dtype=torch.float32, device=device)
    base_t = torch.zeros(1, INPUT_SIZE, dtype=torch.float32, device=device)
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:]
    grads_acc = np.zeros((n_samp, INPUT_SIZE), dtype=np.float32)
    for alpha in alphas:
        x_int = (base_t + float(alpha) * (test_t - base_t)).detach().requires_grad_(True)
        with torch.enable_grad():
            _, pred = model(x_int)
            pred.sum().backward()
        grads_acc += x_int.grad.detach().cpu().numpy()
        del x_int
    grads_acc /= steps
    ig_per_sample = Xnp * grads_acc        # (n_samp, n_feats)
    return float(np.mean(np.abs(ig_per_sample[:, HA_FEAT_IDX])))


def compute_gradient_curve(model, Xnp):
    """
    Returns (ha_vals, grad_vals): per-sample head_angle value and gradient
    magnitude at that point (not integrated — instantaneous ∂f/∂x_HA).
    """
    Xt = torch.tensor(Xnp, dtype=torch.float32, device=device).requires_grad_(True)
    with torch.enable_grad():
        _, pred = model(Xt)
        pred.sum().backward()
    grads = Xt.grad.detach().cpu().numpy()   # (n_samp, n_feats)
    return Xnp[:, HA_FEAT_IDX], grads[:, HA_FEAT_IDX]


perm_results = []

for cat in CATS:
    cat_rows = df_valid[df_valid['category'] == cat].copy()
    # Random sample (or take all if fewer than N_SAMPLE)
    sample = cat_rows.sample(n=min(N_SAMPLE, len(cat_rows)), random_state=0)
    print(f"\n  Category: {cat}  (sampled {len(sample)} pairs)")

    for _, row in sample.iterrows():
        s_i = int(row['s_i'])
        n_i = int(row['n_i'])
        model = load_model(s_i, n_i)
        Xtest = get_test_X(s_i)

        ig_real = compute_ig_ha(model, Xtest)

        # Permuted IG: shuffle only x_HA column
        ig_perm_list = []
        for _ in range(N_PERMS):
            Xperm = Xtest.copy()
            idx   = np.random.permutation(len(Xperm))
            Xperm[:, HA_FEAT_IDX] = Xperm[idx, HA_FEAT_IDX]
            ig_perm_list.append(compute_ig_ha(model, Xperm))
        ig_shuffled = float(np.mean(ig_perm_list))

        ratio = ig_real / (ig_shuffled + 1e-8)
        perm_results.append({
            'category': cat, 's_i': s_i, 'n_i': n_i,
            'ig_real': ig_real, 'ig_shuffled': ig_shuffled, 'ratio': ratio,
            'eta2': row['eta2'], 'mean_abs_ha': row['mean_abs_ha'],
            'gradient_proxy': row['gradient_proxy'],
        })
        print(f"    S{s_i+1:02d} {row['ensemble']}:  "
              f"IG_real={ig_real:.4f}  IG_shuffled={ig_shuffled:.4f}  "
              f"ratio={ratio:.3f}")
        del model

df_perm = pd.DataFrame(perm_results)
df_perm.to_csv(os.path.join(OUT_DIR, 'permutation_results.csv'), index=False)

print("\n  Mean ratio by category:")
for cat in CATS:
    sub = df_perm[df_perm['category'] == cat]
    if len(sub) > 0:
        print(f"    {cat:20s}  ratio={sub['ratio'].mean():.3f}±{sub['ratio'].std():.3f}  n={len(sub)}")

# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — Gradient-function shape: dF/dx_HA vs x_HA (sampled pairs)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n══ TEST 4: Gradient function shape dF/dx_HA vs x_HA ══")

grad_shape_results = {}
for cat in CATS:
    cat_rows = df_valid[df_valid['category'] == cat]
    sample = cat_rows.sample(n=min(N_SAMPLE, len(cat_rows)), random_state=0)
    grad_shape_results[cat] = []
    for _, row in sample.iterrows():
        s_i, n_i = int(row['s_i']), int(row['n_i'])
        model     = load_model(s_i, n_i)
        Xfull     = get_full_X(s_i)          # use full data for denser coverage
        ha_vals, grad_vals = compute_gradient_curve(model, Xfull)
        grad_shape_results[cat].append({
            'ha': ha_vals, 'grad': grad_vals,
            's_i': s_i, 'n_i': n_i,
            'label': f"S{s_i+1} {row['ensemble']}",
            'eta2': row['eta2'], 'ig': row['ig_attr'],
        })
        del model
    print(f"  {cat}: computed gradient curves for {len(grad_shape_results[cat])} pairs")


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

# ── Fig 1: Tests 1+2 — statistical overview ──────────────────────────────
print("\n── Plotting Figs 1-2 ──")
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.patch.set_facecolor('white')

metrics = [
    ('mean_abs_ha',     'Mean |x_HA| (path length)'),
    ('ig_attr',         'IG attribution (head_angle)'),
    ('gradient_proxy',  'IG / mean|x_HA|\n(gradient proxy)'),
    ('eta2',            'η² (nonlinear tuning)'),
]
for ax, (col, label) in zip(axes, metrics):
    parts = ax.violinplot(
        [df_valid[df_valid['category']==c][col].dropna().values for c in CATS],
        positions=range(len(CATS)), showmedians=True, widths=0.7,
    )
    for pc, cat in zip(parts['bodies'], CATS):
        pc.set_facecolor(PALETTE[cat])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    for key in ('cbars', 'cmins', 'cmaxes'):
        parts[key].set_color('black')
        parts[key].set_linewidth(1)
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels([LABELS[c] for c in CATS], fontsize=8)
    ax.set_ylabel(label, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

# Annotate gradient proxy panel with Mann-Whitney p
ax_gp = axes[2]
g1_vals = df_valid[df_valid['category']=='genuine_overattr']['gradient_proxy'].dropna()
g2_vals = df_valid[df_valid['category']=='nonmono_correct']['gradient_proxy'].dropna()
_, p_gp = mannwhitneyu(g1_vals, g2_vals, alternative='less')
ax_gp.text(0.5, 0.97,
           f"genuine_overattr < nonmono\np={p_gp:.3f}",
           transform=ax_gp.transAxes, ha='center', va='top', fontsize=8.5,
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888'))

fig.suptitle("Tests 1+2: path length vs gradient proxy across categories",
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig("fig1_statistical_tests.png")


# ── Fig 2: IG decomposition scatter ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')

ax = axes[0]
for cat in CATS:
    sub = df_valid[df_valid['category'] == cat]
    ax.scatter(sub['mean_abs_ha'], sub['ig_attr'],
               color=PALETTE[cat], s=30, alpha=0.7, linewidths=0, label=LABELS[cat].replace('\n',' '))
# Regression
xv = df_valid['mean_abs_ha'].values; yv = df_valid['ig_attr'].values
m, b = np.polyfit(xv, yv, 1)
xfit = np.linspace(xv.min(), xv.max(), 100)
ax.plot(xfit, m*xfit+b, 'k--', linewidth=1.5, alpha=0.6)
r, p = scipy_stats.spearmanr(xv, yv)
ax.text(0.97, 0.03, f"ρ={r:.3f}, p={p:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
ax.set_xlabel("Mean |x_HA|  (path length proxy)", fontsize=11)
ax.set_ylabel("IG attribution (head_angle)", fontsize=11)
ax.set_title("IG vs path length — all pairs", fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
for cat in CATS:
    sub = df_valid[df_valid['category'] == cat]
    ax.scatter(sub['eta2'], sub['gradient_proxy'],
               color=PALETTE[cat], s=30, alpha=0.7, linewidths=0, label=LABELS[cat].replace('\n',' '))
ax.set_xlabel("η²  (nonlinear tuning)", fontsize=11)
ax.set_ylabel("Gradient proxy  (IG / mean|x_HA|)", fontsize=11)
ax.set_title("After removing path-length effect:\ngradient proxy vs actual tuning (η²)", fontsize=11)
ax.legend(fontsize=8, loc='upper left')
# Annotate: if hypothesis is right, genuine_overattr clusters bottom-left
ax.text(0.97, 0.97,
        "Baseline-distance prediction:\ngenuine_overattr → bottom-left\n(low η², low gradient)",
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF9C4', ec='#F9A825'))
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle("Test 2: IG decomposed into path-length × gradient components",
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig("fig2_ig_decomposition.png")


# ── Fig 3: Permutation test results ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor('white')

# Panel A: real vs shuffled IG per pair, colored by category
ax = axes[0]
for cat in CATS:
    sub = df_perm[df_perm['category'] == cat]
    ax.scatter(sub['ig_shuffled'], sub['ig_real'],
               color=PALETTE[cat], s=60, alpha=0.85, linewidths=0.5,
               edgecolors='white', label=LABELS[cat].replace('\n', ' '), zorder=3)
lim = max(df_perm[['ig_real','ig_shuffled']].max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', linewidth=1.2, alpha=0.5, label='y=x (no effect)')
ax.set_xlabel("IG shuffled x_HA", fontsize=11)
ax.set_ylabel("IG real x_HA", fontsize=11)
ax.set_title("Real vs shuffled IG\n(above line = real > shuffled)", fontsize=11)
ax.legend(fontsize=8.5, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)

# Panel B: ratio real/shuffled by category
ax = axes[1]
for xi, cat in enumerate(CATS):
    sub = df_perm[df_perm['category'] == cat]
    if len(sub) == 0:
        continue
    ratios = sub['ratio'].values
    ax.scatter(np.full(len(ratios), xi) + np.random.normal(0, 0.07, len(ratios)),
               ratios, color=PALETTE[cat], s=55, alpha=0.8, linewidths=0, zorder=3)
    ax.errorbar(xi, ratios.mean(), yerr=ratios.std(),
                fmt='D', color='black', markersize=7, capsize=5, zorder=4, linewidth=1.5)
ax.axhline(1.0, color='gray', linewidth=1.2, linestyle='--', alpha=0.7)
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels([LABELS[c] for c in CATS], fontsize=9)
ax.set_ylabel("Ratio: IG_real / IG_shuffled", fontsize=11)
ax.set_title("Permutation ratio\n(>1 = real structure; ≈1 = path-length only)", fontsize=11)
ax.spines[['top', 'right']].set_visible(False)

# Annotate with Kruskal-Wallis and key pairwise tests
ratio_groups = [df_perm[df_perm['category']==c]['ratio'].values for c in CATS if df_perm[df_perm['category']==c]['ratio'].notna().any()]
if len(ratio_groups) >= 2:
    try:
        stat_kw, p_kw = kruskal(*ratio_groups)
        ax.text(0.5, 0.97, f"Kruskal-Wallis p={p_kw:.3f}", transform=ax.transAxes,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888'))
    except Exception:
        pass

# Panel C: ratio vs eta2 per pair
ax = axes[2]
for cat in CATS:
    sub = df_perm[df_perm['category'] == cat]
    ax.scatter(sub['eta2'], sub['ratio'],
               color=PALETTE[cat], s=55, alpha=0.85, linewidths=0,
               label=LABELS[cat].replace('\n', ' '), zorder=3)
ax.axhline(1.0, color='gray', linewidth=1.2, linestyle='--', alpha=0.7)
r_c, p_c = scipy_stats.spearmanr(df_perm['eta2'], df_perm['ratio'])
ax.text(0.97, 0.03, f"ρ={r_c:.3f}, p={p_c:.3f}", transform=ax.transAxes,
        ha='right', va='bottom', fontsize=9)
ax.set_xlabel("η²  (nonlinear tuning)", fontsize=11)
ax.set_ylabel("Ratio: IG_real / IG_shuffled", fontsize=11)
ax.set_title("Permutation ratio vs η²\n(prediction: η² → ratio; baseline-dist → flat)", fontsize=11)
ax.legend(fontsize=8.5, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle("Test 3: Permutation test — does shuffling x_HA collapse IG?",
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig("fig3_permutation_test.png")


# ── Fig 4: Gradient-function shape (dF/dx_HA vs x_HA) ───────────────────
print("── Plotting Fig 4: gradient shape ──")

N_BINS_GRAD = 15
cats_show = ['genuine_overattr', 'nonmono_correct']
fig, axes = plt.subplots(len(cats_show), N_SAMPLE,
                         figsize=(N_SAMPLE * 2.5, len(cats_show) * 3.2),
                         sharex=False, sharey=False)
fig.patch.set_facecolor('white')

for row_i, cat in enumerate(cats_show):
    color = PALETTE[cat]
    for col_i, entry in enumerate(grad_shape_results[cat]):
        ax = axes[row_i][col_i]
        ha_arr = entry['ha']
        gr_arr = entry['grad']

        # Bin x_HA, take mean gradient per bin (keeps sign)
        percentiles = np.linspace(0, 100, N_BINS_GRAD + 1)
        edges = np.percentile(ha_arr, percentiles); edges[-1] += 1e-6
        bin_centers, bin_means, bin_stds = [], [], []
        for b in range(N_BINS_GRAD):
            mask_b = (ha_arr >= edges[b]) & (ha_arr < edges[b+1])
            if mask_b.sum() >= 3:
                bin_centers.append(np.mean(ha_arr[mask_b]))
                bin_means.append(np.mean(gr_arr[mask_b]))
                bin_stds.append(np.std(gr_arr[mask_b]) / np.sqrt(mask_b.sum()))
        bin_centers = np.array(bin_centers)
        bin_means   = np.array(bin_means)
        bin_stds    = np.array(bin_stds)

        ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                        alpha=0.2, color=color)
        ax.plot(bin_centers, bin_means, 'o-', color=color, linewidth=1.8, markersize=4)
        ax.axhline(0, color='#999', linewidth=0.7, linestyle='--')

        # Annotate
        ax.set_title(f"{entry['label']}\nη²={entry['eta2']:.3f}, IG={entry['ig']:.3f}",
                     fontsize=7.5, color=color)
        if col_i == 0:
            ax.set_ylabel(f"{LABELS[cat].replace(chr(10), ' ')}\n∂f/∂x_HA", fontsize=8)
        if row_i == len(cats_show) - 1:
            ax.set_xlabel("x_HA", fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

fig.suptitle(
    "Test 4: Gradient function shape dF/dx_HA vs x_HA\n"
    "Genuine over-attr: flat/noisy gradient (path drives IG)\n"
    "Non-monotonic correct: structured gradient (real tuning)",
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
savefig("fig4_gradient_shape.png")


# ── Fig 5: Combined summary ───────────────────────────────────────────────
print("── Plotting Fig 5: combined summary ──")
fig = plt.figure(figsize=(16, 8))
gs  = gridspec.GridSpec(2, 4, hspace=0.45, wspace=0.4,
                        left=0.07, right=0.97, top=0.90, bottom=0.08)

# A: IG vs path length
ax_a = fig.add_subplot(gs[0, 0])
for cat in CATS:
    sub = df_valid[df_valid['category'] == cat]
    ax_a.scatter(sub['mean_abs_ha'], sub['ig_attr'],
                 color=PALETTE[cat], s=25, alpha=0.7, linewidths=0,
                 label=LABELS[cat].replace('\n', ' '))
ax_a.set_xlabel("Mean |x_HA|", fontsize=10); ax_a.set_ylabel("IG attribution", fontsize=10)
ax_a.set_title("(A) IG vs path length", fontsize=10, fontweight='bold')
ax_a.legend(fontsize=7, loc='upper left')
ax_a.spines[['top', 'right']].set_visible(False)

# B: gradient proxy violin
ax_b = fig.add_subplot(gs[0, 1])
gp_data = [df_valid[df_valid['category']==c]['gradient_proxy'].dropna().values for c in CATS]
vp = ax_b.violinplot(gp_data, positions=range(len(CATS)), showmedians=True, widths=0.65)
for pc, cat in zip(vp['bodies'], CATS):
    pc.set_facecolor(PALETTE[cat]); pc.set_alpha(0.75)
vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
for k in ('cbars','cmins','cmaxes'):
    vp[k].set_color('black'); vp[k].set_linewidth(1)
ax_b.set_xticks(range(len(CATS)))
ax_b.set_xticklabels([LABELS[c] for c in CATS], fontsize=7.5)
ax_b.set_ylabel("IG / mean|x_HA|\n(gradient proxy)", fontsize=10)
ax_b.set_title("(B) Gradient proxy\nby category", fontsize=10, fontweight='bold')
ax_b.text(0.5, 0.97, f"p(genuine<nonmono)={p_gp:.3f}",
          transform=ax_b.transAxes, ha='center', va='top', fontsize=8.5,
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888'))
ax_b.spines[['top', 'right']].set_visible(False)

# C: permutation ratio
ax_c = fig.add_subplot(gs[0, 2])
for xi, cat in enumerate(CATS):
    sub = df_perm[df_perm['category'] == cat]
    if len(sub) == 0: continue
    jit = np.random.normal(0, 0.06, len(sub))
    ax_c.scatter(np.full(len(sub), xi) + jit, sub['ratio'],
                 color=PALETTE[cat], s=50, alpha=0.8, linewidths=0, zorder=3)
    ax_c.errorbar(xi, sub['ratio'].mean(), yerr=sub['ratio'].std(),
                  fmt='D', color='black', markersize=6, capsize=4, zorder=4, lw=1.5)
ax_c.axhline(1.0, color='gray', linewidth=1.2, linestyle='--', alpha=0.7)
ax_c.set_xticks(range(len(CATS)))
ax_c.set_xticklabels([LABELS[c] for c in CATS], fontsize=7.5)
ax_c.set_ylabel("IG_real / IG_shuffled", fontsize=10)
ax_c.set_title("(C) Permutation ratio\n(≈1 = path-length only)", fontsize=10, fontweight='bold')
ax_c.spines[['top', 'right']].set_visible(False)

# D: ratio vs eta2
ax_d = fig.add_subplot(gs[0, 3])
for cat in CATS:
    sub = df_perm[df_perm['category'] == cat]
    ax_d.scatter(sub['eta2'], sub['ratio'],
                 color=PALETTE[cat], s=50, alpha=0.85, linewidths=0,
                 label=LABELS[cat].replace('\n', ' '), zorder=3)
ax_d.axhline(1.0, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)
ax_d.set_xlabel("η²", fontsize=10); ax_d.set_ylabel("IG_real / IG_shuffled", fontsize=10)
ax_d.set_title(f"(D) η² vs permutation ratio\nρ={r_c:.3f}, p={p_c:.3f}", fontsize=10, fontweight='bold')
ax_d.spines[['top', 'right']].set_visible(False)

# E-H: gradient shape examples (one per category, two best examples)
for col_offset, cat in enumerate(['genuine_overattr', 'nonmono_correct']):
    color = PALETTE[cat]
    entries = grad_shape_results[cat]
    # pick the pair with highest IG (most extreme case)
    entries_sorted = sorted(entries, key=lambda e: e['ig'], reverse=True)
    for row_offset, entry in enumerate(entries_sorted[:2]):
        ax = fig.add_subplot(gs[1, col_offset * 2 + row_offset])
        ha_arr = entry['ha']; gr_arr = entry['grad']
        N_BINS_G = 12
        edges = np.percentile(ha_arr, np.linspace(0, 100, N_BINS_G+1)); edges[-1] += 1e-6
        bcs, bms, bss = [], [], []
        for b in range(N_BINS_G):
            m_b = (ha_arr >= edges[b]) & (ha_arr < edges[b+1])
            if m_b.sum() >= 3:
                bcs.append(np.mean(ha_arr[m_b])); bms.append(np.mean(gr_arr[m_b]))
                bss.append(np.std(gr_arr[m_b]) / np.sqrt(m_b.sum()))
        bcs = np.array(bcs); bms = np.array(bms); bss = np.array(bss)
        ax.fill_between(bcs, bms-bss, bms+bss, alpha=0.2, color=color)
        ax.plot(bcs, bms, 'o-', color=color, linewidth=2, markersize=5)
        ax.axhline(0, color='#999', linewidth=0.7, linestyle='--')
        ax.set_title(f"({chr(69+col_offset*2+row_offset)}) {entry['label']}\n"
                     f"η²={entry['eta2']:.3f}, IG={entry['ig']:.3f}",
                     fontsize=9, color=color, fontweight='bold')
        ax.set_xlabel("x_HA", fontsize=9)
        ax.set_ylabel("∂f/∂x_HA", fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

fig.suptitle(
    "Causal test: Is baseline-distance the mechanism for genuine over-attribution?\n"
    "Prediction: genuine_overattr has large path (A), small gradient (B), ratio≈1 (C), flat gradient shape (E-F)",
    fontsize=12, fontweight='bold'
)
savefig("fig5_combined_causal_test.png")

# ─────────────────────────────── SUMMARY ───────────────────────────────────
print("\n═══════════════════════════════════════════════════════")
print("SUMMARY")
print("═══════════════════════════════════════════════════════")
for cat in CATS:
    sub_v = df_valid[df_valid['category'] == cat]
    sub_p = df_perm[df_perm['category'] == cat]
    print(f"\n  {cat}:")
    print(f"    mean |x_HA|  = {sub_v['mean_abs_ha'].mean():.3f}")
    print(f"    IG           = {sub_v['ig_attr'].mean():.3f}")
    print(f"    grad_proxy   = {sub_v['gradient_proxy'].mean():.3f}")
    print(f"    η²           = {sub_v['eta2'].mean():.3f}")
    if len(sub_p) > 0:
        print(f"    perm ratio   = {sub_p['ratio'].mean():.3f}±{sub_p['ratio'].std():.3f}")

print(f"\n  Key test: gradient_proxy (genuine_overattr < nonmono_correct): p={p_gp:.4f}")
print(f"  Key test: permutation ratio ~ η²: ρ={r_c:.3f}, p={p_c:.4f}")
print(f"\nDone. Output: {OUT_DIR}")
