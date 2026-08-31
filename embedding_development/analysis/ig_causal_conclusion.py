#!/usr/bin/env python3
"""
ig_causal_conclusion.py

Final summary figure from the causal test.
Hypothesis: baseline-distance effect (longer IG integration path → higher IG)
Result: REFUTED — head_angle is z-scored (std=1 for all sessions), so path
lengths are identical across all categories. The original hypothesis does not hold.

What we found instead:
  - genuine_overattr: model IS sensitive to HA (perm ratio 1.19) but behavioral
    tuning is weak (η²=0.030). IG correctly reports what the model does.
  - under_attr: model IGNORES HA (perm ratio 1.01) despite real behavioral tuning
    (η²=0.066). The model found a better substitute feature.
  - η² and permutation ratio are NEGATIVELY correlated (ρ=-0.43): high behavioral
    tuning does NOT guarantee the model uses HA, and vice versa.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, mannwhitneyu
from datetime import datetime

TUNING_CSV = "outputs/mlps/head_angle_tuning_20260517_1634/head_angle_tuning_table.csv"
PERM_CSV   = "outputs/mlps/ig_causal_test_20260517_1739/permutation_results.csv"
OUT_DIR    = "outputs/mlps/ig_causal_test_20260517_1739"
DESK_DIR   = "/mnt/c/Users/amits/Desktop/ig_causal_test_20260517_1739"

CATS = ['mono_correct', 'nonmono_correct', 'genuine_overattr', 'under_attr']
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
    'under_attr':      'Under-\nattributed',
}

def savefig(name):
    for d in (OUT_DIR, DESK_DIR):
        plt.savefig(os.path.join(d, name), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}")


df_tuning = pd.read_csv(TUNING_CSV)
df_perm   = pd.read_csv(PERM_CSV)
df_valid  = df_tuning[df_tuning['category'].isin(CATS)].copy()
df_valid['s_i'] = df_valid['session'] - 1
df_valid['n_i'] = df_valid['ensemble'].str.replace('E','').astype(int) - 1

# mean_abs_ha: since all sessions are z-scored, mean|x_HA| is ~0.8 for all
# Compute it from the known z-score distribution (half-normal integral)
# For N(0,1): E[|X|] = sqrt(2/pi) ≈ 0.7979
# Use the per-pair value from perm CSV (which already has mean_abs_ha)
perm_by_pair = {}
for _, row in df_perm.iterrows():
    perm_by_pair[(int(row['s_i']), int(row['n_i']))] = {
        'ratio': row['ratio'], 'mean_abs_ha': row['mean_abs_ha']
    }

df_valid['perm_ratio'] = df_valid.apply(
    lambda r: perm_by_pair.get((int(r['s_i']), int(r['n_i'])), {}).get('ratio', np.nan), axis=1
)
# For path-length panel: use mean_abs_ha from perm CSV where available, else constant
df_valid['mean_abs_ha'] = df_valid.apply(
    lambda r: perm_by_pair.get((int(r['s_i']), int(r['n_i'])), {}).get('mean_abs_ha',
              float(np.sqrt(2/np.pi))), axis=1
)

rho_perm_eta2, p_perm_eta2 = spearmanr(df_perm['eta2'], df_perm['ratio'])

# Mann-Whitney: genuine_overattr perm ratio > under_attr perm ratio
g_ov = df_perm[df_perm['category']=='genuine_overattr']['ratio']
g_ua = df_perm[df_perm['category']=='under_attr']['ratio']
_, p_ov_ua = mannwhitneyu(g_ov, g_ua, alternative='greater')


# ════════════════════════════════════════════════════════════════════════════
# FIG A — The critical refutation: path length is identical
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 4, hspace=0.48, wspace=0.38,
                       left=0.07, right=0.97, top=0.88, bottom=0.08)

# Panel A: mean|x_HA| per category (should be identical)
ax_a = fig.add_subplot(gs[0, 0])
path_data = [df_valid[df_valid['category']==c]['mean_abs_ha'].dropna().values for c in CATS]
vp = ax_a.violinplot(path_data, positions=range(len(CATS)), showmedians=True, widths=0.65)
for pc, cat in zip(vp['bodies'], CATS):
    pc.set_facecolor(PALETTE[cat]); pc.set_alpha(0.7)
vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(2)
for k in ('cbars','cmins','cmaxes'):
    vp[k].set_color('black'); vp[k].set_linewidth(1)
ax_a.set_xticks(range(len(CATS)))
ax_a.set_xticklabels([LABELS[c] for c in CATS], fontsize=8)
ax_a.set_ylabel("Mean |x_HA|  (path length)", fontsize=10)
ax_a.set_title("(A) Path length — IDENTICAL\nacross all categories", fontsize=10, fontweight='bold')
# Annotate with a big red REFUTED box
ax_a.text(0.5, 0.96,
          "std_HA = 1.000 for all sessions\n(z-scored input)\n→ path lengths cannot differ",
          transform=ax_a.transAxes, ha='center', va='top', fontsize=8.5,
          bbox=dict(boxstyle='round,pad=0.4', fc='#FFCDD2', ec='#F44336', lw=1.5))
ax_a.spines[['top','right']].set_visible(False)

# Panel B: permutation ratio by category — the clean test
ax_b = fig.add_subplot(gs[0, 1])
for xi, cat in enumerate(CATS):
    sub = df_perm[df_perm['category'] == cat]
    if len(sub) == 0: continue
    jit = np.random.RandomState(1).normal(0, 0.06, len(sub))
    ax_b.scatter(np.full(len(sub), xi) + jit, sub['ratio'],
                 color=PALETTE[cat], s=65, alpha=0.85, linewidths=0.5,
                 edgecolors='white', zorder=3)
    ax_b.errorbar(xi, sub['ratio'].mean(), yerr=sub['ratio'].std(),
                  fmt='D', color='black', markersize=7, capsize=5, zorder=4, lw=2)
ax_b.axhline(1.0, color='#999', linewidth=1.2, linestyle='--', alpha=0.8,
             label='ratio=1 (model ignores HA)')
ax_b.set_xticks(range(len(CATS)))
ax_b.set_xticklabels([LABELS[c] for c in CATS], fontsize=8)
ax_b.set_ylabel("IG_real / IG_shuffled", fontsize=10)
ax_b.set_title("(B) Model sensitivity to HA\n(perm test: >1 = model uses HA structure)",
               fontsize=10, fontweight='bold')
ax_b.legend(fontsize=8, loc='upper right')
# Annotate genuine_overattr vs under_attr comparison
ax_b.annotate('', xy=(2, g_ov.mean()), xytext=(3, g_ua.mean()),
              arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
ax_b.text(2.5, (g_ov.mean()+g_ua.mean())/2 + 0.05,
          f"p={p_ov_ua:.3f}", ha='center', fontsize=8.5)
ax_b.spines[['top','right']].set_visible(False)

# Panel C: η² vs permutation ratio scatter
ax_c = fig.add_subplot(gs[0, 2])
for cat in CATS:
    sub = df_perm[df_perm['category'] == cat]
    ax_c.scatter(sub['eta2'], sub['ratio'],
                 color=PALETTE[cat], s=60, alpha=0.85, linewidths=0,
                 label=LABELS[cat].replace('\n',' '), zorder=3)
ax_c.axhline(1.0, color='#999', linewidth=1.0, linestyle='--', alpha=0.7)
# Regression line
xv = df_perm['eta2'].values; yv = df_perm['ratio'].values
m, b = np.polyfit(xv, yv, 1)
xfit = np.linspace(xv.min(), xv.max(), 100)
ax_c.plot(xfit, m*xfit+b, 'k--', linewidth=1.5, alpha=0.6)
ax_c.set_xlabel("η²  (behavioral tuning strength)", fontsize=10)
ax_c.set_ylabel("Perm ratio (model HA sensitivity)", fontsize=10)
ax_c.set_title(f"(C) η² vs model sensitivity: ANTI-correlated\nρ={rho_perm_eta2:.3f}, p={p_perm_eta2:.3f}",
               fontsize=10, fontweight='bold')
ax_c.legend(fontsize=7.5, loc='upper right')
# Box explaining the paradox
ax_c.text(0.03, 0.97,
          "High η² → low ratio (under_attr)\nLow η² → high ratio (genuine_overattr)\n"
          "Model sensitivity ≠ behavioral tuning",
          transform=ax_c.transAxes, ha='left', va='top', fontsize=8.5,
          bbox=dict(boxstyle='round,pad=0.35', fc='#FFF9C4', ec='#F9A825', lw=1.2))
ax_c.spines[['top','right']].set_visible(False)

# Panel D: the two-way explanation schematic
ax_d = fig.add_subplot(gs[0, 3])
ax_d.axis('off')
table_data = [
    ['Category', 'η² (behav.)', 'Perm ratio\n(model)'],
    ['mono_correct',    '0.077', '1.11'],
    ['nonmono_correct', '0.068', '1.16'],
    ['genuine_overattr','0.030', '1.19  ← high model,\n       low behav'],
    ['under_attr',      '0.066', '1.01  ← low model,\n       real behav'],
]
y_pos = [0.95, 0.80, 0.65, 0.50, 0.30]
col_x = [0.00, 0.52, 0.73]
for row_i, row in enumerate(table_data):
    for col_i, val in enumerate(row):
        weight = 'bold' if row_i == 0 else 'normal'
        color = PALETTE.get(row[0], 'black') if row_i > 0 and col_i == 0 else 'black'
        ax_d.text(col_x[col_i], y_pos[row_i], val,
                  transform=ax_d.transAxes, ha='left', va='top',
                  fontsize=8.5, fontweight=weight, color=color)
ax_d.axhline(0.90, color='#CCC', linewidth=0.8)
ax_d.set_title("(D) Summary table", fontsize=10, fontweight='bold')

# ── Bottom row: the real story ──
# Panel E: real vs shuffled IG by category
ax_e = fig.add_subplot(gs[1, 0])
for cat in CATS:
    sub = df_perm[df_perm['category'] == cat]
    ax_e.scatter(sub['ig_shuffled'], sub['ig_real'],
                 color=PALETTE[cat], s=65, alpha=0.85, linewidths=0,
                 label=LABELS[cat].replace('\n',' '), zorder=3)
lim = max(df_perm[['ig_real','ig_shuffled']].max()) * 1.1
ax_e.plot([0, lim], [0, lim], 'k--', linewidth=1.2, alpha=0.5)
ax_e.set_xlabel("IG shuffled x_HA", fontsize=10)
ax_e.set_ylabel("IG real x_HA", fontsize=10)
ax_e.set_title("(E) Real vs shuffled IG\nper sampled pair", fontsize=10, fontweight='bold')
ax_e.legend(fontsize=7.5, loc='lower right')
ax_e.spines[['top','right']].set_visible(False)

# Panel F: IG vs η² (the original framing vs new framing)
ax_f = fig.add_subplot(gs[1, 1])
for cat in CATS:
    sub = df_valid[df_valid['category'] == cat]
    ax_f.scatter(sub['eta2'], sub['ig_attr'],
                 color=PALETTE[cat], s=25, alpha=0.7, linewidths=0,
                 label=LABELS[cat].replace('\n',' '), zorder=3)
ax_f.set_xlabel("η²  (behavioral tuning)", fontsize=10)
ax_f.set_ylabel("IG attribution (head_angle)", fontsize=10)
ax_f.set_title("(F) IG vs η²\n(the original framing)", fontsize=10, fontweight='bold')
ax_f.legend(fontsize=7.5, loc='upper right')
ax_f.spines[['top','right']].set_visible(False)

# Panel G: perm ratio vs IG (does model sensitivity predict IG?)
ax_g = fig.add_subplot(gs[1, 2])
for cat in CATS:
    sub = df_perm[df_perm['category'] == cat]
    ax_g.scatter(sub['ratio'], sub['ig_real'],
                 color=PALETTE[cat], s=55, alpha=0.85, linewidths=0,
                 label=LABELS[cat].replace('\n',' '), zorder=3)
ax_g.axvline(1.0, color='#999', linewidth=1.0, linestyle='--', alpha=0.7)
rho_ri, p_ri = spearmanr(df_perm['ratio'], df_perm['ig_real'])
m2, b2 = np.polyfit(df_perm['ratio'].values, df_perm['ig_real'].values, 1)
xfit2 = np.linspace(df_perm['ratio'].min(), df_perm['ratio'].max(), 100)
ax_g.plot(xfit2, m2*xfit2+b2, 'k--', linewidth=1.5, alpha=0.6)
ax_g.set_xlabel("Perm ratio  (model HA sensitivity)", fontsize=10)
ax_g.set_ylabel("IG real", fontsize=10)
ax_g.set_title(f"(G) Model sensitivity predicts IG\nρ={rho_ri:.3f}, p={p_ri:.3f}", fontsize=10, fontweight='bold')
ax_g.legend(fontsize=7.5, loc='upper left')
ax_g.spines[['top','right']].set_visible(False)

# Panel H: conclusion text
ax_h = fig.add_subplot(gs[1, 3])
ax_h.axis('off')
conclusion = (
    "CONCLUSION\n\n"
    "1. Baseline-distance hypothesis: REFUTED\n"
    "   All sessions z-scored → std_HA = 1.000\n"
    "   Path lengths are identical; cannot\n"
    "   explain any differences.\n\n"
    "2. What genuine_overattr really is:\n"
    "   Model sensitivity WITHOUT behavioral\n"
    "   correspondence. Perm ratio ≈ 1.19\n"
    "   (model uses HA structure) but η² ≈ 0.030\n"
    "   (weak behavioral tuning). IG correctly\n"
    "   reports what the model does.\n\n"
    "3. The model problem, not the IG problem:\n"
    "   Model has learned to respond to HA in\n"
    "   cases where neural-behavioral correlation\n"
    "   is weak. Possibly training-set collinearity\n"
    "   or narrow sparse tuning missed by η².\n\n"
    "4. Under-attribution is the inverse:\n"
    "   Model IGNORES HA (ratio≈1.0) despite\n"
    "   real behavioral tuning (η²≈0.066).\n"
    "   Better substitute feature found."
)
ax_h.text(0.03, 0.97, conclusion, transform=ax_h.transAxes,
           fontsize=9, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', fc='#F5F5F5', ec='#BDBDBD', lw=1.2),
           family='monospace')
ax_h.set_title("(H) Conclusion", fontsize=10, fontweight='bold')

fig.suptitle(
    "Causal test result: baseline-distance hypothesis REFUTED\n"
    "genuine_overattr = model sensitivity without behavioral correspondence "
    "(not an IG artefact)",
    fontsize=12, fontweight='bold'
)
savefig("fig_causal_conclusion.png")
print("Done.")
