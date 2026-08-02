"""
GLM vs MLP on single-unit spike counts.

Fits a multivariate Poisson GLM (all 17 behavioral features, same train/test
split) per (session, unit) predicting raw spike counts, and scores it with
held-out R2 on the count scale -- directly comparable to the MLP and linear
spike R2 already stored in outputs/{mlps,linear}/spikes_multiseed/all_r2.npy.

Shows that even a proper (nonlinear-link) GLM underperforms the MLP.

Outputs:
  outputs/glm_results/glm_vs_mlp_spikes.npz   (per session,unit R2 arrays)
  outputs/glm_results/glm_vs_mlp_spikes.png   (paired scatter + summary)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

warnings.simplefilter("ignore")
os.chdir("/home/amitsant2000/ethz/VirtualReality/analysisVR/embedding_development")
sys.path.insert(0, os.path.abspath("."))

B = "outputs/glm_input_data/"
vals = np.load(B + "behavior_glm_input.npy", allow_pickle=True)
cols = list(np.load(B + "behavior_glm_input_columns.npy", allow_pickle=True))
idx  = np.load(B + "behavior_glm_input_index.npy", allow_pickle=True)
fr   = np.load(B + "fr_full.npy")                      # (T, 77) int counts
split = np.load("test_indices_by_session.npy", allow_pickle=True).item()

CONT = ['frame_raw_500msMedian', 'frame_raw_abs_acc_500msMedian',
        'frame_YawPitch_abs_vel_sum_500msMedian', 'frame_YawPitch_abs_acc_sum_500msMedian',
        'head_angle_vel', 'head_angle', 'frame_position']
CAT  = ['cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected']

sess_name = np.array([t[0] for t in idx])
trial_id  = pd.to_numeric(pd.Series(vals[:, cols.index('trial_id')]), errors='coerce').values
Xcont_all = np.column_stack([pd.to_numeric(pd.Series(vals[:, cols.index(c)]), errors='coerce').values for c in CONT])
Xcat_all  = {c: pd.to_numeric(pd.Series(vals[:, cols.index(c)]), errors='coerce').values for c in CAT}

feat_nan = np.isnan(Xcont_all).any(axis=1) | np.isnan(trial_id)
for c in CAT:
    feat_nan |= np.isnan(Xcat_all[c])
valid_row = ~feat_nan

session_order = pd.unique(sess_name[valid_row])          # appearance order (matches MLP axis)
n_sess, n_unit = len(session_order), fr.shape[1]
print(f"{n_sess} sessions, {n_unit} units")

mlp = np.nanmean(np.load("outputs/mlps/spikes_multiseed/all_r2.npy"), axis=0)   # (29,77)
lin = np.nanmean(np.load("outputs/linear/spikes_multiseed/all_r2.npy"), axis=0)

glm_r2 = np.full((n_sess, n_unit), np.nan)

def build_X(rows):
    xc = Xcont_all[rows]
    xc = (xc - xc.mean(0)) / (xc.std(0) + 1e-8)
    parts = [xc]
    for c in CAT:
        v = Xcat_all[c][rows]
        u = np.unique(v).astype(int)
        if len(u) <= 1:
            continue
        levels = u[1:]                         # reference-code (drop first)
        parts.append(np.column_stack([(v == lv).astype(float) for lv in levels]))
    return sm.add_constant(np.column_stack(parts), has_constant='add')

for s_i, s in enumerate(session_order):
    rows = np.where(valid_row & (sess_name == s))[0]
    X = build_X(rows)
    tr = trial_id[rows]
    test_tr = set(np.asarray(split[s], dtype=float))
    te_mask = np.array([t in test_tr for t in tr])
    tr_mask = ~te_mask
    if te_mask.sum() < 20 or tr_mask.sum() < 50:
        continue
    Xtr, Xte = X[tr_mask], X[te_mask]
    Ys = fr[rows]
    n_fit = 0
    for u in range(n_unit):
        if np.isnan(mlp[s_i, u]):
            continue
        ytr, yte = Ys[tr_mask, u].astype(float), Ys[te_mask, u].astype(float)
        if ytr.mean() < 0.1:
            continue
        try:
            res = sm.GLM(ytr, Xtr, family=sm.families.Poisson()).fit(maxiter=100)
            pred = res.predict(Xte)
            if np.all(np.isfinite(pred)):
                glm_r2[s_i, u] = r2_score(yte, pred)
                n_fit += 1
        except Exception:
            pass
    print(f"  [{s_i+1:2d}/{n_sess}] {s}: fit {n_fit} units")

np.savez("outputs/glm_results/glm_vs_mlp_spikes.npz",
         glm_r2=glm_r2, mlp_r2=mlp, lin_r2=lin, sessions=session_order)

# ---- summary ----
def summ(name, a, ref):
    m = a[np.isfinite(a) & np.isfinite(ref)]
    print(f"{name:8s} median={np.median(m):+.4f}  mean={m.mean():+.4f}  "
          f">=0.01: {(m>=0.01).mean():5.1%}  >=0.05: {(m>=0.05).mean():5.1%}  n={m.size}")

paired = np.isfinite(glm_r2) & np.isfinite(mlp)
print("\n=== paired (session,unit) where both GLM and MLP valid ===")
summ("GLM",  glm_r2, mlp)
summ("MLP",  mlp,    glm_r2)
summ("Linear", lin,  glm_r2)
print(f"MLP > GLM in {(mlp[paired] > glm_r2[paired]).mean():.1%} of units")
print("done")
