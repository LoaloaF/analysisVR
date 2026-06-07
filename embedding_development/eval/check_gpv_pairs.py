#!/usr/bin/env python3
import numpy as np, pickle

with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

mlp_gpv  = np.load('../outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy')
ceb_gpv  = np.load('../outputs/cebra_eval/ensembles/importance_global_pv_semantic.npy')
pred_gpv = np.load('../outputs/cebra_pred_eval/ensembles/importance_global_pv_semantic.npy')
ceb_ig   = np.load('../outputs/cebra_eval/ensembles/importance_ig_semantic.npy')
mlp_ig   = np.load('../outputs/mlps/ensembles_multiseed/importance_ig_semantic.npy')

mlp_r2_all  = np.load('../outputs/mlps/ensembles_multiseed/all_r2.npy')
ceb_r2_all  = np.load('../outputs/cebra_eval/ensembles/all_r2.npy')
pred_r2_all = np.load('../outputs/cebra_pred_eval/ensembles/all_r2.npy')

R2_THRESH = 0.01
shared = (
    (~np.all(np.isnan(mlp_r2_all),  axis=0)) & (np.nanmean(mlp_r2_all,  axis=0) >= R2_THRESH) &
    (~np.all(np.isnan(ceb_r2_all),  axis=0)) & (np.nanmean(ceb_r2_all,  axis=0) >= R2_THRESH) &
    (~np.all(np.isnan(pred_r2_all), axis=0)) & (np.nanmean(pred_r2_all, axis=0) >= R2_THRESH)
)

def norm_pp(arr, mask):
    out = arr.copy()
    for s in range(arr.shape[0]):
        for n in range(arr.shape[1]):
            if mask[s, n]:
                mx = np.nanmax(arr[s, n])
                if mx > 0: out[s, n] /= mx
    return out

ci_n = norm_pp(ceb_ig, shared)
mi_n = norm_pp(mlp_ig, shared)

g_rota = group_names.index('frame_YawPitch_abs_acc_sum_500msMedian')
g_hvel = group_names.index('head_angle_vel')
g_hang = group_names.index('head_angle')
g_uc   = group_names.index('upcoming_choice')

rota_pairs = [(1,5),(1,7),(11,7),(24,16),(9,7),(8,1),(28,6),(25,1)]
hvel_pairs = [(12,13),(11,9),(8,9),(9,9),(10,9),(8,19)]
hang_pairs = [(0,2),(0,1),(0,3),(1,2),(2,1)]
uc_pairs   = [(15,16),(16,4)]

hdr = ('Pair       ceb_ig_n  mlp_ig_n  ceb_gpv   mlp_gpv   pred_gpv  '
       'ceb_r2  mlp_r2')
print('='*len(hdr))
print(hdr)
print('='*len(hdr))

for label, pairs, g_idx in [
    ('--- rot_acc ---',          rota_pairs, g_rota),
    ('--- head_vel ---',         hvel_pairs, g_hvel),
    ('--- head_ang (baseline) ---', hang_pairs, g_hang),
    ('--- upcoming_choice ---',  uc_pairs,   g_uc),
]:
    print(label)
    for s_idx, n_idx in pairs:
        if not shared[s_idx, n_idx]:
            continue
        cig = ci_n[s_idx, n_idx, g_idx]
        mig = mi_n[s_idx, n_idx, g_idx]
        cg  = ceb_gpv[s_idx,  n_idx, g_idx]
        mg  = mlp_gpv[s_idx,  n_idx, g_idx]
        pg  = pred_gpv[s_idx, n_idx, g_idx]
        cr2 = float(np.nanmean(ceb_r2_all[:, s_idx, n_idx]))
        mr2 = float(np.nanmean(mlp_r2_all[:, s_idx, n_idx]))
        print(f'  S{s_idx:02d}E{n_idx:02d}  '
              f'{cig:8.4f}  {mig:8.4f}  '
              f'{cg:9.5f}  {mg:9.5f}  {pg:9.5f}  '
              f'{cr2:.4f}  {mr2:.4f}')

print()
print('Global group means (across all shared valid pairs):')
print(f'  {"group":<42}  ceb_gpv   mlp_gpv   pred_gpv')
for g_idx, g in enumerate(group_names):
    cg = float(np.nanmean(ceb_gpv[shared, g_idx]))
    mg = float(np.nanmean(mlp_gpv[shared, g_idx]))
    pg = float(np.nanmean(pred_gpv[shared, g_idx]))
    print(f'  {g:<42}  {cg:.5f}   {mg:.5f}   {pg:.5f}')
