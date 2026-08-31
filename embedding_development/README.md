# Ensemble-Based Action Embeddings

Code for building and evaluating **neurally-informed behavioral embeddings**: neural-network
encoders that predict neural activity (single units and ICA ensembles) from behavioral
state-actions, i.e. `P(x | s, a)`. Companion to the thesis *"Developing Ensemble-Based Action
Embeddings using Neural Network Based Encoding Models."*

The pipeline has three stages:

```
utils/load_data.ipynb   →  outputs/glm_input_data/*.npy     (1. collect)
utils/generate_split.py →  splits/split_seedNN.npy          (2. split)
training/train_*.py     →  models/<model>/<target>/seedNN/  (3. train)
eval/eval_*.py, gen_*.py →  outputs/... + figures           (4. evaluate)
```

All commands are run from the repository root (`embedding_development/`) with the project
environment active:

```bash
conda activate analysisVR
```

---

## 1. Collect data

> Requires access to the lab **analytics backend** (the parent `analysisVR` processing
> pipeline and its database). This step is only reproducible inside the lab.

Run **`utils/load_data.ipynb`**. It pulls the aligned behavioral and neural recordings via
`analytics.get_analytics(...)`, aligns them to 40 ms bins, and writes the model inputs to
`outputs/glm_input_data/`:

| File (`outputs/glm_input_data/`) | Contents |
|---|---|
| `behavior_glm_input.npy` (+ `_index`, `_columns`) | Behavioral state-action features per 40 ms bin (the model input) |
| `fr_full.npy` (+ `_index`, `_columns`) | Single-unit firing rates (predictive targets) |
| `spikes.npy` (+ `_index`, `_columns`) | Single-unit spike counts (targets for the Poisson-GLM baseline) |
| `ensembles.npy` | ICA ensemble weight matrix (neurons × 23 ensembles) |

Ensemble activity is not stored directly; it is formed at train/eval time as the linear
projection `spikes_values @ ensembles_values[:, i]` and then z-scored (see `training/train_*.py`).

---

## 2. Make train/test splits

Each split holds out ~20% of **whole trials** per session (so no trial is split across
train/test). One file per random seed:

```bash
python utils/generate_split.py --out splits/split_seed42.npy --seed 42
```

Results in the thesis use seeds **42–46**.

---

## 3. Train encoders

Each script trains a per-session encoder for a given seed, targeting either single units
(default) or ensembles (`--use_ensembles`).

| Model | Script | Notes |
|---|---|---|
| Linear | `training/train_linear.py` | affine baseline `Wq + b` |
| MLP | `training/train_mlp.py` | 2 × 64 hidden, 64-dim embedding |
| TempConv-Cont (contrastive, InfoNCE) | `training/train_cebra.py` | CEBRA `offset10` encoder |
| TempConv-Pred (predictive, MSE) | `training/train_cebra_non_contrastive.py` | same encoder + affine head |

Common arguments: `--split_path`, `--seed`, `--use_ensembles`, `--models_dir`. The TempConv
scripts additionally accept `--embed_dim`, `--num_units`, `--batch_size`, `--learning_rate`.

Single run (one MLP on ensembles, seed 42):

```bash
python training/train_mlp.py --split_path splits/split_seed42.npy --seed 42 \
    --use_ensembles --models_dir ./models/mlps/ensembles/seed42
```

Full multi-seed sets (seeds 42–46, both single units and ensembles) via the wrappers:

```bash
bash train_5_linears.sh
bash train_5_mlps.sh
bash train_5_cebras.sh
bash training/retrain_tempconv_64d.sh    # 64-dim TempConv-Cont / -Pred variants
```

Checkpoints are written to `models/<model>/<spikes|ensembles>/seed<NN>/`.

---

## 4. Evaluate

Evaluation scripts load the trained checkpoints, run inference on the held-out trials, and
write metric arrays and figures under `outputs/`. Run any of them from the repo root, e.g.
`python eval/gen_nonlinear_motivation.py`. Key entry points:

**Core metric arrays**
- `eval/eval_mlp_attribution.py` — per-(session, ensemble) held-out R² (`all_r2.npy`) and
  feature attribution: Integrated Gradients, global and conditional permutation variance
  (`importance_ig_semantic.npy`, `importance_global_pv_semantic.npy`,
  `importance_cond_pv_semantic.npy`) → `outputs/mlps/ensembles_multiseed/`
- `eval/eval_cebra_seeds.py` — the same for the TempConv (contrastive / predictive) variants
- `eval/eval_ml_vs_naive.py` — naive per-feature Spearman |ρ| and η² (`naive_importance.npy`,
  `eta2.npy`) → `outputs/mlps/ml_vs_naive/`

**Consistency / cross-prediction**
- `eval/eval_embedding_linear_map.py` — linear-map (ridge) R² of embeddings across seeds,
  sessions, ensembles, and architectures
- `eval/eval_cross_ensemble_prediction.py` — predicting one ensemble's activity from another's
  embedding

**Figures**
- `eval/gen_nonlinear_motivation.py` — best η² vs best Pearson r² (why nonlinearity is needed)
- `eval/gen_glm_vs_mlp_figure.py` — MLP vs Poisson GLM on single units, split by monotone /
  non-monotone tuning
- `eval/gen_tuning_shape_examples.py`, `eval/gen_head_angle_attribution_summary.py` —
  head-angle tuning shapes and attribution
- `eval/honest_attribution_example.py` — correlation-vs-attribution rank reversal (position case study)
- `eval/gen_embedding_space.py` — TempConv-Pred vs -Cont embedding geometry
- `eval/eval_temporal_advantage.py` — fast/slow (high-frequency) component split

**Validation case studies**
- `eval/gen_e07_e23_ig.py`, `eval/eval_e07_cue_zone_attribution.py` — recovery of known task
  variables (cue visibility, upcoming choice)
- `eval/eval_position_ablation.py` — position-only retraining ablation

Figure scripts save into `outputs/` (a few also mirror to a local path via an `OUT_DIRS`
list at the top of the script — edit that if you want them elsewhere).

---

## Repository layout

```
training/   encoder training scripts + multi-seed wrappers
eval/       evaluation and figure-generation scripts
utils/      data loading (load_data.ipynb), split generation, model defs, plot style
models/     trained checkpoints              (gitignored)
outputs/    datasets, metric arrays, figures (gitignored)
splits/     per-seed train/test splits       (gitignored)
docs/       notes
```

`models/`, `outputs/`, and `splits/` are gitignored — they are regenerated by the pipeline
above. The feature set and ensemble-detection procedure are documented in the thesis Methods
chapter.
