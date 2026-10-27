#!/bin/bash
set -e
cd "$(dirname "$0")"

mkdir -p splits

train_seed() {
    local seed=$1
    local split="splits/split_seed${seed}.npy"

    if [ ! -f "$split" ]; then
        echo "=== Seed $seed: generating split ==="
        python utils/generate_split.py --out "$split" --seed "$seed"
    else
        echo "=== Seed $seed: reusing existing split $split ==="
    fi

    echo "=== Seed $seed: training CEBRA contrastive (ensembles) ==="
    python training/train_cebra.py \
        --split_path "$split" \
        --seed "$seed" \
        --use_ensembles \
        --models_dir "./models/cebra/ensembles/seed${seed}"

    echo "=== Seed $seed: training CEBRA predictive (ensembles) ==="
    python training/train_cebra_non_contrastive.py \
        --split_path "$split" \
        --seed "$seed" \
        --use_ensembles \
        --models_dir "./models/cebra_pred/ensembles/seed${seed}"
}

export -f train_seed

seeds=(42 43 44 45 46)
for seed in "${seeds[@]}"; do
    train_seed "$seed"
done

echo "Done. Trained 10 CEBRA models (5 contrastive + 5 predictive, ensembles, seeds 42-46)."
