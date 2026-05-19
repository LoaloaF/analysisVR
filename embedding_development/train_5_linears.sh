#!/bin/bash
set -e
cd "$(dirname "$0")"

mkdir -p splits

for seed in 42 43 44 45 46; do
    split="splits/split_seed${seed}.npy"

    if [ -f "$split" ]; then
        echo "=== Seed $seed: split already exists, skipping generation ==="
    else
        echo "=== Seed $seed: generating split ==="
        python utils/generate_split.py --out "$split" --seed "$seed"
    fi

    echo "=== Seed $seed: training linear (spikes) ==="
    python training/train_linear.py \
        --split_path "$split" \
        --seed "$seed" \
        --models_dir "./models/linear/spikes/seed${seed}"

    echo "=== Seed $seed: training linear (ensembles) ==="
    python training/train_linear.py \
        --split_path "$split" \
        --seed "$seed" \
        --use_ensembles \
        --models_dir "./models/linear/ensembles/seed${seed}"
done

echo "Done. Trained 10 linear models (5 spikes + 5 ensembles, seeds 42-46)."
