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

    echo "=== Seed $seed: training MLP (spikes) ==="
    python training/train_mlp.py \
        --split_path "$split" \
        --seed "$seed" \
        --models_dir "./models/mlps/spikes/seed${seed}"

    echo "=== Seed $seed: training MLP (ensembles) ==="
    python training/train_mlp.py \
        --split_path "$split" \
        --seed "$seed" \
        --use_ensembles \
        --models_dir "./models/mlps/ensembles/seed${seed}"
}

export -f train_seed

seeds=(42 43 44 45 46)
for ((i=0; i<${#seeds[@]}; i+=2)); do
    train_seed "${seeds[i]}" &
    if [ $((i+1)) -lt ${#seeds[@]} ]; then
        train_seed "${seeds[i+1]}" &
    fi
    wait
done

echo "Done. Trained 10 MLPs (5 spikes + 5 ensembles, seeds 42-46)."
