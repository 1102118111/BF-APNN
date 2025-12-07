#!/bin/bash
# -*- coding: utf-8 -*-

exp_id=1

run_experiment () {
    local model_type=$1
    local embed_size=$2
    local seed=$3

    log_file="exp_id=${exp_id}_model_${model_type}_embed_size_${embed_size}_seed${seed}.log"

    echo "Running: model_type=${model_type}, embed_size=${embed_size}, seed=${seed}"

    nohup python -u train.py \
        --exp_id "${exp_id}" \
        --embed_size "${embed_size}" \
        --model_type "${model_type}" \
        --seed "${seed}" \
        > "${log_file}" 2>&1

    if [ $? -ne 0 ]; then
        echo "ERROR: exp_id=${exp_id} model=${model_type} embed_size=${embed_size} failed. Stopping script."
        exit 1
    fi

    echo "DONE: exp_id=${exp_id} model=${model_type} embed_size=${embed_size} completed."
}


# ============================
#   Run model_type = 0
# ============================
for seed in 168; do
    run_experiment 0 128 "${seed}"
done

# ============================
#   Run model_type = 1
# ============================
for seed in 168; do
    run_experiment 1 12 "${seed}"
done

# ============================
#   Run model_type = 2
# ============================
for seed in 168; do
    run_experiment 2 16 "${seed}"
done

# ============================
#   Run model_type = 3 
# ============================
for seed in 168; do
    model_type=3
    log_file="exp_id=${exp_id}_model_${model_type}_seed${seed}.log"

    echo "Running: model_type=3, seed=${seed}"

    nohup python -u train.py \
        --exp_id "${exp_id}" \
        --model_type 3 \
        --seed "${seed}" \
        > "${log_file}" 2>&1

    if [ $? -ne 0 ]; then
        echo "ERROR: exp_id=${exp_id} model=3 failed. Stopping script."
        exit 1
    fi

    echo "DONE: exp_id=${exp_id} model=3 completed."
done

echo "All experiments have finished successfully!"
