#!/bin/bash
# CoT-assisted Q-SFT Training Script
set -euo pipefail
nproc_per_node=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=$((20000 + RANDOM % 20000))
export MASTER_ADDR=127.0.0.1
RDZV_ID="qsft_cot_${USER}_$RANDOM"
echo "[train_qsft_cot] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
torchrun --nproc_per_node "${nproc_per_node}" --rdzv_backend c10d --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" --rdzv_id "${RDZV_ID}" train_qsft_cot.py --model Qwen/Qwen2.5-VL-7B-Instruct --train_type lora --dataset my_training_data_qsft --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 --learning_rate 1e-4 --lora_rank 32 --lora_alpha 64 --use_ummcot true --q_gamma 0.95 --lambda_cot 0.5 --lambda_align 0.1 --label_smoothing 0.1 --clip_weight 5.0 --ema_decay 0.995
