#!/bin/bash

# WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-25000}
# set with CUDA_VISIBLE_DEVICES
# NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --nproc_per_node=${NGPUS} \

DATAPATH="/nas2/preference/emu3_tokenized/magicbrush_train/list/train.json"
EXP_NAME="1226-Emu3-T2I-SFT-MagicBrush"
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    emu3/train/train.py \
    --model_name_or_path BAAI/Emu3-Gen \
    --deepspeed scripts/zero3_offload.json \
    --data_path ${DATAPATH} \
    --null_prompt_prob 0.05 \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_text False \
    --image_area 262144 \
    --max_position_embeddings 5120 \
    --output_dir "/nas2/checkpoints/emu3_dpo/"${EXP_NAME} \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --min_learning_rate 1e-6 \
    --weight_decay 0.1 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --warmup_steps 30 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name ${EXP_NAME}
