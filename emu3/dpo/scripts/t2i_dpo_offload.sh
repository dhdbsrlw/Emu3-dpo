#!/bin/bash

WORLD_SIZE=${WORLD_SIZE:-1} # num_gpus
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-25000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} # Default to GPU 0
# NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# export CUDA_VISIBLE_DEVICES=0,1 # num_gpus
export PYTHONPATH=$(pwd)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 필수
export TORCH_USE_CUDA_DSA=1

# --nproc_per_node=${NGPUS} \

# cmd
# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=5 ./t2i_dpo_offload.sh
# Token indices sequence length is longer than the specified maximum sequence length for this model (8210 > 5120). Running this sequence through the model will result in indexing errors

# .json
# TRAIN_DATAPATH="/nas2/preference/emu3_tokenized/human_edit_train/list/train.json"
# VAL_DATAPATH="/nas2/preference/emu3_tokenized/human_edit_val/list/train.json"

TRAIN_DATAPATH="/nas2/preference/emu3_tokenized/human_edit_train_256/list/train.json"
VAL_DATAPATH="/nas2/preference/emu3_tokenized/human_edit_val_256/list/train.json"
EXP_NAME="1229_Emu3_T2I_DPO_HumanEdit_256_debug"

torchrun \
    --nproc_per_node=${WORLD_SIZE} \
    --nnodes=1 \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    emu3/dpo/run_dpo_Emu3.py \
    --model_name_or_path BAAI/Emu3-Gen \
    --deepspeed emu3/dpo/scripts/zero3_offload.json \
    --train_data_path ${TRAIN_DATAPATH} \
    --val_data_path ${VAL_DATAPATH} \
    --null_prompt_prob 0.00 \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_text False \
    --image_area 65536 \
    --max_position_embeddings 4200 \
    --output_dir "/home/yjoh/project/Emu3-dpo/emu3/dpo/results/"${EXP_NAME} \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
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
    --report_to wandb \
    --run_name ${EXP_NAME} \
    --beta 0.1 \
