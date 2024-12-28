
### Conda env (agilab2)
- L40S 40G * 8대
```
conda activate emu3_dpo
pip uninstall -y deepspeed && pip install deepspeed==0.15.4
```

### TODO

- [X] Clone repository.
- [X] Prepare data with HumanEdit dataset. 
- [X] Train official code (SFT) with HumanEdit dataset.
- [ ] Eval SFT Model for MagicBrush (Edit) train/test dataset (L1, CLIP-T)
- [ ] Eval Original Model for MagicBrush (Edit) train/test dataset (L1, CLIP-T)
- [ ] Implement DPOTrainer with Emu3-Gen.
- [ ] Train DPO code with HumanEdit dataset.


### [issue 1] OOM in Emu3 PEFT Training
(오피셜 코드 대비 수정사항 기록)
#### 1. t2i_sft_offload.sh 
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `--image_area 262144`
- `--max_position_embeddings 5120` 
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 8` 

#### 2. zero3_offload.json
- `sub_group_size`
- `reduce_bucket_size`
- `stage3_prefetch_bucket_size`
- `stage3_max_live_parameters`
- `stage3_max_reuse_distance`
- `pin_memory`

#### 3. train.py
1. add LoRA adapter