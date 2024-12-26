# -*- coding: utf-8 -*-

# tmux session 3 에서 학습
#  Run data is saved locally in /tmp/wandb/run-20241226_083904-i69bqrc3

from dataclasses import dataclass, field
import os
import os.path as osp
import pathlib
from typing import Optional, List

import transformers as tf
import torch

# add
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from peft import get_peft_model, LoraConfig, TaskType
from peft import LoraConfig

from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM
from emu3.train.datasets_HumanEdit import Emu3FeatureDataset


# conda activate emu3_dpo
# cd /home/yjoh/project/Emu3-dpo
# sh /home/yjoh/project/Emu3-dpo/scripts/t2i_sft_offload.sh


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)


@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)

def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)


def train():

    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    # model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")
    

    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    if training_args.logging_dir is None:
        training_args.logging_dir = os.path.join(training_args.output_dir, "./logs")
    print(f"\n# Logging to: {training_args.report_to}")
    print(f"\n# Logging Dir: {training_args.logging_dir}\n")
    


    # LoRA 추가
    lora_config = LoraConfig(
        r=8,  # Rank of the LoRA matrix
        lora_alpha=16,  # Scaling factor for LoRA
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific model modules
        lora_dropout=0.1,  # Dropout rate for LoRA
        task_type=TaskType.CAUSAL_LM  # Task type: Causal Language Modeling
    )

    model = Emu3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )


    # Apply LoRA
    # (수정) /root/anaconda3/lib/python3.12/site-packages/deepspeed/runtime/zero/stage3.py
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Optional: View trainable parameters
    # Ensure trainable parameters require gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}")


    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )


    train_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer)

    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"\n# Logging to: {training_args.report_to}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.save_pretrained(training_args.output_dir) # add for LoRA adapter

    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
