# -*- coding: utf-8 -*-

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
import logging
import transformers
from accelerate.utils import DistributedType
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import GPTQConfig, deepspeed

from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM
from emu3.train.datasets_HumanEdit import Emu3FeatureDataset

# conda activate emu3
# cd /home/yjoh/project/Emu3-dpo
# sh scripts/t2i_sft_offload.sh

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
    # use_lora: bool = field(default=True) # add
    # cache_dir: Optional[str] = field(default="") # add (TODO)

    
# add
@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


# # add
# def maybe_zero_3(param, ignore_status=False, name=None):
#     from deepspeed import zero
#     from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
#     if hasattr(param, "ds_id"):
#         if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
#             if not ignore_status:
#                 logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
#         with zero.GatheredParameters([param]):
#             param = param.data.detach().cpu().clone()
#     else:
#         param = param.detach().cpu().clone()
#     return param


# # add
# # Borrowed from peft.util.get_peft_model_state_dict
# def get_peft_state_maybe_zero_3(named_params, bias):
#     if bias == "none":
#         to_return = {k: t for k, t in named_params if "lora_" in k}
#     elif bias == "all":
#         to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
#     elif bias == "lora_only":
#         to_return = {}
#         maybe_lora_bias = {}
#         lora_bias_names = set()
#         for k, t in named_params:
#             if "lora_" in k:
#                 to_return[k] = t
#                 bias_name = k.split("lora_")[0] + "bias"
#                 lora_bias_names.add(bias_name)
#             elif "bias" in k:
#                 maybe_lora_bias[k] = t
#         for k, t in maybe_lora_bias:
#             if bias_name in lora_bias_names:
#                 to_return[bias_name] = t
#     else:
#         raise NotImplementedError
#     to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
#     return to_return


# # add
# def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
#     to_return = {k: t for k, t in named_params if "lora_" not in k}
#     if require_grad_only:
#         to_return = {k: t for k, t in to_return.items() if t.requires_grad}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return


# # add
# def safe_save_model_for_hf_trainer(
#     trainer: transformers.Trainer, output_dir: str, bias="none"
# ):
#     """Collects the state dict and dump to disk."""
#     # check if zero3 mode enabled
#     if deepspeed.is_deepspeed_zero3_enabled():
#         state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
#     else:
#         if trainer.args.use_lora:
#             state_dict = get_peft_state_maybe_zero_3(
#                 trainer.model.named_parameters(), bias
#             )
#         else:
#             state_dict = trainer.model.state_dict()
#     if trainer.args.should_save and trainer.args.local_rank == 0:
#         trainer._save(output_dir, state_dict=state_dict)


# # add
# def find_all_linear_names(model):
#     cls = torch.nn.Linear
#     lora_module_names = set()
#     multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
#     for name, module in model.named_modules():
#         if any(mm_keyword in name for mm_keyword in multimodal_keywords):
#             continue
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     if 'lm_head' in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)


# # add
# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#             # print(_)
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )



def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)


def train():
    # global local_rank   

    # parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    # # add
    # if getattr(training_args, "deepspeed", None) and getattr(
    #         lora_args, "q_lora", False
    #     ):
    #         training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    # local_rank = training_args.local_rank
    # device_map = None
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if lora_args.q_lora:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    #     if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
    #         logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")


    # add
    # # Set RoPE scaling factor
    # config = transformers.AutoConfig.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     trust_remote_code=True,
    #     fp32=True,
    # )
    # config.use_cache = False
    # config.embd_pdrop = 0


    # os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")


    # # LoRA 추가
    # lora_config = LoraConfig(
    #     r=8,  # Rank of the LoRA matrix
    #     lora_alpha=16,  # Scaling factor for LoRA
    #     target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific model modules
    #     lora_dropout=0.1,  # Dropout rate for LoRA
    #     task_type=TaskType.CAUSAL_LM  # Task type: Causal Language Modeling
    # )

    model = Emu3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        # cache_dir=training_args.cache_dir, # add
        # device_map='auto',  # add (반드시 주석 처리할 것, 그렇지 않을 시 오류 발생; not compatible with zero3)
        # quantization_config=GPTQConfig(bits=4, disable_exllama=True) # add
        # if training_args.use_lora and lora_args.q_lora
        # else None,
        attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )

    # add
    # if not training_args.use_lora:
    #     if (
    #         training_args.fix_vit
    #         and hasattr(model, "transformer")
    #         and hasattr(model.transformer, "visual")
    #     ):
    #         model.transformer.visual.requires_grad_(False)
    #         if hasattr(model.transformer.visual, "attn_pool"):
    #             model.transformer.visual.attn_pool.requires_grad_(True)
    
    lora_config = LoraConfig(
                    r=lora_args.lora_r,
                    lora_alpha=lora_args.lora_alpha,
                    target_modules=lora_args.lora_target_modules, # lora_target_modules,
                    lora_dropout=lora_args.lora_dropout,
                    bias=lora_args.lora_bias,
                    task_type="CAUSAL_LM",
                    # modules_to_save=None,  # This argument serves for adding new tokens.
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
        # cache_dir=training_args.cache_dir, # add
    )


    # add
    # if training_args.use_lora:
    #     if lora_args.lora_target_modules == "all-linear":
    #         lora_target_modules = find_all_linear_names(model)
    #     elif "," in lora_args.lora_target_modules:
    #         lora_target_modules = lora_args.lora_target_modules.split(",")
    #     else:
    #         lora_target_modules = lora_args.lora_target_modules

    #     lora_config = LoraConfig(
    #         r=lora_args.lora_r,
    #         lora_alpha=lora_args.lora_alpha,
    #         target_modules=lora_args.lora_target_modules, # lora_target_modules,
    #         lora_dropout=lora_args.lora_dropout,
    #         bias=lora_args.lora_bias,
    #         task_type="CAUSAL_LM",
    #         # modules_to_save=None,  # This argument serves for adding new tokens.
    #     )
    #     if lora_args.q_lora:
    #         model = prepare_model_for_kbit_training(
    #             model, use_gradient_checkpointing=training_args.gradient_checkpointing
    #         )

    #     if training_args.gradient_checkpointing:
    #         model.enable_input_require_grads()

    train_dataset = Emu3FeatureDataset(data_args, tokenizer=tokenizer)

    # add
    # if training_args.use_lora:
    #     model = get_peft_model(model, lora_config)

    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # trainer = tf.Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     peft_config=lora_config if training_args.use_lora else None, # add
    # )

    
    # trainer = PeftTrainer(
    # model=model,
    # args=training_args,
    # train_dataset=train_dataset,
    # peft_config=lora_config if training_args.use_lora else None,
    # )
    # print_trainable_parameters(model)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.save_pretrained(training_args.output_dir) # add for LoRA adapter

    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)

    # add
    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                 output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
