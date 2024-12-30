from dataclasses import dataclass, field
import os
import os.path as osp
import pathlib
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.distributed
import transformers as tf

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from peft import get_peft_model, LoraConfig, TaskType
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from datasets import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl.trainer.dpo_config import DPOConfig, FDivergenceType, FDivergenceConstants

from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM
from emu3.dpo.dpo_datasets import PreferenceHumanEditDatasetEmu3
from emu3.dpo.data_collator import DPODataCollatorEmu3
from emu3.dpo.dpo_trainer import DPOTrainerEmu3


# conda activate emu3_dpo
# cd /home/yjoh/project/Emu3-dpo
# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=7 ./emu3/dpo/scripts/t2i_dpo_offload.sh
# chmod +x ./emu3/dpo/scripts/t2i_dpo_offload.sh

# TODO: LoRA arguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")


@dataclass
class DataArguments:
    train_data_path: Optional[str] = field(default=None)
    val_data_path: Optional[str] = field(default=None) # add
    # truncation_mode: str = field(default="keep_end") # add (중복 w/ TrainingArgs)
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
    # dpo 
    # model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
    # ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Literal["sigmoid", "hinge", "ipo", "bco_pair", "robust", "aot", "aot_pair"] = "sigmoid"
    args: Optional[DPOConfig] = None
    data_collator: Optional[DataCollator] = None
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    train_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    model_init: Optional[Callable[[], PreTrainedModel]] = None
    callbacks: Optional[List[TrainerCallback]] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    max_length: Optional[int] = None # Emu3 에서는 실질적으로 의미 없음
    max_prompt_length: Optional[int] = None # Emu3 에서는 실질적으로 의미 없음
    max_target_length: Optional[int] = None
    peft_config: Optional[Dict] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    use_lora: bool = True
    f_divergence_type: Optional[FDivergenceType] = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: Optional[float] = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None


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

    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")
    

    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    if training_args.logging_dir is None:
        training_args.logging_dir = os.path.join(training_args.output_dir, "./logs")
    print(f"\n# Logging to: {training_args.report_to}")
    print(f"# Logging Dir: {training_args.logging_dir}\n")
    


    # # LoRA 추가
    # lora_config = LoraConfig(
    #     r=8,  # Rank of the LoRA matrix
    #     lora_alpha=16,  # Scaling factor for LoRA
    #     target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific model modules
    #     lora_dropout=0.1,  # Dropout rate for LoRA
    #     task_type=TaskType.CAUSAL_LM  # Task type: Causal Language Modeling
    # )

    # DPO TRAINER INITIALIZE THE MODEL IN ITS TRAINER CLASS, NOT HERE.
    
    model = Emu3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )


    # # Apply LoRA
    # # (수정) /root/anaconda3/lib/python3.12/site-packages/deepspeed/runtime/zero/stage3.py
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()
    # else:
    #     def make_inputs_require_grad(module, input, output):
    #         output.requires_grad_(True)

    #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()  # Optional: View trainable parameters
    # # Ensure trainable parameters require gradients
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable parameter: {name}")


    if training_args.use_lora:
        lora_config = LoraConfig(
        r=8,  # Rank of the LoRA matrix
        lora_alpha=16,  # Scaling factor for LoRA
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific model modules
        lora_dropout=0.1,  # Dropout rate for LoRA
        task_type=TaskType.CAUSAL_LM  # Task type: Causal Language Modeling
        )

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()


    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )


    EMU_HUB = "BAAI/Emu3-Gen"
    VQ_HUB = "BAAI/Emu3-VisionTokenizer"
    # tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    # image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    # image_tokenizer = AutoModel.from_pretrained(VQ_HUB, trust_remote_code=True).eval()
    # processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    train_dataset = PreferenceHumanEditDatasetEmu3(data_args, tokenizer=tokenizer, split="train") 
    train_dataset_hf = Dataset.from_list(train_dataset.to_list())
    
    val_dataset = PreferenceHumanEditDatasetEmu3(data_args, tokenizer=tokenizer, split="val")
    val_dataset_hf = Dataset.from_list(val_dataset.to_list())

    # trainer = tf.Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    # )


    trainer = DPOTrainerEmu3(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset_hf,
        eval_dataset=val_dataset_hf,
        data_collator=DPODataCollatorEmu3( 
            model,
            tokenizer,
            # image_tokenizer=image_tokenizer,
            args=data_args
        ),
        peft_config=lora_config if training_args.use_lora else None
    )


    # print(f"\n# Logging to: {training_args.report_to}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.save_pretrained(training_args.output_dir) # add for LoRA adapter

    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


    
    # # Define trainner
    # trainer = mDPOTrainer(
    #     model,
    #     args=training_args,
    #     beta=training_args.beta,
    #     train_dataset=train_dataset,
    #     # eval_dataset=eval_dataset,
    #     data_collator=mDPODataCollatorLLaMA( # 수정 파트 2
    #         tokenizer,
    #         model,
    #         label_pad_token_id=-100, # LabelSmoother.ignore_index, # hard-coding
    #         padding_value=tokenizer.pad_token_id,
    #         truncation_mode="keep_end",
    #     ),
    #     tokenizer=tokenizer,
    #     max_length=training_args.model_max_length,
    #     peft_config=lora_config if training_args.use_lora else None,
    # )

    # print_trainable_parameters(model)

    # trainer.train(resume_from_checkpoint=False)
    # trainer.save_state()

    # model.config.save_pretrained(training_args.output_dir)
    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                 output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
