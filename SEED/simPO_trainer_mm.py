""" This code is based on the official SimPO implementation with trl (https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py) """
# README: This is for Training with Multi-Modal preference dataset.
# README: This is based SEED-LLaMA model.

# (9/30) validation 수정 - 여러 종류의 Val dataset 합하기

# conda activate seed2
# cd /home/yjoh/project/MAGVLT2/MultiModalLLM

import os
import sys
import numpy as np
import logging
import warnings
import argparse
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Tuple, Union
import pdb

import torch
import torch.nn as nn  
import torch.distributed as dist
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler

import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.strategies import DDPStrategy

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model
from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length 
from utils.config import build_config
from lr_scheduler import CosineDecayWarmUpRestarts
from models.seed_llama_tokenizer import SeedLlamaTokenizer
from train_rf_main_6_util import _load_visual_tokenizer, _save_config
# from simPO_dataset_mm import SimPODataset_MM 
from simPO_dataset_mm_2025 import SimPODataset_MM 
# from simPO_dataset_mm_datasampling import SimPODataset_MM

logger = logging.getLogger(__name__)


class SimPODataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = []
        self.val_dataset = []
    
    def setup(self):
        # train set
        data_dirs = self.config.dataset.train.data_dir             
        sampling_rates = self.config.dataset.train.sampling_rate   
        assert len(data_dirs) == len(sampling_rates), "The number of data and sampling rate must be same !"

        for data, sampling in zip(data_dirs, sampling_rates):
            assert 0.0 <= sampling <= 1.0, "Sampling rate must be between 0 and 1."
            try:
                self.train_dataset.append(SimPODataset_MM(data_dir=data, sampling_rate=sampling, tokenizer=self.tokenizer, tokenizer_cfg=self.config['tokenizer']))
            except:
                raise ValueError(f"All dataset should generated in torchData.Dataset. `{data}` raised ERROR.")

        self.data_collator = DPODataCollatorWithPadding(pad_token_id=self.tokenizer.pad_token_id, 
                                                        label_pad_token_id=self.config.tokenizer.label_pad_token_id)
        
        # val set
        # self.val_dataset = CocoDataset(self.config.dataset['val'], train_type='sft', tokenizer=self.tokenizer)
        self.val_dataset.append(SimPODataset_MM(data_dir=self.config.dataset.val.data_dir[0], sampling_rate=self.config.dataset.val.sampling_rate[0], tokenizer=self.tokenizer, tokenizer_cfg=self.config['tokenizer']))
        # TODO: list 하드코딩 수정

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.config.dataset.per_device_train_batch_size, shuffle=True, num_workers=self.config.dataset.preprocessing_num_workers, collate_fn=self.data_collator)
        return DataLoader(
                dataset=ConcatDataset(self.train_dataset), 
                sampler=RandomSampler(ConcatDataset(self.train_dataset)), 
                collate_fn=self.data_collator,
                batch_size=self.config.dataset.per_device_train_batch_size,  # Batch size per process
                num_workers=self.config.dataset.preprocessing_num_workers,   # Number of data loading workers
                pin_memory=True, 
                drop_last=True                                               # Drop the last incomplete batch (important for DDP)
                )

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.config.dataset.per_device_eval_batch_size, shuffle=False, num_workers=self.config.dataset.preprocessing_num_workers, collate_fn=self.data_collator)
        # return DataLoader(
        #     self.val_dataset, 
        #     num_workers=self.config.dataset.preprocessing_num_workers, 
        #     batch_size=self.config.dataset.per_device_eval_batch_size,
        #     collate_fn=self.val_dataset.collate_fn
        # )
        return DataLoader(
                dataset=ConcatDataset(self.val_dataset), 
                shuffle=False,
                collate_fn=self.data_collator,
                batch_size=self.config.dataset.per_device_eval_batch_size,  # Batch size per process
                num_workers=self.config.dataset.preprocessing_num_workers,   # Number of data loading workers
                pin_memory=True, 
                drop_last=True                                               # Drop the last incomplete batch (important for DDP)
                )

class SimPOModel(pl.LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: Dict[str, Any],
        ):
        
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accumulated_loss = []
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # default SimPO hyperparameter 
        simpo_config = self.config['simpo']
        tokenizer_config = self.config['tokenizer']

        self.loss_type = simpo_config.get('loss_type', 'sigmoid')
        self.beta = simpo_config.get('beta', 1.0)
        self.gamma_beta_ratio = simpo_config.get('gamma_beta_ratio', 0.0)
        self.label_smoothing = simpo_config.get('label_smoothing', 0.0)
        self.sft_weight = simpo_config.get('sft_weight', 0.0)

        self.label_pad_token_id = tokenizer_config.get('label_pad_token_id', -100)
        self.padding_value = self.tokenizer.pad_token_id # tokenizer_config.get('padding_value', 0) # = self.tokenizer.pad_token_id (SEED-LLaMA 기준)
        self.max_length = tokenizer_config.get('max_length', 512)
        self.max_prompt_length = tokenizer_config.get('max_prompt_length', 128)
        self.max_target_length = tokenizer_config.get('max_target_length', 128)
        self.use_dpo_data_collator = True # hard-coding

    def setup(self, stage: str):
        
        # load visual tokenizer (SEED tokenizer)
        if self.tokenizer is None:
            self.tokenizer = SeedLlamaTokenizer.from_pretrained(
                        pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
                        vit_precision=self.config.tokenizer.seed.vit_precision,
                        diffusion_precision=self.config.tokenizer.seed.diffusion_precision,
                        load_diffusion=self.config.tokenizer.seed.load_diffusion,
                        device='cpu',
                        encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
                        diffusion_path='stabilityai/stable-diffusion-2-1-unclip'
                        ) 
        
        self.model.config.bos_token_id = self.config.vocab.text_token_s # 1
        self.model.config.eos_token_id = self.config.vocab.text_token_e # 2 
        self.model.config.pad_token_id = self.config.vocab.text_token_p # 0
        logger.info(f"Model Config: \n{self.model.config}")  

    def on_train_start(self):
        """ before training step """
        # set train mode
        self.model.train()
        _save_config(self.logger.log_dir, self.config) 
        print("\n\n=== Saved Config ===")
    
    
    def training_step(self, batch, batch_idx):
        """ A batch is a dict. """

        # pdb.set_trace() # TODO: remove

        ac = self.config.experiment.gradient_accumulation_steps
        if ac > 1 and batch_idx % ac == 0:
            self.accumulated_loss.clear()
        loss = self.compute_loss(inputs=batch) # `return outputs=False`` hard coding
        if ac > 1:
                self.accumulated_loss.append(loss / ac)
        self.log_dict({'train/loss': (sum(self.accumulated_loss) if ac > 1 else loss),
                       'train/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                       'train/global_step': self.global_step}, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return loss # simPO loss


    # def on_train_batch_end(
    #     self, outputs, batch, batch_idx
    # ):
    #     logger.info(f"\n\n*** {batch_idx}th batch example ***")  
    #     logger.info(f"1. prompt: {batch['prompt_input_ids'][0]}")
    #     logger.info(f"2. chosen_input: {batch['chosen_input_ids'][0]}")  
    #     logger.info(f"3. chosen_label: {batch['chosen_labels'][0]}\n")  
        

    # def on_validation_start(self):
    #     # set eval mode
    #     self.model.eval()
    #     os.makedirs(f"{self.logger.log_dir}/val/step_{self.global_step}/images", exist_ok=True) # for t2i (CLIP score)
        
    #     # move tokenizer to GPU 
    #     self.tokenizer.to(self.device)
    #     if self.tokenizer.image_tokenizer.diffusion_model is not None:
    #         self.tokenizer.image_tokenizer.diffusion_model.to(self.device)
        
    #     # read original val data file
    #     with open(self.config.dataset.val.t2i.gt_txt_dir, 'r') as f:
    #         self.validation_data = json.load(f)
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
    
        """ A batch is a dict. """
        # calculate loss (정량 메트릭)
        loss = self.prediction_step(inputs=batch, prediction_loss_only=True) # hard-coding 
        self.log('val/loss', loss, prog_bar=True, logger=True, sync_dist=True)

        # # generate image (정성 메트릭)
        # generation_cfg_t2i = self.config.dataset.val.t2i.generation_cfg
        # # prompt_batch = f"USER: {batch['prompt_input_ids']} Please generate an image.\nASSISTANT:"

        # with torch.no_grad():
        #     output_batch = self.model.generate(
        #                     input_ids=batch['prompt_input_ids'],
        #                     **generation_cfg_t2i
        #                     )

        #     outputs = output_batch[:, batch['prompt_input_ids'].shape[1]:]
            
        #     for idx, img in enumerate(outputs):
        #         boi_list = torch.where(img == self.config.vocab.image_token_s)[0]
        #         eoi_list = torch.where(img == self.config.vocab.image_token_e)[0]

        #         if len(boi_list) == 0 and len(eoi_list) == 0:
        #             continue
        #         elif len(boi_list) != 0 and len(eoi_list) != 0:
        #             boi_idx = boi_list[0]
        #             eoi_idx = eoi_list[0]
        #             image_ids = (img[boi_idx+1:eoi_idx] - 32000)
        #         elif len(boi_list) == 0 and len(eoi_list) != 0:
        #             eoi_idx = eoi_list[0]
        #             image_ids = (img[:eoi_idx] - 32000)
        #         else:
        #             continue
                
        #         # fill zero
        #         if image_ids.shape[0] < 32:
        #             image_ids = torch.cat([image_ids, torch.zeros(32 - image_ids.shape[0], dtype=torch.int64).to(image_ids)], dim=0)
        #         else:
        #             image_ids = image_ids[:32]

        #         # process tokens in incorrect range
        #         has_illegal = False
        #         for token in image_ids:
        #             if token < 0 or token > 8191:
        #                 has_illegal = True
        #                 break
        #         if has_illegal:
        #             print(f"=== Error ! This is invalid range of token, {image_ids} ===")
        #             time.sleep(0.5)
        #             continue
                
        #         # save image
        #         # file_name = f"{idx:.5d}.jpg"
        #         file_name = f"{self.validation_data[idx]['chosen']}.jpg"
        #         try:
        #             image = self.tokenizer.decode_image(image_ids.reshape(1, -1))
        #             image[0].save(f"{self.logger.log_dir}/val/step_{self.global_step}/images/{file_name}") # same file name as val gt_img_dir
        #         except Exception as e:
        #             print(f"=== Error ! Cannot save image, {file_name} '{e}' ===")
            
        return loss 
        

    # def on_validation_end(self):
        
    #     # move tokenizer to CPU 
    #     self.tokenizer.to('cpu')
    #     if self.tokenizer.image_tokenizer.diffusion_model is not None:
    #         self.tokenizer.image_tokenizer.diffusion_model.to('cpu')
        
    #     # wait for all GPU finishing work
    #     self.trainer.strategy.barrier()

    #     if dist.get_rank() == 0:
    #         print("\n=== Start CLIP score calculation ===")
    #         # gt_dir and gen_dir has same img_file name 
    #         gt_image_dir = self.config.dataset.val.t2i.gt_img_dir # TODO: use REAL val dataset, not slice of train data
    #         generated_image_dir = f'{self.logger.log_dir}/val/step_{self.global_step}/images'

    #         clip_score = calculate_clip_s_for_folder(gt_image_dir, generated_image_dir)  
    #         num_images = len(glob.glob(os.path.join(generated_image_dir, '*.jpg')))

    #         print(f'=== Number of generated images: {num_images}')
    #         print(f"=== CLIP score: {clip_score}")

    #         self.logger.experiment.add_scalar(
    #             'val/num_generated_imgs_for_clip_score', num_images, global_step=self.global_step
    #         )
    #         self.logger.experiment.add_scalar(
    #             'val/clip_score', clip_score, global_step=self.global_step,
    #         )


    def on_before_optimizer_step(self, optimizer, _):
        # log total grad norm
        self.log('train/grad_norm', self.compute_total_grad_norm(), 
                 on_step=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = AdamW(
        self.model.parameters(), 
        lr=self.config.optimizer.init_lr,
        betas=self.config.optimizer.betas,
        # weight_decay=self.config.experiment.weight_decay,
        eps=self.config.optimizer.eps
        ) # 이전 코드까지는 betas, wd, eps 주석처리였으나 실상 무의미 (config values are same as default)
        warmup_step = self.config.experiment.max_training_step * self.config.scheduler.warmup_ratio
        # scheduler = CosineDecayWarmUpRestarts(optimizer, warmup_iter=warmup_step, max_iter=self.config.experiment.max_training_step, eta_min=self.config.experiment.min_lr, eta_max=self.config.experiment.lr)
        scheduler = CosineDecayWarmUpRestarts(optimizer, warmup_iter=warmup_step, max_iter=self.config.experiment.max_training_step, eta_max=self.config.scheduler.lr)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_config]
        
        
    # ======= Loss Calculation Order ======= 
    # 1. compute_loss
    # 2. get_batch_loss_metrics
    # 3. concatenated_forward, simpo_loss
    # 4. get_batch_logps, concatenated_inputs (for concatenated_forward)

    # keys:  dict_keys(['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask'])
    # Presuming that the model is NO encoder-decoder model.
    
    
    # ======== Util Functions ========
    # @staticmethod
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=self.device_type)

        return concatenated_batch
        
    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.device_type)
        logits = pi_logratios - self.gamma_beta_ratio

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = self.beta * policy_chosen_logps.to(self.device_type).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.device_type).detach()

        return losses, chosen_rewards, rejected_rewards


    def concatenated_forward(
            self, batch: Dict[str, Union[List, torch.LongTensor]]
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
            """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

            We do this to avoid doing two forward passes, because it's faster for FSDP.
            """
            concatenated_batch = self.concatenated_inputs(batch=batch)
            len_chosen = batch["chosen_labels"].shape[0]

            model_kwargs = ({})

            all_logits = self.model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                **model_kwargs,
            ).logits

            all_logps = self.get_batch_logps(
                all_logits,
                concatenated_batch["concatenated_labels"],
                average_log_prob=True,
                # is_encoder_decoder=self.is_encoder_decoder,
                # label_pad_token_id=self.label_pad_token_id,
            )

            chosen_logps = all_logps[:len_chosen]
            rejected_logps = all_logps[len_chosen:]

            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]

            chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)
        
        
    # @staticmethod
    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

        
    def get_batch_loss_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "val"] = "train",
    ):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        prefix = "val" if train_eval == "val" else "train"

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
        ) = self.concatenated_forward(batch)

        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )

        loss = losses.mean()

        if self.sft_weight > 0.0:
            # if not self.is_encoder_decoder:
            policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
            chosen_labels = chosen_labels[..., 1:].clone()
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            ce_loss = loss
            loss = self.sft_weight * sft_loss + ce_loss
            self.log_dict({f"{prefix}/sft_loss": sft_loss.detach().cpu(),
                           f"{prefix}/ce_loss": ce_loss.detach().cpu(),}, prog_bar=True, logger=True, sync_dist=True)
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        self.log_dict({f"{prefix}/rewards/chosen": chosen_rewards.mean().cpu(),
                       f"{prefix}/rewards/rejected": rejected_rewards.mean().cpu(),
                       f"{prefix}/rewards/accuracies": reward_accuracies.mean().cpu(),
                       f"{prefix}/rewards/margins": (chosen_rewards - rejected_rewards).mean().cpu(),
                       f"{prefix}/logps/rejected": policy_rejected_logps.detach().mean().cpu(),
                       f"{prefix}/logps/chosen": policy_chosen_logps.detach().mean().cpu(),
                       f"{prefix}/logits/rejected": policy_rejected_logits.detach().mean().cpu(),
                       f"{prefix}/logits/chosen": policy_chosen_logits.detach().mean().cpu(),
                    #    }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
                       }, prog_bar=True, logger=True, sync_dist=True)


        return loss


    def compute_loss(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        # return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        # TODO: check the role 
        compute_loss_context_manager = torch.cuda.amp.autocast # if self._peft_has_been_casted_to_bf16 else nullcontext

        # TODO: if error, remove `context_manager`
        with compute_loss_context_manager():
            loss = self.get_batch_loss_metrics(inputs, train_eval="train")

        return loss


    def prediction_step(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        # ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
            
        # if ignore_keys is None:
        #     if hasattr(self.model, "config"):
        #         ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        #     else:
        #         ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast # if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss = self.get_batch_loss_metrics(inputs, train_eval="val")

        if prediction_loss_only:
            return loss.detach()
            # return (loss.detach(), None, None)
        
        # TODO
        # logits for the chosen and rejected samples from model
        # logits_dict = {
        #     "eval_logits/chosen": metrics["eval_logits/chosen"],
        #     "eval_logits/rejected": metrics["eval_logits/rejected"],
        # }
        # logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items()) # remove ignore_keys
        # logits = torch.stack(logits).mean(axis=1).to(self.device_type)
        # labels = torch.zeros(logits.shape[0], device=self.device_type)

        # return (loss.detach(), logits, labels)


    def compute_total_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm



def main():
    """ 1. Set up"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default="configs/simpo/train_mm_v10.yaml")
    # parser.add_argument('--exp_name', type=str, required=True)
    # parser.add_argument('--lr', type=float, required=True)
    # parser.add_argument('--beta', type=float, required=True)
    # parser.add_argument('--gamma_beta_ratio', type=float, required=True)

    args = parser.parse_args()

    config, _ = build_config(cfg_path=args.cfg_path)
    # config.exp_name = args.exp_name # TODO: temp
    config.save_dir = os.path.join(str(config.result_path), str(config.exp_name))
    os.makedirs(config.save_dir, exist_ok=True) # make directory

    # for grid-search
    # if config.optimizer.init_lr is None:
    #     config.optimizer.init_lr = args.lr
    # if config.scheduler.lr is None:
    #     config.scheduler.lr = args.lr
    # if config.simpo.beta is None:
    #     config.simpo.beta = args.beta
    # if config.simpo.gamma_beta_ratio is None:
    #     config.simpo.gamma_beta_ratio = args.gamma_beta_ratio

    # add (9/1)
    # if config.experiment.max_training_step is None:
    #     config.experiment.max_training_step = config.experiment.epoch_size * config.experiment.num_train_epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(config.experiment.seed, workers=True) 
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
                # logging.StreamHandler(sys.stdout),  # Console output
                logging.FileHandler(f'{config.save_dir}/{config.exp_name}.txt')  # File output
                ],
        )
    log_level = config.log_level
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    """ 2. Load model and tokenizer """
    # clarify precision
    if config.precision != "auto":
        precision = config.precision
        assert precision in ["fp32", "fp16", "bf16"], f"Invalid precision: {precision} ! Must be one of ['fp32', 'fp16', 'bf16']"
        if precision == "bf16":
            torch_dtype = torch.bfloat16
        elif precision == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # load model
        model_kwargs = dict(
            pretrained_model_name_or_path=config.model_name_or_path,
            # trust_remote_code=config.get('trust_remote_code', False),
            torch_dtype=torch_dtype,     
            attn_implementation=config.attn_implementation,
            use_cache=False if config.experiment.gradient_checkpointing else True,
            # device_map=get_kbit_device_map() if quantization_config is not None else None,
            # quantization_config=quantization_config,
            # device_map=device, # modify, Not in official SimPO Code.
            # Error message: https://github.com/openai/whisper/discussions/1948
        )
        model = LlamaForCausalLM.from_pretrained(**model_kwargs)
        logger.info(f"Loading {torch_dtype} Model from originally {model.config.torch_dtype} torch_dtype Model before Training.")  
    
    # import original pretrained-model's torch_dtype (SEED-LLaMA-8B-SFT:fp32)
    else:
        model_kwargs = dict(
            pretrained_model_name_or_path=config.model_name_or_path,
            torch_dtype="auto",
            use_cache=False if config.experiment.gradient_checkpointing else True,
            # attn_implementation=config.attn_implementation, # only support fp16, bf16
        )
        model = LlamaForCausalLM.from_pretrained(**model_kwargs)
        logger.info(f"Loading {model.config.torch_dtype} torch_dtype Model same as ORIGINAL before Training.")  

    if config.experiment.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha, 
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
        # modules_to_save=config.lora.modules_to_save
    )
    peft_model = get_peft_model(model, lora_config)
    logger.info(f"Loading PEFT Model done.")
        
    # load tokenizer
    # text_tokenizer = LlamaTokenizer.from_pretrained(config.text_tokenizer_name_or_path)
    visual_tokenizer = SeedLlamaTokenizer.from_pretrained(
                        pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
                        vit_precision=config.tokenizer.seed.vit_precision,
                        diffusion_precision=config.tokenizer.seed.diffusion_precision,
                        load_diffusion=config.tokenizer.seed.load_diffusion,
                        device='cpu',
                        encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
                        diffusion_path='stabilityai/stable-diffusion-2-1-unclip'
                        ) 
    
    # additional setting for SimPO (add 8/31)
    try:
        if visual_tokenizer.pad_token_id is None:
            visual_tokenizer.pad_token_id = 0 # visual_tokenizer.eos_token_id
    except Exception as e1:
        print (f"Error in setting tokenizer: {e1}")

    if config.tokenizer.truncation_side is not None: # default left
        try:
            visual_tokenizer.truncation_side = config.tokenizer.truncation_side
        except Exception as e2:
            print (f"Error in setting tokenizer: {e2}")

    # set reasonable default for models without max length
    try:
        if visual_tokenizer.model_max_length > 100_000:
            visual_tokenizer.model_max_length = config.tokenizer.max_length # 2048
    except Exception as e3:
        print (f"Error in setting tokenizer: {e3}")

    logger.info(f"2 __ Load model and tokenizer done.")  
    
    
    """ 3. Load dataset """
    # columns_to_keep=["chosen", "rejected", "prompt"]
    datamodule = SimPODataModule(config, visual_tokenizer) 
    # datamodule = SimPODataModule(config, visual_tokenizer) 
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader() 
    logger.info(f"3 __ Load dataset done.")
    
    
    """ 4. Define training wrapper and callbacks """
    if config.model_type != 'visual_llama':
        wrapper = SimPOModel(model=peft_model, tokenizer=visual_tokenizer, config=config) 
    else:
        wrapper = SimPOModel.load_from_checkpoint(config.model_ckpt_path, model=peft_model, tokenizer=visual_tokenizer, config=config)
    # wrapper = SimPOModel(model=peft_model, tokenizer=visual_tokenizer, config=config) 
    # wrapper.setup(stage="fit")
    wrapper.model.print_trainable_parameters()
    
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=config.save_dir, name=config.exp_name)
    # Error
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=tb_logger.log_dir,
    #     save_top_k=-1, # save all ckpt corresponding to saving interval    
    #     # save based on `epoch`
    #     filename="{epoch:02d}",  
    #     every_n_epochs=1,  # Save every epoch
    # )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{step:06d}",
        save_top_k=-1, # save all ckpt corresponding to saving interval              
        every_n_train_steps=config.experiment.save_steps, 
        # save_last=True
    )

    # Set up the PyTorch Lightning trainer
    trainer = pl.Trainer(
        devices=config.world_size,
        accelerator=device,
        logger=tb_logger,
        default_root_dir=config.save_dir,
        callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], # max_depth: layers displayed in summary
        strategy=DDPStrategy(                                                      
            find_unused_parameters=False
        ),            
        log_every_n_steps=config.experiment.log_steps,
        gradient_clip_val=config.experiment.gradient_clip_val, # max_grad_norm
        enable_checkpointing=config.experiment.gradient_checkpointing,
        accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
        precision="bf16" if config.precision is None or config.precision == "auto" else config.precision, #config.precision, 
        max_steps=config.experiment.max_training_step, # or max_epochs   
        check_val_every_n_epoch=None,
        val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps, # TODO: check
        # num_sanity_val_steps = 0,           
    )
    logger.info("4 __ Trainer set up done.")


    """ 5. Train model """
    # Train the model
    if config.resume is not None and os.path.exists(config.resume):
        logger.info("*** Training resume ***")
        trainer.fit(wrapper, train_dataloader, val_dataloader, ckpt_path=config.resume)
    else:
        trainer.fit(wrapper, train_dataloader, val_dataloader)
        
    logger.info("5 __ Training done.")

    
if __name__ == "__main__":
    main()