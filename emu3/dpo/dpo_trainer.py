# This code is modified from mDPO Trainer.
# Modified for Emu3-Gen; T2I ver. DPO Training code

from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import pad_to_length


# Custom trainer class for T2I task
class DPOTrainerEmu3(DPOTrainer):
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}

        # Emu3 is llama-like model, therefore it is (only) decoder model.
        # print(batch)
        # print(batch["chosen_input_ids"].shape) # torch.Size([1, 8212]), 가장 최신
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0, # dim=1 시 아래 로직과 맞지 않음
                ).to(self.accelerator.device)

        # print("# concatenated_batch")
        # print(concatenated_batch["concatenated_input_ids"].shape) # torch.Size([2, 8212])
        
        return concatenated_batch 
    
    
    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        """
        (참고)

        batch["chosen_input_ids"] = chosen_input_ids.squeeze(0)
        batch["chosen_attention_mask"] = chosen_attention_mask.squeeze(0)
        batch["chosen_labels"] = chosen_labels.squeeze(0)
        
        batch["rejected_input_ids"] = rejected_input_ids.squeeze(0)
        batch["rejected_attention_mask"] = rejected_attention_mask.squeeze(0)
        batch["rejected_labels"] = rejected_labels.squeeze(0)

        """
        concatenated_batch = self.concatenated_inputs(batch) 
        len_chosen = batch["chosen_labels"].shape[0]
        # print("# chosen_labels shape: ", batch["chosen_labels"].shape) # torch.Size([1, 5120])

        model_kwargs = {
            "labels": concatenated_batch["concatenated_labels"],
        }

        # (참고) /home/yjoh/project/Emu3-dpo/emu3/mllm/modeling_emu3.py
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                **model_kwargs,
            )

        all_logits = outputs.logits.to(torch.float32)
        # refined_labels = concatenated_batch["concatenated_labels"] 

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True
            # average_log_prob=False, (not official args)
        )

        # print("\ndebug 5")
        # print("# all_logits: ", all_logits) 
        # print(all_logits.shape) # torch.Size([1, 10240, 184622])
        # print("# all_logps: ", all_logps)
        # print(all_logps.shape)
        # all_logps:  tensor([-8.1283, -8.1058], device='cuda:0')
        # torch.Size([2])

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        # 일반 DPO Trianer 이므로 imageless 부분 삭제

        # return 4
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    
    def cal_likelihood(self, logits, input_ids, attention_mask):
        """
        The likelihood (or more accurately, the log-probability) of a sequence is obtained by taking the log-softmax of the logits and summing the log-probabilities for each token in the sequence.
        
        Calculate log likelihood of a given input sequence.
        
        Args:
            logits: Tensor of shape [batch_size, seq_len, vocab_size].
            input_ids: Tensor of shape [batch_size, seq_len] with token IDs.
            attention_mask: Tensor of shape [batch_size, seq_len] with attention mask.

        Returns:
            log_likelihood: Tensor of shape [batch_size] representing the log-likelihood of each sequence.
        """
        # Calculate log-softmax of logits to get log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]
        
        # Gather the log-probabilities corresponding to the actual token IDs
        token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens using the attention mask
        token_log_probs = token_log_probs * attention_mask
        
        # Sum the log-probabilities over the sequence length to get the total log-likelihood for each sequence
        log_likelihood = token_log_probs.sum(dim=-1)
        
        return log_likelihood
    
 
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0
        logits = pi_logratios - ref_logratios  # response preference

        # General DPO Loss
        loss= -torch.nn.functional.logsigmoid(self.beta * logits)

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return loss, chosen_rewards, rejected_rewards


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        # print("\ndebug 4")
        # print("# policy_chosen_logps: ", policy_chosen_logps) # tensor([-8.0444], device='cuda:0', grad_fn=<SliceBackward0>)
        # print("# policy_rejected_logps: ", policy_rejected_logps) # tensor([-8.0879], device='cuda:0', grad_fn=<SliceBackward0>)
        # print("# reference_chosen_logps: ", reference_chosen_logps) # tensor([-8.0387], device='cuda:0')
        # print("# reference_rejected_logps: ", reference_rejected_logps) # tensor([-8.0960], device='cuda:0')

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else "train_" # modify
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()


        return losses.mean(), metrics
    

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

        # print("\ndebug 6")
        # print("# logits: ", logits.shape)
        # print("# labels: ", labels.shape)

        # logits:  torch.Size([1, 10240, 184622])
        # labels:  torch.Size([1, 10240])

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