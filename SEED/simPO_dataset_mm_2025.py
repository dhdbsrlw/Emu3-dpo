# README: v4 / Last update: 9/30
# With Instruction, unified-1 instruction
# add: start_idx, end_idx

import random
import torch.nn as nn 
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler
import torch.distributed as dist


from typing import Dict
from omegaconf import OmegaConf
from datasets import load_dataset
from alignment.data import maybe_insert_system_message, is_openai_format

from transformers import LlamaTokenizer
from models.seed_llama_tokenizer import SeedLlamaTokenizer
from trl.trainer.utils import DPODataCollatorWithPadding # installed in `seed_flash`

from constant import *
from utils.config import build_config
from dataclass_main_dev import load_data, load_data_from_tar

# Constant (TEMP)
t2i_dataset = ["pickapic", "hpdv2", "image_reward_db"]
it2t_dataset = ["rlhf_v", "rlaif_v"]

# apply to SINGLE dataset
class SimPODataset_MM(Dataset):
    def __init__(self, data_dir, tokenizer, tokenizer_cfg, sampling_rate=1.0): # presume `train`-split
        self.tokenizer = tokenizer
        for k, v in tokenizer_cfg.items():
            setattr(self, k, v)

        # load data
        try: 
            self.dataset = []
            data_files = load_data(data_dir=data_dir, recursive=True)
            for data_file in tqdm(data_files, desc="\n### Formatting comparisons with prompt template: "):
                self.dataset.extend(load_data_from_tar(data_file))

        except:
            try:
                self.dataset = load_data_from_tar(data_dir)
            except Exception as e:
                raise ValueError("Error while loading data !")

        if sampling_rate != 1.0:
            self.total_length = int(len(self.dataset) * sampling_rate)
            print(f"### Total length of data before sampling: {len(self.dataset)}")
            print(f"### Total length of data after sampling: {self.total_length}\n\n")
            assert self.total_length > 0, "Dataset size must be bigger than 1."
            self.dataset = self.dataset[:self.total_length] # 앞에서부터 N개를 샘플링
        else:
            self.total_length = len(self.dataset)
            print(f"### Total length of data (not sampling): {self.total_length}\n\n")

        # ready to tokenize
        if self.tokenizer.pad_token_id != 0:
            self.tokenizer.pad_token_id = 0
                
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """ Inputs should be triples of (prompt, chosen, rejected) """
        # 2 type of data
        # TODO: 고정
        dcode_function = self.decode_t2i_preference_pair_with_edit
        try:
            data = dcode_function(
                self.dataset[idx],
            )
            return data 
        except Exception as e:
            print(f"Error in __getitem__ because ... {e}")
            return self.__getitem__(idx+1) # recursive call

    def decode_it2t_preference_pair(self, example: Dict):

        if "image_ids" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(
                    f"Could not format `MultiModal` example as dialogue for SimPO task! This example only has {example.keys()} keys.\n"
                )

        # Phase 1. Make a Template
        system_message = ''

        image_ids = example['image_ids']
        image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN
        
        # question = example["prompt"].strip()
        question = example["question"].strip()
        prompt_messages = system_message + s_token + " " + image_tokens + question + sep + e_token + " "
        
        chosen_messages = example["chosen"].strip()
        if not chosen_messages.endswith('.'):
            chosen_messages = chosen_messages+ '.'
        if chosen_messages.startswith(self.tokenizer.bos_token):
            chosen_messages = chosen_messages[len(self.tokenizer.bos_token):]

        rejected_messages = example["rejected"].strip()
        if not rejected_messages.endswith('.'):
            rejected_messages = rejected_messages + '.'
        if rejected_messages.startswith(self.tokenizer.bos_token):
            rejected_messages = rejected_messages[len(self.tokenizer.bos_token):]
            
        # example["mm_prompt"] = prompt_messages
        # example["text_chosen"] = chosen_messages
        # example["text_rejected"] = rejected_messages
        
        example["prompt"] = prompt_messages
        example["chosen"] = chosen_messages
        example["rejected"] = rejected_messages

        # remove 'image_ids' key in original dict
        del example["image_ids"]
        del example["question"]

        # Phase 2. Tokenize each Message
        row = self.tokenize_it2t_preference_pair(example) # single row(=sample)

        return row 

    def tokenize_it2t_preference_pair(self, feature: Dict):
        """Tokenize a single row(sample) from a SimPO specific dataset."""
        batch = {}
        prompt = feature["prompt"] 
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        # prompt = feature["mm_prompt"] 
        # chosen = feature["text_chosen"]
        # rejected = feature["text_rejected"]

        # Presuming SEED-LLaMA is 'not self.is_encoder_decoder'

        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        
        # original
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        # Answer part still consists of ONLY text.
        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen) # check correct tokenization

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected) # check correct tokenization 

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt. Avoid adding if it's already there
        bos_token_id = self.tokenizer.bos_token_id
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer. Avoid adding if it's already there
        eos_token_id = self.tokenizer.eos_token_id
        if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)
        if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
            rejected_tokens["input_ids"].append(eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        return batch

    def decode_t2i_preference_pair(self, example: Dict):

        if "chosen_image_ids" not in example.keys() or "rejected_image_ids" not in example.keys() or "text" not in example.keys():
            raise ValueError(
                    f"Could not format `T2I - MultiModal` example as dialogue for SimPO training !"
                )

        # Phase 1. Make a Template
        system_message = ''

        chosen_image_ids = example['chosen_image_ids']
        rejected_image_ids = example['rejected_image_ids']

        # image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN
        chosen_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in chosen_image_ids]) + EOI_TOKEN
        rejected_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in rejected_image_ids]) + EOI_TOKEN
        
        caption = example['text'].strip()
        
        # according to `decode_sft.py` 444 line
        if caption.endswith('.'):
            caption = caption.rstrip(".")

        # indx = random.randint(0, 19)
        prompt = "Please generate an image." #gen_prompt[indx]
        # response = gen_prompt_response[indx]
        # if indx >= 14:
        #     punct = '?'
        # else:
        #     punct = '.'

        # prompt_messages = question
        # image_tokens = answer
        # re-order (modify)
        prompt_messages = system_message + s_token + " " + caption + ' ' + prompt + sep + e_token
        
        example["prompt"] = prompt_messages         
        example["chosen"] = chosen_image_tokens     
        example["rejected"] = rejected_image_tokens 

        # remove unnecessary keys in original dict
        del example["text"]
        del example["chosen_image_ids"]
        del example["rejected_image_ids"]
        
        # Phase 2. Tokenize each Message
        row = self.tokenize_t2i_preference_pair(example) # single row(=sample)

        return row 
    
    def tokenize_t2i_preference_pair(self, feature: Dict):
        """Tokenize a single row(sample) from a SimPO specific dataset."""

        batch = {}
        prompt = feature["prompt"] 
        chosen = feature["chosen"]      
        rejected = feature["rejected"]  


        # Presuming SEED-LLaMA is 'not self.is_encoder_decoder'
        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        
        # original
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        # Answer part stil consists of ONLY text.
        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen) # check correct tokenization

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected) # check correct tokenization 

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt. Avoid adding if it's already there
        bos_token_id = self.tokenizer.bos_token_id
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer. Avoid adding if it's already there
        eos_token_id = self.tokenizer.eos_token_id
        if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)
        if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
            rejected_tokens["input_ids"].append(eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        # print("\n\n... debug ...")
        # print(f"# prompt_input_ids: {batch['prompt_input_ids']}")
        # print(f"# chosen_ii: {batch['chosen_input_ids']}")
        # print(f"# chosen_lb: {batch['chosen_labels']}")
        # print(f"# chosen_am: {batch['chosen_attention_mask']}")

        return batch
    
    def decode_t2i_preference_pair_with_edit(self, example: Dict):

        if "source_image_ids" not in example.keys() or "target_image_ids" not in example.keys() or "output_description" not in example.keys():
            raise ValueError(
                    f"Could not format `T2I with Edit dataset - MultiModal` example as dialogue for SimPO training !"
                )

        # Phase 1. Make a Template
        system_message = ''

        chosen_image_ids = example['target_image_ids']
        rejected_image_ids = example['source_image_ids']

        # image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in image_ids]) + EOI_TOKEN
        chosen_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in chosen_image_ids]) + EOI_TOKEN
        rejected_image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in rejected_image_ids]) + EOI_TOKEN
        
        caption = example['output_caption_by_llama'].strip()
        
        # according to `decode_sft.py` 444 line
        if caption.endswith('.'):
            caption = caption.rstrip(".")

        # indx = random.randint(0, 19)
        prompt = "Please generate an image." #gen_prompt[indx]
        # response = gen_prompt_response[indx]
        # if indx >= 14:
        #     punct = '?'
        # else:
        #     punct = '.'

        # prompt_messages = question
        # image_tokens = answer
        # re-order (modify)
        prompt_messages = system_message + s_token + " " + caption + ' ' + prompt + sep + e_token
        

        batch={}
        batch["prompt"] = prompt_messages         
        batch["chosen"] = chosen_image_tokens     
        batch["rejected"] = rejected_image_tokens 

        # Phase 2. Tokenize each Message
        row = self.tokenize_t2i_preference_pair_with_edit(batch) # single row(=sample)

        return row 
    
    def tokenize_t2i_preference_pair_with_edit(self, feature: Dict):
        """Tokenize a single row(sample) from a SimPO specific dataset."""

        batch = {}
        prompt = feature["prompt"] 
        chosen = feature["chosen"]      
        rejected = feature["rejected"]  


        # Presuming SEED-LLaMA is 'not self.is_encoder_decoder'
        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        
        # original
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        # Answer part stil consists of ONLY text.
        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        chosen_tokens = self.build_tokenized_answer(prompt, chosen) # check correct tokenization

        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        rejected_tokens = self.build_tokenized_answer(prompt, rejected) # check correct tokenization 

        # Last prompt token might get merged by tokenizer and
        # it should not be included for generation if that happens
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        for k, v in prompt_tokens.items():
            prompt_tokens[k] = v[:prompt_len_input_ids]

        # Make sure prompts only have one different token at most
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )
        num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        if num_diff_tokens > 1 or num_diff_len > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
            )

        # add BOS token to head of prompt. Avoid adding if it's already there
        bos_token_id = self.tokenizer.bos_token_id
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # add EOS token to end of answer. Avoid adding if it's already there
        eos_token_id = self.tokenizer.eos_token_id
        if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
            chosen_tokens["input_ids"].append(eos_token_id)
            chosen_tokens["attention_mask"].append(1)
        if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
            rejected_tokens["input_ids"].append(eos_token_id)
            rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

        # Create labels
        chosen_sequence_tokens = {
            k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        rejected_sequence_tokens = {
            k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        }
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(chosen_tokens["prompt_input_ids"])
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
            self.label_pad_token_id
        ] * len(rejected_tokens["prompt_input_ids"])

        for k, toks in {
            "chosen_": chosen_sequence_tokens,
            "rejected_": rejected_sequence_tokens,
            "": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        # print("\n\n... debug ...")
        # print(f"# prompt_input_ids: {batch['prompt_input_ids']}")
        # print(f"# chosen_ii: {batch['chosen_input_ids']}")
        # print(f"# chosen_lb: {batch['chosen_labels']}")
        # print(f"# chosen_am: {batch['chosen_attention_mask']}")

        return batch
    


    def build_tokenized_answer(self, prompt, answer) -> Dict:
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
        
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
    





# Debugging
if __name__ == '__main__':
    
    # (SEED) tokenizer import 
    # tokenizer_cfg_path="/home/ubuntu/MAGVLT2/MultiModalLLM/configs/tokenizer/seed_llama_tokenizer_hf.yaml" 
    # transform_cfg_path="/home/ubuntu/MAGVLT2/MultiModalLLM/configs/transform/clip_transform.yaml"
    
    # tokenizer = hydra.utils.instantiate(OmegaConf.load(tokenizer_cfg_path), device='cpu', load_diffusion=False) # text tokenizer (llama tokenizer)
    # transform = hydra.utils.instantiate(OmegaConf.load(transform_cfg_path))
    
    # (LLaMA2-chat) tokenizer import
    # tokenizer = LlamaTokenizer.from_pretrained('/nas/backup/checkpoints/llama2/Llama-2-7b-chat-hf')
    
    # SEED-Tokenizer
    tokenizer = SeedLlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2',
            vit_precision='fp16',
            diffusion_precision='fp16',
            load_diffusion=True, # TODO: change state as your need
            device='cpu',
            encoder_url='https://huggingface.co/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt',
            diffusion_path='stabilityai/stable-diffusion-2-1-unclip'
        )

    config, _ = build_config(cfg_path="configs/simpo/train_mm_v2.yaml")
    data_dirs = config.dataset.data_dir             # list
    sampling_rates = config.dataset.sampling_rate   # list
    assert len(data_dirs) == len(sampling_rates), "The number of data and sampling rate must be same !"

    # generate collection of dataset
    datasets=[]
    for data, sampling in zip(data_dirs, sampling_rates):
        assert 0.0 <= sampling <= 1.0, "Sampling rate must be between 0 and 1."
        try:
            datasets.append(SimPODataset_MM(data_dir=data, sampling_rate=sampling, tokenizer=tokenizer, tokenizer_cfg=config['tokenizer']))
        except:
            raise ValueError(f"All dataset should generated in torchData.Dataset. `{data}` raised ERROR.")

    # data_collator
    data_collator=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id, 
                label_pad_token_id=config.tokenizer.label_pad_token_id
            )
    
    # dataloader (DO NOT USE `SAMPLER`)
    dataloader = DataLoader(
    dataset=ConcatDataset(datasets), 
    sampler=RandomSampler(ConcatDataset(datasets)), 
    collate_fn=data_collator,
    batch_size=config.dataset.per_device_train_batch_size,  # Batch size per process
    num_workers=config.dataset.preprocessing_num_workers,   # Number of data loading workers
    # shuffle=True,
    pin_memory=True, 
    drop_last=True                                          # Drop the last incomplete batch (important for DDP)
    )
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collator) 

    for idx, data in enumerate(dataloader):
        print(data)
        break
            # print(f"### {i}th ###")
            # print(data['chosen_labels'])
            # print()
    
    print("\nDone !")
    