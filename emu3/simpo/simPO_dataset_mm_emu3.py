# README: v4 / Last update: 12/30
# HumanEdit - Emu3 tokenized ver.

import json
import torch
import os.path as osp

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset, RandomSampler

from typing import Dict
# from omegaconf import OmegaConf
# from datasets import load_dataset
# from alignment.data import maybe_insert_system_message, is_openai_format
from trl.trainer.utils import DPODataCollatorWithPadding # installed in `seed_flash`
import traceback


# apply to SINGLE dataset
class SimPODataset_MM(Dataset):
    def __init__(self, data_dir, tokenizer, tokenizer_cfg, sampling_rate=1.0): # presume `train`-split
        self.tokenizer = tokenizer
        for k, v in tokenizer_cfg.items():
            setattr(self, k, v)
        # 특별 추가
        if self.max_position_embeddings is not None:
            self.max_length = self.max_position_embeddings
        else:
            self.max_length = 4200 # hard-coding based on experience

        # (참고) /home/yjoh/project/Emu3-dpo/emu3/dpo/run_dpo_Emu3.py
        self.visual_token_pattern = "<|visual token {token_id:0>6d}|>"
        self.codebook_size = 32768

        # load data
        try:
            with open(data_dir) as f:
                d = json.load(f)
        except:
            raise ValueError("Cannot load data !")

        self.path_prefix = d["prefix"]
        self.filelist = d["path_list"]

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id != 151643:
            self.tokenizer.pad_token_id = 151643
        self.boi_token_id = 151852
            
                

        # sampling
        if sampling_rate != 1.0:
            self.total_length = int(len(self.filelist) * sampling_rate)
            print(f"### Total length of data before sampling: {len(self.filelist)}")
            print(f"### Total length of data after sampling: {self.total_length}\n\n")
            assert self.total_length > 0, "Dataset size must be bigger than 1."
            # self.dataset = self.dataset[:self.total_length] # 앞에서부터 N개를 샘플링
            self.filelist - self.filelist[:self.total_length]
        else:
            self.total_length = len(self.filelist)
            print(f"### Total length of data (not sampling): {self.total_length}\n\n")


        # for Emu3 (special token) 
        # TODO: self attribute
        self.bov = tokenizer.encode(self.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(self.visual_token_pattern.format(token_id=self.codebook_size - 1))[0]


    def __len__(self):
        return self.total_length


    def __getitem__(self, idx):
        """ Inputs should be triples of (prompt, chosen, rejected) """
        # 2 type of data
        # TODO: (일단) 고정
        dcode_function = self.decode_t2i_preference_pair_with_edit

        path = osp.join(self.path_prefix, self.filelist[idx])
        data = torch.load(path)

        try:
            ddata = dcode_function(data)
            return ddata # =decoded data 
        
        except Exception as e:
            print(f"Error in __getitem__ because ... {e}")
            print(traceback.format_exc())
            raise ValueError("Error in __getitem__")
            # return self.__getitem__(idx+1) # recursive call



    # 이하는 태스크별 디코드

    # 아직 SEED-LLaMA 에서 미수정
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

    # 아직 SEED-LLaMA 에서 미수정
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
        # prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        prompt_tokens = {f"prompt_{k}": v.squeeze(0) for k, v in prompt_tokens.items()} # squeeze 미수행시 오류 발생

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

    # 아직 SEED-LLaMA 에서 미수정
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
    
    # 아직 SEED-LLaMA 에서 미수정
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

        return batch
    
    # Emu3 반영 완료
    def decode_t2i_preference_pair_with_edit(self, example: Dict):

        if "input_images" not in example.keys() or "output_images" not in example.keys() or "output_description" not in example.keys():
            raise ValueError(
                    f"Could not format `T2I with Edit dataset - MultiModal` example as dialogue for SimPO training !"
                )
        

        instruction = "Please generate an image." #gen_prompt[indx]

        # image process
        rejected_image_prompt = self.format_image_prompt(example["input_images"]) # =input_image_tokens
        chosen_image_prompt = self.format_image_prompt(example["output_images"]) # =output_image_tokens
        prompt = example["output_description"] # != output_caption


        # print("debug 1")
        # print(len(rejected_image_prompt))
        # cnt = 0
        # for c in rejected_image_prompt:
        #     if c == "<":
        #         cnt+=1 
        # print("# cnt: ", cnt) 
        
        # # 각각 다른 image area 가 될 때: 4194, 8194

        # make input sequence (ONLY ANSWER PART)
        # # (1) generate `chosen` sequence
        # chosen_input = self.tokenizer.bos_token + prompt + " " + self.instruction + chosen_image_prompt + self.tokenizer.eos_token
        # # (2) generate `chosen` sequence
        # rejected_input = self.tokenizer.bos_token + prompt + " " + self.instruction + rejected_image_prompt + self.tokenizer.eos_token
        prompt_messages = self.tokenizer.bos_token + prompt + " " + instruction 


        # TODO: USER/ASSISTANT
        batch={}
        batch["prompt"] = prompt_messages         
        batch["chosen"] = chosen_image_prompt     
        batch["rejected"] = rejected_image_prompt 

        # Phase 2. Tokenize each Message
        row = self.tokenize_t2i_preference_pair_with_edit(batch) # single row(=sample)

        return row 

    
    # Emu3 반영 완료 (TODO: eos_token_id 수정)
    def tokenize_t2i_preference_pair_with_edit(self, feature: Dict):
        """Tokenize a single row(sample) from a SimPO specific dataset.
        
        (SEED-LLaMA 대비 주요 변경 사항)
        - prompt_tokens 삭제
        - prompt 와 answer 를 한번에 통합하여 토크나이징함으로써 연관 코드 모두 수정
        """

        batch = {}
        prompt = feature["prompt"] 
        chosen = feature["chosen"]      
        rejected = feature["rejected"]  


        # Presuming SEED-LLaMA is 'not self.is_encoder_decoder'
        if not isinstance(prompt, str):
            raise ValueError(f"prompt should be an str but got {type(prompt)}")
        
        # original
        prompt_tokens = self.tokenizer(prompt, padding="max_length", return_token_type_ids=False, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

        # prompt_tokens = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     return_token_type_ids=False,
        #     return_tensors="pt",
        # )
        # prompt_tokens = {f"prompt_{k}": v.squeeze(0) for k, v in prompt_tokens.items()} # squeeze 미수행시 오류 발생

        # Answer part stil consists of ONLY text.
        # Emu3 는 Llama-2-tokenizer 를 사용하지 않기 때문에, 바로 input sequence 를 만들어준다.
        if not isinstance(chosen, str):
            raise ValueError(f"chosen should be an str but got {type(chosen)}")
        # chosen_tokens = self.build_tokenized_answer(prompt, chosen) 
        chosen_input = prompt + chosen + self.tokenizer.eos_token
        chosen_tokens = self.tokenizer(
            chosen_input,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="pt",
        )
        chosen_labels = chosen_tokens["input_ids"]

        # loss mask (only answer;image)
        chosen_labels = torch.where(torch.logical_and(chosen_labels >= self.bov, chosen_labels <= self.eov), chosen_labels, self.label_pad_token_id) # =ignore_index
        chosen_tokens["labels"] = chosen_labels


        if not isinstance(rejected, str):
            raise ValueError(f"rejected should be an str but got {type(rejected)}")
        # rejected_tokens = self.build_tokenized_answer(prompt, rejected) 
        rejected_input = prompt + rejected + self.tokenizer.eos_token
        rejected_tokens = self.tokenizer(
            rejected_input,
            padding="max_length",
            return_token_type_ids=False,
            return_tensors="pt",
        )
        rejected_labels = rejected_tokens["input_ids"]

        # loss mask (only answer;image)
        rejected_labels = torch.where(torch.logical_and(rejected_labels >= self.bov, rejected_labels <= self.eov), rejected_labels, self.label_pad_token_id) # =ignore_index
        rejected_tokens["labels"] = rejected_labels


        # squeeze 
        for k, v in chosen_tokens.items():
            chosen_tokens[k] = v.squeeze(0)
        for k, v in rejected_tokens.items():
            rejected_tokens[k] = v.squeeze(0)


        # SANITY CHECK


        # # Last prompt token might get merged by tokenizer and
        # # it should not be included for generation if that happens
        # prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"]) # args is list.

        # chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        # rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        # prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

        # for k, v in prompt_tokens.items():
        #     prompt_tokens[k] = v[:prompt_len_input_ids]


        # Emu3 Tokenizer 에 맞춰 변경

        chosen_image_token_start = torch.where(chosen_tokens["input_ids"] == self.boi_token_id)[0]
        chosen_tokens["prompt_input_ids"] = chosen_tokens["input_ids"][:chosen_image_token_start]

        rejected_image_token_start = torch.where(rejected_tokens["input_ids"] == self.boi_token_id)[0]
        rejected_tokens["prompt_input_ids"] = rejected_tokens["input_ids"][:rejected_image_token_start]

        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)


        # Make sure prompts only have one different token at most
        # and length only differs by 1 at most
        num_diff_tokens = sum(
            [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
        )

        # print('# chosen_tokens["prompt_input_ids"]')
        # print(chosen_tokens["prompt_input_ids"])
        # print()
        # print('# rejected_tokens["prompt_input_ids"]')
        # print(rejected_tokens["prompt_input_ids"])

        # num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
        # if num_diff_tokens > 1 or num_diff_len > 1:
        #     raise ValueError(
        #         "Chosen and rejected prompt_input_ids might only differ on the "
        #         "last token due to tokenizer merge ops."
        #         f"# chosen_prompt_len_input_ids: {chosen_prompt_len_input_ids}"
        #         f"\n# rejected_prompt_len_input_ids: {rejected_prompt_len_input_ids}"
        #         f"\n# num_diff_tokens: {num_diff_tokens}"
                
        #     )

        if num_diff_tokens > 1:
            raise ValueError(
                "Chosen and rejected prompt_input_ids might only differ on the "
                "last token due to tokenizer merge ops."
                f"\n# Current num_diff_tokens: {num_diff_tokens}"
            )

        # add BOS token to head of prompt. Avoid adding if it's already there

        # Ensure BOS token at the start
        bos_token_id = self.tokenizer.bos_token_id

        # TENSOR VERSION
        # TODO: check device

        # prompt_tokens 자체는 불필요하므로 생략 
        # if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0].item():
        #     prompt_tokens["prompt_input_ids"] = torch.cat(
        #         (torch.tensor([bos_token_id], dtype=prompt_tokens["prompt_input_ids"].dtype, device=prompt_tokens["prompt_input_ids"].device),
        #         prompt_tokens["prompt_input_ids"])
        #     )
        #     prompt_tokens["prompt_attention_mask"] = torch.cat(
        #         (torch.tensor([1], dtype=prompt_tokens["prompt_attention_mask"].dtype, device=prompt_tokens["prompt_attention_mask"].device),
        #         prompt_tokens["prompt_attention_mask"])
        #     )

        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["input_ids"][0].item():
            chosen_tokens["input_ids"] = torch.cat(
                (torch.tensor([bos_token_id], dtype=chosen_tokens["input_ids"].dtype, device=chosen_tokens["input_ids"].device),
                chosen_tokens["input_ids"])
            )
            chosen_tokens["attention_mask"] = torch.cat(
                (torch.tensor([1], dtype=chosen_tokens["attention_mask"].dtype, device=chosen_tokens["attention_mask"].device),
                chosen_tokens["attention_mask"])
            )
            # (Emu3) labels 처리 추가
            chosen_tokens["labels"] = torch.cat(
                (torch.tensor([self.label_pad_token_id], dtype=chosen_tokens["labels"].dtype, device=chosen_tokens["labels"].device),
                chosen_tokens["labels"])
            )
 

        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["input_ids"][0].item():
            rejected_tokens["input_ids"] = torch.cat(
                (torch.tensor([bos_token_id], dtype=rejected_tokens["input_ids"].dtype, device=rejected_tokens["input_ids"].device),
                rejected_tokens["input_ids"])
            )
            rejected_tokens["attention_mask"] = torch.cat(
                (torch.tensor([1], dtype=rejected_tokens["attention_mask"].dtype, device=rejected_tokens["attention_mask"].device),
                rejected_tokens["attention_mask"])
            )
            # (Emu3) labels 처리 추가
            rejected_tokens["labels"] = torch.cat(
                (torch.tensor([self.label_pad_token_id], dtype=rejected_tokens["labels"].dtype, device=rejected_tokens["labels"].device),
                rejected_tokens["labels"])
            )

        # 생략, 이미 prompt 에 포함
        # Add EOS token to the end of the answer
        # eos_token_id = self.tokenizer.eos_token_id

        # if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1].item():
        #     chosen_tokens["input_ids"] = torch.cat(
        #         (chosen_tokens["input_ids"],
        #         torch.tensor([eos_token_id], dtype=chosen_tokens["input_ids"].dtype, device=chosen_tokens["input_ids"].device))
        #     )
        #     chosen_tokens["attention_mask"] = torch.cat(
        #         (chosen_tokens["attention_mask"],
        #         torch.tensor([1], dtype=chosen_tokens["attention_mask"].dtype, device=chosen_tokens["attention_mask"].device))
        #     )
        #     # (Emu3) labels 처리 추가
        #     chosen_tokens["labels"] = torch.cat(
        #         (chosen_tokens["labels"],
        #         torch.tensor([eos_token_id], dtype=chosen_tokens["labels"].dtype, device=chosen_tokens["labels"].device))
        #     )
            

        # if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1].item():
        #     rejected_tokens["input_ids"] = torch.cat(
        #         (rejected_tokens["input_ids"],
        #         torch.tensor([eos_token_id], dtype=rejected_tokens["input_ids"].dtype, device=rejected_tokens["input_ids"].device))
        #     )
        #     rejected_tokens["attention_mask"] = torch.cat(
        #         (rejected_tokens["attention_mask"],
        #         torch.tensor([1], dtype=rejected_tokens["attention_mask"].dtype, device=rejected_tokens["attention_mask"].device))
        #     )
        #     # (Emu3) labels 처리 추가
        #     rejected_tokens["labels"] = torch.cat(
        #         (rejected_tokens["labels"],
        #         torch.tensor([eos_token_id], dtype=rejected_tokens["labels"].dtype, device=rejected_tokens["labels"].device))
        #     )

        # longer_response_length = max(chosen_tokens["input_ids"].size(0), rejected_tokens["input_ids"].size(0))
        # longer_sequence_length = max(chosen_tokens["input_ids"].size(0), rejected_tokens["input_ids"].size(0))

        # Truncate the prompt if combined sequence is too long
        # for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        # if answer_tokens["prompt_input_ids"].size(0) + longer_response_length > self.max_length:
        #     if self.truncation_mode == "keep_start":
        #         answer_tokens["prompt_input_ids"] = answer_tokens["prompt_input_ids"][: self.max_prompt_length]
        #         answer_tokens["prompt_attention_mask"] = answer_tokens["prompt_attention_mask"][: self.max_prompt_length]
        #     elif self.truncation_mode == "keep_end":
        #         answer_tokens["prompt_input_ids"] = answer_tokens["prompt_input_ids"][-self.max_prompt_length:]
        #         answer_tokens["prompt_attention_mask"] = answer_tokens["prompt_attention_mask"][-self.max_prompt_length:]
        #     else:
        #         raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
            
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if answer_tokens["input_ids"].size(0) > self.max_length:
                if self.truncation_mode == "keep_start":
                    answer_tokens["input_ids"] = answer_tokens["input_ids"][: self.max_prompt_length]
                    answer_tokens["attention_mask"] = answer_tokens["attention_mask"][: self.max_prompt_length]
                    answer_tokens["labels"] = answer_tokens["labels"][: self.max_prompt_length]
            
                elif self.truncation_mode == "keep_end":
                    answer_tokens["input_ids"] = answer_tokens["input_ids"][-self.max_prompt_length:]
                    answer_tokens["attention_mask"] = answer_tokens["attention_mask"][-self.max_prompt_length:]
                    answer_tokens["labels"] = answer_tokens["labels"][-self.max_prompt_length:]
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # If still too long, truncate the response
        # for answer_tokens in [chosen_tokens, rejected_tokens]:
        #     if answer_tokens["prompt_input_ids"].size(0) + longer_response_length > self.max_length:
        #         trunc_length = self.max_length - self.max_prompt_length
        #         answer_tokens["input_ids"] = answer_tokens["input_ids"][:trunc_length]
        #         answer_tokens["attention_mask"] = answer_tokens["attention_mask"][:trunc_length]

        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if answer_tokens["input_ids"].size(0) > self.max_length:
                trunc_length = self.max_length - self.max_prompt_length
                answer_tokens["input_ids"] = answer_tokens["input_ids"][:trunc_length]
                answer_tokens["attention_mask"] = answer_tokens["attention_mask"][:trunc_length]
                answer_tokens["labels"] = answer_tokens["labels"][:trunc_length]


        # LIST VERSION
        # if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
        #     prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
        #     prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        # if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
        #     chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
        #     chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        # if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
        #     rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
        #     rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

        # # add EOS token to end of answer. Avoid adding if it's already there
        # eos_token_id = self.tokenizer.eos_token_id
        # if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        #     chosen_tokens["input_ids"].append(eos_token_id)
        #     chosen_tokens["attention_mask"].append(1)
        # if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        #     rejected_tokens["input_ids"].append(eos_token_id)
        #     rejected_tokens["attention_mask"].append(1)

        # longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))


        # # if combined sequence is too long, truncate the prompt
        # for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        #     if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
        #         if self.truncation_mode == "keep_start":
        #             for k in ["prompt_input_ids", "prompt_attention_mask"]:
        #                 answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
        #         elif self.truncation_mode == "keep_end":
        #             for k in ["prompt_input_ids", "prompt_attention_mask"]:
        #                 answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
        #         else:
        #             raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # # if that's still too long, truncate the response
        # for answer_tokens in [chosen_tokens, rejected_tokens]:
        #     if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
        #         for k in ["input_ids", "attention_mask"]:
        #             answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]


        # # Create labels
        # chosen_sequence_tokens = {
        #     k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
        # }
        # rejected_sequence_tokens = {
        #     k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
        # }
        # # chosen
        # chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        # # chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        # #     self.label_pad_token_id
        # # ] * len(chosen_tokens["prompt_input_ids"])
 
        # # rejected
        # rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        # # rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        # #     self.label_pad_token_id
        # # ] * len(rejected_tokens["prompt_input_ids"])
 

        for k, toks in {
            "chosen_": chosen_tokens, # chosen_sequence_tokens, # tensor
            "rejected_": rejected_tokens, # rejected_sequence_tokens, # tensor
            "": prompt_tokens, # list
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}{type_key}"] = tokens

        return batch
    
    

    # 공통; # Emu3 반영 완료
    def build_tokenized_answer(self, prompt, answer) -> Dict:
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
    

        # # 이렇게 토큰화할 시, 단순 리스트로 반환
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # full_tokenized = self.tokenizer(prompt + answer, padding="max_length", return_token_type_ids=False, add_special_tokens=False)
        # prompt_input_ids = self.tokenizer(prompt, padding="max_length", return_token_type_ids=False, add_special_tokens=False)["input_ids"]

        # print("\ndebug now")
        # print(full_tokenized)
        # print(prompt_input_ids) # [151849, 32, 19362, 374, 83450, 389, 279, 16359, 1933, 5209, 6923, 458, 2168, 13]

        # full_tokenized = self.tokenizer(
        #     prompt + answer,
        #     padding="max_length",
        #     return_token_type_ids=False,
        #     return_tensors="pt",
        # )

        # prompt_input_ids = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     return_token_type_ids=False,
        #     return_tensors="pt",
        # )["input_ids"].squeeze(0)

        # # specific for Emu3Tokenizer
        # for k, v in full_tokenized.items():
        #     full_tokenized[k] = v.squeeze(0)


        # print("\ndebug now")
        # print(full_tokenized)
        # print(prompt_input_ids) 


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

        # print("\n# debug 1")
        # print(response_token_ids_start_idx)

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

        # print("\n# debug 2")
        # print(prompt_input_ids)

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
    

    # for Emu3
    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    # for Emu3
    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

    # for Emu3
    def to_list(self):
        return [self[i] for i in range(len(self))]



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
    