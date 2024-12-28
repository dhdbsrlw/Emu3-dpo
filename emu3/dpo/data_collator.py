import os
from dataclasses import dataclass
from typing import Dict, List, Any
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image
from tqdm import tqdm
import torch



# T2I Task ver.
@dataclass
class DPODataCollatorEmu3(DPODataCollatorWithPadding):
    # def __init__(self, model, tokenizer: "Emu3Tokenizer", image_tokenizer: "Emu3VisionTokenizer", args: "DataArguments"):
    def __init__(self, model, tokenizer: "Emu3Tokenizer", args: "DataArguments"):
        super().__init__()
        self.model=model
        self.tokenizer=tokenizer

        # 특이점 1) Human instruction 추가 
        self.instruction="Please generate an image."
        # TODO: User/Assistant 추가?
        
        # set all 
        # for k, v in collate_cfg.items():
        #     setattr(self, k, v)
        self.args = args
        self.ignore_index=-100 # TODO: 임시

        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]


    # ImageQA Task - VLFeedfack
    # T2I Task - MagicBrush 기준 (pretokenized feature 사용)
    def tokenize_batch_element(
        # self,
        # prompt_image: List,
        # prompt_text: str,
        # chosen: str,    # answer 1
        # rejected: str,  # answer 2
        # img_path: str, 
        self,
        prompt: str,    # prompt text
        chosen_image_prompt: str,    # chosen img prompt (text 취급)
        rejected_image_prompt: str,  # rejected img prompt (text 취급)
    ) -> Dict:
        
        batch = {}
        
        # (1) generate `chosen` sequence
        # input = self.tokenizer.bos_token + input_image_prompt + prompt + output_image_prompt > EDIT
        chosen_input = self.tokenizer.bos_token + prompt + " " + self.instruction + chosen_image_prompt

        # (2) generate `chosen` sequence
        rejected_input = self.tokenizer.bos_token + prompt + " " + self.instruction + rejected_image_prompt

        # print("# chosen_image_prompt: ", len(chosen_image_prompt)) 
        # chosen_image_prompt:  95101 - 256
        # chosen_image_prompt:  187531 - 720

        # (3) tokenize each sequence
        # TODO: EOS token ?
        chosen_sample = self.tokenizer(
            chosen_input,
            padding="max_length",
            # max_length=16450, # 추가
            return_token_type_ids=False,
            return_tensors="pt",
        )
        # TODO: check chosen sample includes attention_mask
        chosen_labels = chosen_sample["input_ids"]
        # loss mask (only answer;image)
        chosen_labels = torch.where(torch.logical_and(chosen_labels >= self.bov, chosen_labels <= self.eov), chosen_labels, self.ignore_index)
        chosen_sample["labels"] = chosen_labels

        # max_position_embeddings 의 값에 따라 최종 input_ids 의 토큰 개수가 결정된다. (디버깅으로 확인)
        rejected_sample = self.tokenizer(
            rejected_input,
            padding="max_length",
            # max_length=16450, # 추가
            return_token_type_ids=False,
            return_tensors="pt",
        )
        rejected_labels = rejected_sample["input_ids"]

        # loss mask 
        rejected_labels = torch.where(torch.logical_and(rejected_labels >= self.bov, rejected_labels <= self.eov), rejected_labels, self.ignore_index)
        rejected_sample["labels"] = rejected_labels

        ### 시퀀스 모두 완성 ### 

        # squeeze 하지 않으면 오류 발생 확인
        # for k, v in chosen_sample.items():
        #     chosen_sample[k] = v.squeeze(0)
        # for k, v in rejected_sample.items():
        #     rejected_sample[k] = v.squeeze(0)

        batch["chosen"] = chosen_sample
        batch["rejected"] = rejected_sample

        # print("# batch-chosen: ", len(chosen_sample["input_ids"][0]))  ==max_position_embeddings (in cfg)

        # print("# in tokenize_batch_element")
        # print(batch["chosen"]["input_ids"].shape) # torch.Size([1, 8210])

        return batch


        # (4) pad / truncate 처리
        longer_sequence_length = max(len(chosen_sample["input_ids"]), len(rejected_sample["input_ids"]))

        chosen_input_ids, chosen_attention_mask, chosen_labels = self.pad_or_truncate(longer_sequence_length, chosen_sample["input_ids"], chosen_sample["attention_mask"], chosen_labels)
        rejected_input_ids, rejected_attention_mask, rejected_labels = self.pad_or_truncate(longer_sequence_length, rejected_sample["input_ids"], rejected_sample["attention_mask"], rejected_labels)

        
        # TODO: necessary check (squeeze)
        # for k, v in sample.items():
        #     sample[k] = v.squeeze(0)
    

        # 최종 Output (squeeze ver.)
        # batch["chosen_input_ids"] = chosen_input_ids.squeeze(0)
        # batch["chosen_attention_mask"] = chosen_attention_mask.squeeze(0)
        # batch["chosen_labels"] = chosen_labels.squeeze(0)
        
        # batch["rejected_input_ids"] = rejected_input_ids.squeeze(0)
        # batch["rejected_attention_mask"] = rejected_attention_mask.squeeze(0)
        # batch["rejected_labels"] = rejected_labels.squeeze(0)


        batch["chosen_input_ids"] = chosen_input_ids
        batch["chosen_attention_mask"] = chosen_attention_mask
        batch["chosen_labels"] = chosen_labels
        
        batch["rejected_input_ids"] = rejected_input_ids
        batch["rejected_attention_mask"] = rejected_attention_mask
        batch["rejected_labels"] = rejected_labels

        return batch
        



    def pad_or_truncate(self, length, input_ids=None, attention_mask=None, labels=None):

        if (input_ids is None) or (attention_mask is None) or (labels is None):
            # raise ValueError("One of the elements in the tokenized is missing !")
            print("One of the elements in the tokenized is missing !")

        if len(input_ids) >= length:
                input_ids = input_ids[:length]
                attention_mask = attention_mask[:length]
                labels = labels[:length]
        else:
            padding_length = length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

        # They are already a tensor.
        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # labels = torch.tensor(labels, dtype=torch.long)


        return input_ids, attention_mask, labels
         

    
    def collate(self, example):
        
        # print("\n# debug")
        # print("example: ", example)
        batch = {}

        # 주의 index 0
        chosen_sample = example[0]["chosen"]
        rejected_sample = example[0]["rejected"]

        # (4) pad / truncate 처리
        longer_sequence_length = max(len(chosen_sample["input_ids"]), len(rejected_sample["input_ids"]))
        # print("# Longer Seq: ", longer_sequence_length)
        # print("# Chosen sample: ", chosen_sample)
        # print(chosen_sample["input_ids"].shape)

        chosen_input_ids, chosen_attention_mask, chosen_labels = self.pad_or_truncate(longer_sequence_length, chosen_sample["input_ids"], chosen_sample["attention_mask"], chosen_sample["labels"])
        rejected_input_ids, rejected_attention_mask, rejected_labels = self.pad_or_truncate(longer_sequence_length, rejected_sample["input_ids"], rejected_sample["attention_mask"], rejected_sample["labels"])

        batch["chosen_input_ids"] = chosen_input_ids
        batch["chosen_attention_mask"] = chosen_attention_mask
        batch["chosen_labels"] = chosen_labels
        
        batch["rejected_input_ids"] = rejected_input_ids
        batch["rejected_attention_mask"] = rejected_attention_mask
        batch["rejected_labels"] = rejected_labels


        # print("# in collate")
        # print(chosen_input_ids.shape) # torch.Size([1, 8213])

        return batch

        # 작성 중 중단
        # 필요하지 않은 것 같다.
        # This is implemented based on https://github.com/huggingface/trl/blob/v0.12.2/trl/trainer/dpo_trainer.py#L81
    
        # # Convert to tensor

        # chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        # chosen_labels = [torch.tensor(example["chosen_labels"]) for example in examples]
        # chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]

        # rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        # rejected_labels = [torch.tensor(example["rejected_labels"]) for example in examples]
        # rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]

        # # # They are already torch.Tensor.
        # # chosen_pixel_values = [example["chosen_pixel_values"] for example in examples]
        # # rejected_pixel_values = [example["rejected_pixel_values"] for example in examples]


        # # Pad
        # output = {}
        # output["chosen_input_ids"] = pad(prompt_input_ids, padding_value=self.tokenizer.pad_token_id, padding_side="left")
        # output["chosen_labels"] = pad(prompt_labels, padding_value=self.label_pad_token_id, padding_side="left")
        # output["chosen_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        
        # return output


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

            # 이때 features 는 하나의 데이터셋 클래스(데이터셋 종류) 로 볼 수 있다.

            # (참고)
            #  return {
            # "prompt": output_description,       # text
            # "chosen": output_image_prompt,      # image token 1
            # "rejected": input_image_prompt      # image token 2
            # } 

            for feature in features:
                # 아래 셋 모두 prompt to be need tokenized
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]

                batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
                tokenized_batch.append(batch_element)

            collated_batch = self.collate(tokenized_batch)

            return collated_batch


    
