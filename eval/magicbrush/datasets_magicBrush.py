# -*- coding: utf-8 -*-

import json
import os.path as osp
import random

import torch
from torch.utils.data import Dataset


# TODO: args
class MagicBrushFeatureDataset(Dataset):

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__()

        self.args = args
        with open(args.data_path) as f:
            d = json.load(f)

        self.path_prefix = d["prefix"]
        self.filelist = d["path_list"]

        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]

        # (only for test) 원 데이터
        self.json_path = "/data/visual_llama_eval/magicbrush_test/edit_turns.json"
        with open(self.json_path) as f:
            self.raw_data = json.load(f)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        try:
            path = osp.join(self.path_prefix, self.filelist[index])
            data = torch.load(path)

            # metadata
            gen_img_id = data["name"]
            prompt = data["edit_instruction"]
            # 원 데이터(json) 와 비교대조하여 추출
            save_img_name = self.find_save_img_name(gen_img_id, prompt)



            # input
            input_image_tokens = data["input_images"]
            input_image_prompt = self.format_image_prompt(input_image_tokens)


            # genereate sequence (not GT)
            input = self.tokenizer.bos_token + input_image_prompt + prompt 

            # tokenize sequence
            sample = self.tokenizer(
                input,
                padding="max_length",
                max_length=16450, # 추가
                return_token_type_ids=False,
                return_tensors="pt",
            )

            # delete labels (No Need)

            # TODO: mandatory? CHECK
            # for k, v in sample.items():
            #     sample[k] = v.squeeze(0)

            # return sample
            return {
                "input_ids":  sample["input_ids"],
                "gen_img_id": gen_img_id,
                "save_img_name": save_img_name
            }

        except Exception as e:
            print(e)
            return self.__getitem__(index+1)
        

    def find_save_img_name(self, id_text, instruction_text):
        # Loop through the JSON list
        for sample in self.raw_data:
            # Extract the base ID from the input file name
            sample_id = sample["input"].split('-')[0]

            # Match the ID and instruction
            if sample_id == id_text and sample["instruction"] == instruction_text:
                # Set save_img_name
                if "input" in sample["input"]:
                    save_img_name = f"{sample_id}_1.png"
                else:
                    turn_id = int(sample["input"].split('output')[1].split('.png')[0]) + 1
                    save_img_name = f"{sample_id}_inde_{turn_id}.png"
                return save_img_name
            
        return None  # Return None if no match is found



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
    

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

