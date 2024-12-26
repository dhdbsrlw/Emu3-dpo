# -*- coding: utf-8 -*-

import json
import os.path as osp
import random

import torch
from torch.utils.data import Dataset


class Emu3FeatureDataset(Dataset):

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

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        try:
            path = osp.join(self.path_prefix, self.filelist[index])
            data = torch.load(path)

            # input
            input_image_tokens = data["input_images"]
            input_image_prompt = self.format_image_prompt(input_image_tokens)

            output_image_tokens = data["output_images"]
            output_image_prompt = self.format_image_prompt(output_image_tokens)

            # p_prob = random.random()
            # if p_prob < self.args.null_prompt_prob:
            #     prompt = ""
            # else:
            #     # prompt = data["texts"]
            
            prompt = data["edit_instruction"]

            # genereate sequence
            # input = self.tokenizer.bos_token + prompt + image_prompt
            input = self.tokenizer.bos_token + input_image_prompt + prompt + output_image_prompt
            # tokenize sequence
            sample = self.tokenizer(
                input,
                padding="max_length",
                max_length=16450, # 추가
                return_token_type_ids=False,
                return_tensors="pt",
            )

            labels = sample["input_ids"]
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            # TODO: check len(labels)
            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # print("# sample: ", sample['input_ids'].shape)
            # print("# labels: ", labels)
            # print(len(labels[0]))

            return sample

        except Exception as e:
            print(e)
            return self.__getitem__(index+1)

        

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


# debug
# if __name__ == "__main__":
