import os.path as osp
import json
import torch
from torch.utils.data import Dataset

# HumanEdit dataset 기준
class PreferenceHumanEditDatasetEmu3(Dataset):
    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer", split="train"):
        super().__init__()

        self.args = args
        if split == "train":
            with open(args.train_data_path) as f:
                d = json.load(f)
        elif split == "val":
            with open(args.val_data_path) as f:
                d = json.load(f)
        else:
            raise NotImplementedError(f"Unknown split: {split} !")

        self.path_prefix = d["prefix"]
        self.filelist = d["path_list"][:10] # size adjust 가능

        self.tokenizer = tokenizer
        # self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        # self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        path = osp.join(self.path_prefix, self.filelist[index])
        data = torch.load(path)

        # input
        input_image_tokens = data["input_images"]
        input_image_prompt = self.format_image_prompt(input_image_tokens)

        output_image_tokens = data["output_images"]
        output_image_prompt = self.format_image_prompt(output_image_tokens)
        
        # edit_instruction = data["edit_instruction"]
        output_description = data["output_description"] # != output_caption

        return {
            "prompt": output_description,       # text
            "chosen": output_image_prompt,      # image token 1
            "rejected": input_image_prompt      # image token 2
            } 


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

    def to_list(self):
        return [self[i] for i in range(len(self))]