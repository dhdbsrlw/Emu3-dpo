# -*- coding: utf-8 -*-

import argparse
import json
import os

from PIL import Image
import torch

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor
from tqdm import tqdm

# conda activate emu3_dpo
# cd /home/yjoh/project/Emu3-dpo/eval/magicbrush
# CUDA_VISIBLE_DEVICES=5 python /home/yjoh/project/Emu3-dpo/eval/magicbrush/prepare_data.py 

"""
 {
    "input": "242679-input.png",
    "mask": "242679-mask1.png",
    "output": "242679-output1.png",
    "instruction": "Put a cat on the seat."
},
"""

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='vision tokenizer path', default='BAAI/Emu3-VisionTokenizer')
    parser.add_argument('--data_path', type=str, help='data path', default='/data/visual_llama_eval/magicbrush_test/edit_turns.json')
    parser.add_argument('--image_path', type=str, help='image path', default='/data/visual_llama_eval/magicbrush_test/images')
    parser.add_argument('--output_path', type=str, help='tokenized data save path', default='/nas2/preference/emu3_tokenized/magicbrush_test')
    parser.add_argument('--image_area', type=int, default=720 * 720)

    args = parser.parse_args()
    return args


def smart_resize(image, image_area: int = 720 * 720):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image


def main():
    args = prepare_args()

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path) #, cache_dir=args.cache_dir)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(args.model_path, device_map="cuda:0") #, cache_dir=args.cache_dir)
    image_tokenizer.eval()

    os.makedirs(f"{args.output_path}/feature", exist_ok=True)
    os.makedirs(f"{args.output_path}/list", exist_ok=True)

    datalist = {
        "prefix": f"{args.output_path}/feature",
        "path_list": []
    }

    with open(args.data_path) as f:
        input_data = json.load(f)

    for inp in tqdm(input_data, desc="Tokenizing data ..."):

        # name = inp["name"]
        # prompt = inp["text"]

        # HumanEdit
        # name = inp["image_id"]
        # edit_type = inp["editing_type"]
        # edit_instruction = inp["editing_instruction"]
        # output_description = inp["output_description"]
        # input_caption = inp["input_caption_by_llama"]
        # output_caption = inp["output_caption_by_llama"]
        # Image
        # "input_img": "BEwrVP6o0yQ_input.jpg",
        # "mask_img": "BEwrVP6o0yQ_mask.jpg",
        # "output_img": "BEwrVP6o0yQ_output.jpg"

        # MagicBrush
        name = inp["input"].split("-")[0]
        # print("name: ", name)
        edit_instruction = inp["instruction"]

        input_image_path = os.path.join(args.image_path, name, inp["input"])
        output_image_path = os.path.join(args.image_path, name, inp["output"])
        mask_image_path = os.path.join(args.image_path, name, inp["mask"])
        

        # input
        input_image = Image.open(input_image_path).convert("RGB")
        input_image = smart_resize(input_image, args.image_area)

        input_image = image_processor(input_image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            input_image = input_image.cuda()
            input_token_ids = image_tokenizer.encode(input_image)

        input_token_ids = input_token_ids.squeeze(0).cpu().numpy()


        # output
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = smart_resize(output_image, args.image_area)

        output_image = image_processor(output_image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            output_image = output_image.cuda()
            output_token_ids = image_tokenizer.encode(output_image)

        output_token_ids = output_token_ids.squeeze(0).cpu().numpy()


        # mask
        mask_image = Image.open(mask_image_path).convert("RGB")
        mask_image = smart_resize(mask_image, args.image_area)

        mask_image = image_processor(mask_image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            mask_image = mask_image.cuda()
            mask_token_ids = image_tokenizer.encode(mask_image)

        mask_token_ids = mask_token_ids.squeeze(0).cpu().numpy()

        # HumanEdit
        # data = {
        #     "name": name,
        #     "edit_type": edit_type,
        #     "edit_instruction": edit_instruction,
        #     "output_description": output_description,
        #     "input_caption": input_caption,
        #     "output_caption": output_caption,
        #     "input_images": input_token_ids,
        #     "output_images": output_token_ids,
        # }

        # MagicBrush
        data = {
            "name": name,
            "edit_instruction": edit_instruction,
            "input_images": input_token_ids,
            "output_images": output_token_ids,
            "mask_images": mask_token_ids,
        }

        torch.save(data, f"{args.output_path}/feature/{name}.pth")
        datalist["path_list"].append(f"{name}.pth")

    with open(f"{args.output_path}/list/test.json", 'w') as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    main()
