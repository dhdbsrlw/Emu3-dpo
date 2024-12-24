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

# conda activate emu3
# cd /home/yjoh/project/Emu3-dpo/emu3/train

# 15000MB 소요

# CUDA_VISIBLE_DEVICES=1 python /home/yjoh/project/Emu3-dpo/emu3/train/prepare_data.py --model_path BAAI/Emu3-VisionTokenizer --output_path /nas2/preference/emu3_tokenized/human_edit_val

"""
 {
        "image_id": "BEwrVP6o0yQ",
        "editing_type": "Action",
        "core": 0,
        "mask": 1,
        "editing_instruction": "Have the boy stretch out his hands.",
        "output_description": "The boy standing on the ground with outstretched hands",
        "input_caption_by_llama": "A man walks along a desert trail, gazing out at the vast expanse of Monument Valley in Arizona. The landscape is characterized by its distinctive red rock formations and sparse vegetation, with the iconic \"Mittens\" monoliths standing tall in the distance. The hazy sky adds to the sense of desolation and isolation, as the man's back is turned to the camera, lost in thought as he takes in the breathtaking view.",
        "output_caption_by_llama": "The image depicts a man standing in the desert, with his back to the camera, and his arms outstretched in a gesture of freedom and joy. He is wearing a dark green t-shirt, black shorts, and sneakers, and has dark hair. The desert landscape stretches out behind him, with red rock formations and sparse vegetation. The sky above is bright and hazy, suggesting a hot and sunny day. The overall atmosphere of the image is one of liberation and exhilaration, as if the man is embracing the vastness and beauty of the desert.",
        "input_img": "BEwrVP6o0yQ_input.jpg",
        "mask_img": "BEwrVP6o0yQ_mask.jpg",
        "output_img": "BEwrVP6o0yQ_output.jpg"
    },
"""

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='vision tokenizer path')
    parser.add_argument('--data_path', type=str, help='data path', default='/nas2/preference/HumanEdit/data/processed/split_ver/HumanEdit_val.json')
    parser.add_argument('--image_path', type=str, help='image path', default='/nas2/preference/HumanEdit/images') # 추가
    parser.add_argument('--cache_dir', type=str, default='/nas2/checkpoints/hf_cache_yj') # 추가 (고정)
    parser.add_argument('--output_path', type=str, help='tokenized data save path')
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

        name = inp["image_id"]
        edit_type = inp["editing_type"]
        edit_instruction = inp["editing_instruction"]
        output_description = inp["output_description"]
        input_caption = inp["input_caption_by_llama"]
        output_caption = inp["output_caption_by_llama"]

        # Image
        # "input_img": "BEwrVP6o0yQ_input.jpg",
        # "mask_img": "BEwrVP6o0yQ_mask.jpg",
        # "output_img": "BEwrVP6o0yQ_output.jpg"

        input_image_path = os.path.join(args.image_path, name, inp["input_img"])
        output_image_path = os.path.join(args.image_path, name, inp["output_img"])
        

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

        
        data = {
            "name": name,
            "edit_type": edit_type,
            "edit_instruction": edit_instruction,
            "output_description": output_description,
            "input_caption": input_caption,
            "output_caption": output_caption,
            "input_images": input_token_ids,
            "output_images": output_token_ids,
        }

        torch.save(data, f"{args.output_path}/feature/{name}.pth")
        datalist["path_list"].append(f"{name}.pth")

    with open(f"{args.output_path}/list/train.json", 'w') as f:
        json.dump(datalist, f)


if __name__ == "__main__":
    main()
