import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer
import argparse
import os
import json
from llava.model import LlavaLlamaForCausalLM
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle

from tqdm import tqdm
import time

def model_fn(model_dir, use_flash_attention_2=False, load_in_4bit=False):
    kwargs = {"device_map": "auto"}
    kwargs["torch_dtype"] = torch.float16
    kwargs["use_flash_attention_2"] = use_flash_attention_2
    kwargs["load_in_4bit"] = load_in_4bit
    #update for flashattention, pip install flash-attn==2.5.8 --no-build-isolation
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_dir, low_cpu_mem_usage=True, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)
    image_processor = vision_tower.image_processor
    return model, tokenizer, image_processor

def cal_acc(res_ls):
    res = 0
    gt_label_ls = []
    pred_ls = []

    for i in tqdm(res_ls):
        gt_label_value = i['gt_title'].split(" ")[-1].replace(".", "")
        pred_value = i['gen_title'].split(" ")[-1].replace(".", "")
        gt_label_ls.append(gt_label_value)
        pred_ls.append(pred_value)
        if gt_label_value == pred_value:
            res = res + 1

    print("<<< accuracy: ", res / len(res_ls))
    return

def predict_fn(image_file, raw_prompt, model, tokenizer, image_processor):
    # get prompt & parameters
    max_new_tokens = 1024
    temperature = 0.2

    conv_mode = 'llava_v1'

    # use raw_prompt as prompt
    if conv_mode == "raw":
        # use raw_prompt as prompt
        prompt = raw_prompt
        stop_str = "###"
    else:
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        inp = f"{roles[0]}: {raw_prompt}"
        inp = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + inp
        )
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    disable_torch_init()
    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )

    keywords = [stop_str]
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
           # do_sample=True,
           # temperature=temperature,
            max_new_tokens=max_new_tokens,
            #use_cache=True,
            #stopping_criteria=[stopping_criteria],
        )
    outputs = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    ).strip()
    return outputs

def main(model_dir,test_json_file, save_file_name, use_flash_attention_2, load_in_4bit):
    res_ls = []
    # unpack model and tokenizer
    model, tokenizer, image_processor = model_fn(model_dir, use_flash_attention_2, load_in_4bit)

    # Open the JSON file in read mode
    with open(test_json_file, "r") as f:
        # Parse the JSON data and store it in a variable
        data = json.load(f)

    output_folder = os.path.dirname(test_json_file)
    start_time = time.time()
    #loop over test files
    for i in tqdm(data):
        image_file = os.path.join(output_folder,i['image'])
        raw_prompt = i['conversations'][0]['value'][8:]
        #print("raw_prompt: ", raw_prompt)
        gt_title = i['conversations'][1]['value']
        res = predict_fn(image_file, raw_prompt, model, tokenizer, image_processor)
        #print ("res: ", res)
        res_dict = {'image': image_file,
                    'input': raw_prompt,
                    'gt_title': gt_title,
                     'gen_title': res}

        res_ls.append(res_dict)

    end_time = time.time()
    elapsed_time_in_seconds = end_time - start_time
    elapsed_time_in_minutes = elapsed_time_in_seconds / 60 / len(data) * 1000
    print(f"Code block took {elapsed_time_in_minutes:.2f} minutes to execute 1000 images")

    #output_test
    with open(os.path.join(output_folder, save_file_name), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(res_ls, f)  # Optional parameter for indentation
    print('Data written to json')

    # calculate accuracy score
    cal_acc(res_ls)

    return res_ls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_json_file', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--save_file_name', default='pred_res.json' ,type=str)
    parser.add_argument('--use_flash_attention_2', default=False, type=bool)
    parser.add_argument('--load_in_4bit', default=False, type=bool)
    args = parser.parse_args()

    main(args.model_dir, args.test_json_file, args.save_file_name, args.use_flash_attention_2, args.load_in_4bit)
