import os
import boto3
import json
import torch
import logging
import requests
from io import BytesIO

from PIL import Image
from djl_python import Input, Output

from llava.model import LlavaLlamaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import AutoTokenizer
# from transformers import TextStreamer

model_dict = None


class ModelConfig:
    def __init__(self):
        image_aspect_ratio = "pad"


def load_model(properties):
    s3 = boto3.client('s3')

    disable_torch_init()

    model_location = properties["model_dir"]
    if "model_id" in properties:
        model_location = properties["model_id"]
    logging.info(f"Loading model from: {model_location}")

    model_dir_list = os.listdir(model_location)
    logging.info(f"Dir file list : {model_dir_list}")

    model_path = model_location

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None, model_name="llava", load_4bit=True)
    model_cfg = ModelConfig()
    model_dict = {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "model_cfg": model_cfg
    }

    return model_dict


def handle(inputs: Input):
    global model_dict

    if not model_dict:
        model_dict = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    image_processor = model_dict['image_processor']

    data = inputs.get_as_json()
    raw_prompt = data["text"][0]
    image_file = data["input_image"]

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
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=None,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    return Output().add(output)
