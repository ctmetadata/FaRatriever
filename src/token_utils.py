import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer




def cut_tag_tail(template_text, end_tag):

    last_end_pos = template_text.rfind(end_tag)
    part_before = template_text[:last_end_pos]
    return part_before


def get_doc_message(path):
    content = open(path, 'r', encoding='utf-8').readlines()
    result = []
    for line in content:
        obj = json.loads(line)
        result.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": obj["doc_form"]},
        ])
    return result


def get_student_doc_tamplate_text(tokenizer, msg_list):
    template_result = []
    for msg in msg_list:
        template_res = tokenizer.apply_chat_template(msg,
                                                     tokenize=False,
                                                     add_generation_prompt=True)
        doc_template = cut_tag_tail(template_res, '<|im_start|>')
        template_result.append(doc_template)
    return template_result


def get_student_query_tamplate_text(tokenizer, path):
    content = open(path, 'r', encoding='utf-8').readlines()
    template_result = []
    for line in content:
        obj = json.loads(line)
        template_res = tokenizer.apply_chat_template(obj["messages"],
                                                     tokenize=False,
                                                     add_generation_prompt=True)
        template_res = template_res.split(obj["doc_form"])[1]
        template_res = cut_tag_tail(template_res, '<|im_start|>')
        template_result.append(template_res)
    return template_result


def get_teacher_query_tamplate_text(tokenizer, path):
    content = open(path, 'r', encoding='utf-8').readlines()
    template_result = []
    for line in content:
        obj = json.loads(line)
        template_res = tokenizer.apply_chat_template(obj["messages"],
                                                     tokenize=False,
                                                     add_generation_prompt=True)

        template_res = cut_tag_tail(template_res, '<|im_start|>')
        template_result.append(template_res)
    return template_result


def get_tokens(template_list, tokenizer, padding_side, max_length):

    tokenizer.padding_side = padding_side
    token_list = tokenizer(template_list, add_special_tokens=True, max_length=max_length,
                           padding="max_length")
    return token_list


def decode_tokens(input_ids):
    single_tensor = torch.tensor(input_ids)

    single_decoded = tokenizer.decode(
    single_tensor,
    skip_special_tokens=False
    )
    single_decoded_no_special = tokenizer.decode(
    single_tensor,
    skip_special_tokens=True
    )

    print(f"{single_decoded}")
    print(f"{single_decoded_no_special}")


#

