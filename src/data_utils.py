import json

from datasets import load_dataset
from transformers import AutoTokenizer



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


def get_doc_message_from_list(doc_form_list):
    result = []
    for doc_form in doc_form_list:
        result.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": doc_form},
        ])
    return result


def get_student_doc_tamplate_text(tokenizer, msg_list):
    template_result = []
    for msg in msg_list:
        template_res = tokenizer.apply_chat_template(msg,
                                                     tokenize=False,  # 返回张量
                                                     add_generation_prompt=True)
        doc_template = cut_tag_tail(template_res, '<|im_start|>')
        template_result.append(doc_template)
    return template_result


def get_student_doc_tamplate_text_single_line(tokenizer, msg):
    template_res = tokenizer.apply_chat_template(msg,
                                                 tokenize=False,  # 返回张量
                                                 add_generation_prompt=True)
    doc_template = cut_tag_tail(template_res, '<|im_start|>')
    return doc_template


def get_student_query_tamplate_text(tokenizer, examples):
    template_result = []
    # print(examples["messages"])
    for index in range(len(examples["messages"])):
        # print("messages",examples["messages"][index])
        template_res = tokenizer.apply_chat_template(examples["messages"][index],
                                                     tokenize=False,  # 返回张量
                                                     add_generation_prompt=True)
        # print("template res",template_res)
        template_res = template_res.split(examples["doc_form"][index])[1]
        # template_res = cut_tag_tail(template_res, '<|im_start|>')
        template_result.append(template_res)
    return template_result


def get_teacher_query_tamplate_text(tokenizer, path):
    content = open(path, 'r', encoding='utf-8').readlines()
    template_result = []
    for line in content:
        obj = json.loads(line)
        template_res = tokenizer.apply_chat_template(obj["messages"],
                                                     tokenize=False,  # 返回张量
                                                     add_generation_prompt=True)
        template_res = cut_tag_tail(template_res, '<|im_start|>')
        template_result.append(template_res)
    return template_result


def get_tokens(template_list, tokenizer, padding_side, max_length):
    """
    进行tokenizer过程，得到 inputs和atmk
    :param template_list:
    :param tokenizer:
    :param padding_side:
    :param max_length:
    :return:
    """
    tokenizer.padding_side = padding_side
    token_list = tokenizer(template_list, add_special_tokens=True, max_length=max_length,
                           padding="max_length")
    return token_list


def load_and_preprocess_data(model_path, data_path, split_position=32):
    dataset = load_dataset("json",
                           data_files=data_path,
                           )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding=True

    def preprocess_function(examples):
        student_doc_msg_list = get_doc_message_from_list(examples["doc_form"])
        student_doc_template_list = get_student_doc_tamplate_text(tokenizer, student_doc_msg_list)
        student_doc_tokens = get_tokens(student_doc_template_list, tokenizer, padding_side="left",
                                        max_length=split_position)

        student_query_template = get_student_query_tamplate_text(tokenizer, examples)
        student_query_tokens = get_tokens(student_query_template, tokenizer, padding_side="right",
                                          max_length=4096 - split_position)

        teacher_input_ids = []
        teacher_atmk_list = []
        for i in range(len(student_query_tokens["input_ids"])):
            teacher_input_ids.append(student_doc_tokens["input_ids"][i] + student_query_tokens["input_ids"][i])
            teacher_atmk_list.append(
                student_doc_tokens["attention_mask"][i] + student_query_tokens["attention_mask"][i])

        return {
            "input_ids": teacher_input_ids,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_atmk_list,
            "student_doc_input_ids": student_doc_tokens["input_ids"],
            "student_doc_attention_mask": student_doc_tokens["attention_mask"],
            "student_query_input_ids": student_query_tokens["input_ids"],
            "student_query_attention_mask": student_query_tokens["attention_mask"],
            "label":examples["label"]
        }

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset,tokenizer


def load_and_preprocess_data_infer(model_path, data_path, split_position=32):
    dataset = load_dataset("json",
                           data_files=data_path,
                           )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding=True

    def preprocess_function(examples):
        student_doc_msg_list = get_doc_message_from_list(examples["doc_form"])
        student_doc_template_list = get_student_doc_tamplate_text(tokenizer, student_doc_msg_list)
        student_doc_tokens = get_tokens(student_doc_template_list, tokenizer, padding_side="left",
                                        max_length=split_position)

        student_query_template = get_student_query_tamplate_text(tokenizer, examples)
        student_query_tokens = get_tokens(student_query_template, tokenizer, padding_side="right",
                                          max_length=4096 - split_position)

        teacher_input_ids = []
        teacher_atmk_list = []
        for i in range(len(student_query_tokens["input_ids"])):
            teacher_input_ids.append(student_doc_tokens["input_ids"][i] + student_query_tokens["input_ids"][i])
            teacher_atmk_list.append(
                student_doc_tokens["attention_mask"][i] + student_query_tokens["attention_mask"][i])

        return {
            "input_ids": teacher_input_ids,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_atmk_list,
            "student_doc_input_ids": student_doc_tokens["input_ids"],
            "student_doc_attention_mask": student_doc_tokens["attention_mask"],
            "student_query_input_ids": student_query_tokens["input_ids"],
            "student_query_attention_mask": student_query_tokens["attention_mask"],

        }

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset,tokenizer



