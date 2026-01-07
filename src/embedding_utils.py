import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer



def forward_doc(sdoc_input_ids, sdoc_attention_mask, model, batch_size=2):
    device = next(model.parameters()).device
    total_samples = sdoc_input_ids.shape[0]
    hidden_states_list = []

    if not isinstance(sdoc_input_ids, torch.Tensor):
        sdoc_input_ids = torch.tensor(sdoc_input_ids)
    if not isinstance(sdoc_attention_mask, torch.Tensor):
        sdoc_attention_mask = torch.tensor(sdoc_attention_mask)

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_ids = sdoc_input_ids[i:i + batch_size].to(device, non_blocking=True)
            batch_mask = sdoc_attention_mask[i:i + batch_size].to(device, non_blocking=True)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask, output_hidden_states=True)
            last_sdoc_embedding = outputs.hidden_states[-1].cpu()
            hidden_states_list.append(last_sdoc_embedding)  # B seq dim ([10, 2048, 3584])
            del batch_ids, batch_mask, outputs
            torch.cuda.empty_cache()
    all_hidden_states = torch.cat(hidden_states_list, dim=0)

    assert all_hidden_states.shape[0] == total_samples, \
        f"deal sample not matfh：input {total_samples}，output {all_hidden_states.shape[0]}"

    return all_hidden_states


def split_hidden_state(hidden_state, split_position=32):

    if len(hidden_state.shape) != 3:
        raise ValueError(f"hidden_state should be 3 dimision，actual shape is {hidden_state.shape}")

    batch_size, seq_len, hidden_size = hidden_state.shape

    if seq_len <= split_position:
        part1 = hidden_state
        part2 = torch.empty(batch_size, 0, hidden_size, device=hidden_state.device)
    else:
        part1 = hidden_state[:, :split_position, :]
        part2 = hidden_state[:, split_position:, :]

    return part1, part2


class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states):

        self.hidden_states = hidden_states

    def __len__(self):
        return self.hidden_states.shape[0]

    def __getitem__(self, idx):
        return self.hidden_states[idx]


class DistillationDataset(Dataset):
    def __init__(self, teacher_query_hidden, student_doc_hidden, student_query_hidden, temperature=1.0):

        assert teacher_query_hidden.shape == student_query_hidden.shape, \
            f"teacher's and student's hidden_state not match：{teacher_query_hidden.shape} vs {student_query_hidden.shape}"

        self.teacher_query_hidden = teacher_query_hidden
        self.student_query_hidden = student_query_hidden
        self.student_doc_hidden = student_doc_hidden
        self.temperature = temperature

    def __len__(self):
        return self.teacher_query_hidden.shape[0]

    def __getitem__(self, idx):
        return self.teacher_query_hidden[idx], self.student_doc_hidden[idx], self.student_query_hidden[idx]


class SplitHiddenStateDataset(Dataset):
    def __init__(self, hidden_states, split_pos=32):

        self.hidden_states = hidden_states
        self.split_pos = split_pos

        seq_len = hidden_states.shape[1]
        if seq_len < split_pos:
            raise ValueError(f"cant split")

    def __len__(self):
        return self.hidden_states.shape[0]  # 样本数量

    def __getitem__(self, idx):

        sample = self.hidden_states[idx]

        part1 = sample[:self.split_pos, :]
        part2 = sample[self.split_pos:, :]

        return part1, part2


def process_embeddings(student_doc_emb, student_query_emb, l_query):

    # [b, -1, e] → [b, e]
    last_doc_emb = student_doc_emb[:, -1, :]  #  [b, e]


    repeated_doc_emb = last_doc_emb.unsqueeze(1).repeat(1, l_query, 1)  #  [b, l_query, e]

    result = repeated_doc_emb * student_query_emb  #  [b, l_query, e]

    return result

