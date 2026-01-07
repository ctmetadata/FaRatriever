import argparse
import json
import logging
import os
import sys
import gc
from itertools import chain
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset
import pandas as pd

sys.path.append("/vepfs/group04/user/chenteng/lahore/lahore_distill_student_luban")

from src.data_utils import get_student_query_tamplate_text, get_tokens, load_and_preprocess_data, \
    load_and_preprocess_data_infer
from src.train_utils import QwenStudentModel, chunked_linear


class SmartCollator:
    def __init__(self, tokenizer, mlm=False):
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)

    def __call__(self, batch):
        query_id = [ex.pop('query_id') for ex in batch]
        doc_id = [ex.pop('doc_id') for ex in batch]
        label = [ex.pop('label') for ex in batch]
        padded = self.data_collator(batch)
        padded['query_id'] = query_id
        padded['doc_id'] = doc_id
        padded['label'] = label
        return padded


def save_results(results, output_path):
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Open the file in append mode and write the DataFrame
    with open(output_path, 'a', encoding='utf-8') as f:
        df.to_json(f, orient='records', lines=True, force_ascii=False)

    logging.info(f"Saved results to {output_path}")



def get_global_rank(local_rank, local_size, node_rank):
    return node_rank * local_size + local_rank


def setup(rank, world_size, node_rank, local_size, master_addr, master_port):
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Adjust as necessary

    global_rank = get_global_rank(rank, local_size, node_rank)
    logging.info(f"Setting up process group: local_rank={rank}, global_rank={global_rank}, world_size={world_size}")
    dist.init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}', rank=global_rank,
                            world_size=world_size)
    torch.cuda.set_device(rank)
    logging.info(f"Process group set up complete: global_rank={global_rank}")


def cleanup():
    dist.destroy_process_group()
    logging.info("Process group destroyed")


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Inference with Transformers")
    parser.add_argument('--base_model', type=str, required=True, help='Model ID to use for inference')
    parser.add_argument('--model_path', type=str, required=True, help='Tokenizer ID to use for inference')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input JSONL data file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--split_position', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the inference results')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of GPUs across all nodes')
    parser.add_argument('--node_rank', type=int, required=True, help='Rank of the current node')
    parser.add_argument('--local_size', type=int, required=True, help='Number of GPUs per node')
    parser.add_argument('--master_addr', type=str, required=True, help='Address of the master node')
    parser.add_argument('--master_port', type=int, required=True, help='Port of the master node')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens to generate')
    parser.add_argument('--do_sample', action='store_true', help='Enable sampling for generation')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p (nucleus) sampling')
    parser.add_argument('--ppl_mask', action='store_true')
    args = parser.parse_args()

    return args


def calculate_perplexity(model, the_pormpt, tokenizer, the_answer, device, args, max_length=4096, stride=512):
    model_inputs = tokenizer([the_pormpt], max_length=max_length, truncation=True, return_tensors='pt').to(device)

    seq_len = model_inputs['input_ids'].shape[-1]
    doc_tok = tokenizer([the_answer + '<|im_end|>\n<|im_start|>assistant\n'], return_tensors="pt")['input_ids'][0]
    if seq_len > max_length:
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = model_inputs['input_ids'][:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log = outputs.loss
            nlls.append(neg_log)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean()).tolist()
    else:
        input_ids = model_inputs['input_ids'].to(device)
        target_ids = input_ids.clone()
        if args.ppl_mask:
            trg_len = len(doc_tok)
            target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log = outputs.loss
        ppl = torch.exp(neg_log.mean()).tolist()

    return ppl


def get_logp(tokenizer, generated_ids, scores, t_the_logprob_tag, the_logprob_tag):
    logprob_tag_list = []
    all_scores = []
    the_score = scores[0]

    all_scores += [(tokenizer.decode([generated_ids[0]]), generated_ids[0], the_score.tolist()[0][generated_ids[0]])]
    tmp = []
    for j in range(len(t_the_logprob_tag)):
        tmp.append((the_logprob_tag[j], t_the_logprob_tag[j], the_score.tolist()[0][t_the_logprob_tag[j]]))

    logprob_tag_list.append(tmp)

    return logprob_tag_list, all_scores


def write_json_line(tar_data, tar_file, lock):
    with lock:
        with open(tar_file, 'a') as fw:
            for x in tar_data:
                json.dump(x, fw, ensure_ascii=False)
                fw.write('\n')


def inference(rank, args, lock):
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - %(levelname)s - %(message)s [Rank {rank}]')

    global_rank = get_global_rank(rank, args.local_size, args.node_rank)

    setup(rank, args.world_size, args.node_rank, args.local_size, args.master_addr, args.master_port)
    world_size = dist.get_world_size()
    torch.manual_seed(0)  # Ensure every process has same seed
    torch.cuda.set_device(rank)  # Ensure CUDA device is set for this process
    #
    # print(f"base model,{args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=f"cuda:{global_rank}"
    )
    student_model = QwenStudentModel(base_model=base_model)
    student_model.load(model_path=args.model_path)


    data_path = args.data_path[1:].split(',')
    tokenized_dataset, tokenizer = load_and_preprocess_data_infer(model_path=args.base_model, data_path=data_path,
                                                                  split_position=args.split_position)
    train_dataset = tokenized_dataset["train"]

    tokenizer.padding = True

    data_collator = SmartCollator(tokenizer, mlm=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )

    results = []
    student_model.base_model.eval()
    student_model.additional_layer.eval()
    with torch.no_grad():
        the_logprob_tag = ["A", "B"]
        t_logprob_tag = tokenizer(the_logprob_tag, return_tensors='pt')['input_ids'].tolist()
        t_the_logprob_tag = list(chain(*t_logprob_tag))

        generation_configs = {
            'max_new_tokens': args.max_new_tokens,
            'eos_token_id': tokenizer.eos_token_id,
            'output_scores': True,
            'return_dict_in_generate': True,
        }

        if args.do_sample:
            generation_configs.update({
                'do_sample': True,
                'top_k': args.top_k,
                'top_p': args.top_p,
                'temperature': 0.7,
                'repetition_penalty': 1.05
            })
        else:
            generation_configs.update({
                'do_sample': False,
            })
        try:
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                try:
                    logging.info(f"Processing batch {batch_idx}")
                    for example in batch:
                        student_query_embedding, teacher_query_embedding, teacher_query_logits, student_query_logits = student_model.forward(
                            teacher_input_ids=example["teacher_input_ids"],
                            teacher_attention_mask=example["teacher_attention_mask"],
                            student_doc_input_ids=example["student_doc_input_ids"],
                            student_doc_attention_mask=example["student_doc_attention_mask"],
                            student_query_input_ids=example["student_query_input_ids"],
                            student_query_attention_mask=example["student_query_attention_mask"],
                        )
                        # print(student_query_logits.shape,student_query_logits)

                        reversed_teacher_attention_mask = torch.flip(
                            torch.tensor(example["student_query_attention_mask"]).to(base_model.device), dims=[1])
                        first_one_in_reversed = torch.argmax(reversed_teacher_attention_mask.int(), dim=1)
                        last_one_index = (torch.tensor(example["student_query_attention_mask"]).size(1) - 1) - first_one_in_reversed

                        answer_token_index = last_one_index - 2

                        student_answer_logits = student_query_logits[:, answer_token_index.tolist(), :]

                        token_id_A = 32  # A token ID
                        token_id_B = 33  # B token ID
                        target_token_ids = torch.tensor([token_id_A, token_id_B], device=base_model.device)
                        # print("target_token_ids", target_token_ids)
                        student_logits_AB = student_answer_logits[:, :, target_token_ids]
                        print("student_logits_AB", student_logits_AB.shape, student_logits_AB.tolist())

                        logprob_tag_list = [[["A", 32, student_logits_AB.tolist()[0][0][0]],
                                             ["B", 33, student_logits_AB.tolist()[0][0][1]]]]
                        if student_logits_AB.tolist()[0][0][0] > student_logits_AB.tolist()[0][0][1]:
                            log_prob = [["A", 32, student_logits_AB.tolist()[0][0][0]]]
                        else:
                            log_prob = [["B", 33, student_logits_AB.tolist()[0][0][1]]]



                        generated_text = tokenizer.decode(
                            example["student_query_input_ids"][answer_token_index.tolist()[0]], skip_special_tokens=False)
                        print("generated_text",generated_text)

                        results.append({
                            "query_id": example['query_id'],
                            "doc_id": example['doc_id'],
                            "label": example['label'],
                            "response": generated_text,
                            "log_prob": log_prob,
                            "logprob_tag_list": logprob_tag_list,
                        })
                    # print(results)

                    # Clear CUDA cache after processing a batch
                    torch.cuda.empty_cache()
                    gc.collect()

                    write_json_line(results, args.output_path, lock)
                    logging.info(f"Batch {batch_idx} data write {args.output_path} complete")
                    results = []

                except RuntimeError as e:
                    logging.error(f"Runtime error during batch {batch_idx}: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()

            cleanup()
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Unhandled exception: {e}")
            cleanup()
            raise
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting distributed inference")
    lock = mp.RLock()
    logging.info(f"lock: {lock}")

    mp.spawn(inference, args=(args, lock), nprocs=args.local_size, join=True)
    logging.info("Distributed inference complete")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
