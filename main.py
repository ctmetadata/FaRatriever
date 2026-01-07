#!/usr/bin/env python3

import os
import torch
import torch.multiprocessing as mp

import argparse

from src.train_utils import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen-7B")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--split_position", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=2, help="每张卡的 batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--embedding_loss_weight", type=float, default=0.5, help="embedding_loss_weight")
    parser.add_argument("--logits_loss_weight", type=float, default=0.5, help="logits_loss_weight")
    parser.add_argument("--answer_token_loss_weight", type=float, default=0.5, help="answer_token_loss_weight")

    parser.add_argument("--local_rank", type=int, default=-1, help="由torchrun自动传递的本地进程编号")
    parser.add_argument("--master_addr", type=str, default=os.getenv("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master_port", type=str, default=os.getenv("MASTER_PORT", "29500"))
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    train(args)


if __name__ == "__main__":
    main()