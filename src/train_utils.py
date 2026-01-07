import logging
import os
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling
from torch.nn.parallel import DistributedDataParallel as DDP

from src.embedding_utils import process_embeddings
import torch.distributed as dist

from src.data_utils import load_and_preprocess_data
from src.embedding_utils import split_hidden_state
from src.logging_utils import _setup_logger


class SingleLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        return self.layer(x)  # [batch_size, seq_len, hidden_dim]


class QwenStudentModel(nn.Module):
    def __init__(self, base_model, hidden_dim=3584, vocab_size=151936, kl_weight=0.5,
                 temperature=2.0):
        super().__init__()
        self.base_model = base_model

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.additional_layer = SingleLayer(hidden_dim)

        self.additional_layer = self.additional_layer.to(self.base_model.device)

        # load
    def load(self, model_path):
        # load additional layer
        ckpt = torch.load(model_path, map_location=self.base_model.device)
        self.additional_layer.load_state_dict(ckpt)
        print(f"addtitional layer,{model_path},succeed!!")

    def forward(self, teacher_input_ids, teacher_attention_mask, student_doc_input_ids, student_doc_attention_mask,
                student_query_input_ids,
                student_query_attention_mask, labels=None):
        with autocast():
            teacher_outputs = self.base_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                output_hidden_states=True,
            )
            student_doc_outputs = self.base_model(
                input_ids=student_doc_input_ids,
                attention_mask=student_doc_attention_mask,
                output_hidden_states=True,
            )
            student_query_outputs = self.base_model(
                input_ids=student_query_input_ids,
                attention_mask=student_query_attention_mask,
                output_hidden_states=True,
            )

            #  [batch_size, seq_len, hidden_dim]
            student_doc_embeddings = student_doc_outputs.hidden_states[-1]
            student_query_embeddings = student_query_outputs.hidden_states[-1]
            teacher_embeddings = teacher_outputs.hidden_states[-1]

            teacher_doc_embedding, teacher_query_embedding = split_hidden_state(teacher_embeddings, 32)
            student_query_embeddings_transfer = self.additional_layer(
                student_query_embeddings)  # [batch_size, seq_len, vocab_size]
            student_query_output = process_embeddings(student_doc_embeddings, student_query_embeddings_transfer,
                                                      4096 - 32)

            teacher_query_logits = chunked_linear(
                self.base_model.lm_head,
                teacher_query_embedding,
                chunk_size=256
            )
            student_query_logits = chunked_linear(
                self.base_model.lm_head,
                student_query_output,
                chunk_size=256
            )

            return student_query_output, teacher_query_embedding, teacher_query_logits, student_query_logits


def mse_on_last_hidden(hidden_states, target):

    if isinstance(hidden_states, (tuple, list)):
        last_hidden = hidden_states[-1]  # (B, L, D)
    else:
        #  tensor: (num_layers, B, L, D)
        last_hidden = hidden_states[-1] if hidden_states.dim() == 4 else hidden_states

    assert last_hidden.shape == target.shape, \
        f"Shape mismatch: {last_hidden.shape} vs {target.shape}"
    #  MSE
    loss = nn.functional.mse_loss(last_hidden, target, reduction='mean')  # 标量
    return loss


def mse_logits(logits1, logits2, reduction='mean'):
    """
    logits1, logits2: (B, L, V)
    return:  MSE
    """
    if logits1.shape != logits2.shape:
        raise ValueError(f"Shape mismatch: {logits1.shape} vs {logits2.shape}")
    if logits1.dtype != logits2.dtype:
        logits2 = logits2.to(logits1.dtype)

    mse = F.mse_loss(logits1, logits2, reduction=reduction)
    return mse


def chunked_linear(linear: torch.nn.Linear, x: torch.Tensor, chunk_size: 512):

    B, L, D = x.shape
    V = linear.out_features
    out = torch.empty(B, L, V, dtype=x.dtype, device=x.device)
    for i in range(0, L, chunk_size):
        j = min(i + chunk_size, L)
        out[:, i:j] = linear(x[:, i:j])
    return out


def get_global_rank(local_rank, local_size, node_rank):
    return node_rank * local_size + local_rank


def setup(local_rank, master_addr, master_port):
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # Adjust as necessary

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    logging.info(
        f"Setting up process group: local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}")

    dist.init_process_group(backend='nccl', init_method=f'tcp://{master_addr}:{master_port}', rank=global_rank,
                            world_size=world_size)
    torch.cuda.set_device(local_rank)
    logging.info(f"Process group set up complete: global_rank={global_rank}")


def train(args):

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # local_rank = args.local_rank
    # local_rank = "0,1,2,3,4,5,6,7"
    local_rank = 0
    # logger.info("local_rank",args.local_rank)
    # setup(local_rank,  args.master_addr, args.master_port)
    dist.init_process_group(backend='nccl')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    logger = _setup_logger(global_rank)
    logger.info(f"global rank,{global_rank},world size,{world_size}")
    logger.info(f"local_rank,{local_rank}")

    torch.manual_seed(0)  # Ensure every process has same seed
    torch.cuda.set_device(local_rank)  # Ensure CUDA device is set for this process

    device = torch.device(f"cuda:{local_rank}")

    loss_q = {
        'embedding_mse': deque(maxlen=args.log_interval),
        'token_logits_mse_loss': deque(maxlen=args.log_interval),
        'total': deque(maxlen=args.log_interval),
    }
    writer = SummaryWriter(log_dir=os.path.join(args.output_path, 'tb_logs'))

    tokenized_dataset, tokenizer = load_and_preprocess_data(data_path=args.data_path,
                                                            model_path=args.base_model_name_or_path,
                                                            split_position=args.split_position)
    train_dataset = tokenized_dataset["train"]

    tokenizer.padding = True
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )

    if global_rank == 0:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map=f"cuda:{global_rank}"
        )
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map=f"cuda:{global_rank}",
        )
    # logger.info("teacher_model",teacher_model.device)
    # teacher_model = teacher_model.to(rank)
    logger.info(f"teacher_model,{teacher_model.device}")
    student_model = QwenStudentModel(base_model=teacher_model)
    logger.info(f"student_model base,{student_model.base_model.device}")
    logger.info(f"student_model additional_layer,{next(student_model.additional_layer.parameters()).device}")

    student_ddp = DDP(student_model, device_ids=[local_rank])

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    optimizer = optim.AdamW(
        student_ddp.module.additional_layer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )


    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader)
    )

    if global_rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        logger.info(f"model save: {args.output_path}")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        student_ddp.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{args.epochs}",
                    ncols=100,
                    disable=(global_rank != 0))

        for step, batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # logger.info("batch",len(batch["teacher_input_ids"]))

            student_query_embedding, teacher_query_embedding, teacher_query_logits, student_query_logits = student_ddp(
                teacher_input_ids=batch["teacher_input_ids"],
                teacher_attention_mask=batch["teacher_attention_mask"],
                student_doc_input_ids=batch["student_doc_input_ids"],
                student_doc_attention_mask=batch["student_doc_attention_mask"],
                student_query_input_ids=batch["student_query_input_ids"],
                student_query_attention_mask=batch["student_query_attention_mask"],
            )
            # logger.info(f"teacher_query_logits,{teacher_query_logits.shape},student_query_logits,{student_query_logits.shape}")

            # embeddingn mse
            embedding_mse = mse_on_last_hidden(student_query_embedding, teacher_query_embedding)
            # logger.info(f"embedding_mse,{embedding_mse.tolist()}")


            # logits_mse = mse_logits(student_query_logits, teacher_query_logits)
            # logger.info(f"logits_mse,{logits_mse.tolist()}")

            # answer mse
            reversed_teacher_attention_mask = torch.flip(batch["teacher_attention_mask"], dims=[1])
            first_one_in_reversed = torch.argmax(reversed_teacher_attention_mask.int(), dim=1)
            last_one_index = (batch["teacher_attention_mask"].size(1) - 1) - first_one_in_reversed
            answer_token_index = last_one_index - 2



            teacher_logits_mask = torch.zeros_like(teacher_query_logits)
            student_logits_mask = torch.zeros_like(student_query_logits)
            batch_idx = torch.arange(len(batch["teacher_input_ids"]), device=teacher_query_logits.device)

            teacher_logits_mask[batch_idx, answer_token_index, :] = 1.0
            student_logits_mask[batch_idx, answer_token_index-32, :] = 1.0
            teacher_answer_logits = teacher_query_logits * teacher_logits_mask
            student_answer_logits = student_query_logits * teacher_logits_mask
            teacher_answer_token_logits = teacher_answer_logits[batch_idx, answer_token_index].unsqueeze(1)
            student_answer_token_logits = student_answer_logits[batch_idx, answer_token_index].unsqueeze(1)

            token_id_A = 32  # A token ID
            token_id_B = 33  # B token ID
            target_token_ids = torch.tensor([token_id_A, token_id_B], device=teacher_answer_token_logits.device)
            teacher_logits_AB = teacher_answer_token_logits[:, :, target_token_ids]
            student_logits_AB = student_answer_token_logits[:, :, target_token_ids]
            # logger.info("teacher_logits_AB",teacher_logits_AB)
            # logger.info("student_logits_AB",student_logits_AB)
            normalized_teacher_logits_AB = F.softmax(teacher_logits_AB, dim=-1)
            normalized_student_logits_AB = F.softmax(student_logits_AB, dim=-1)
            # logger.info("normalized_teacher_logits_AB",normalized_teacher_logits_AB)
            # logger.info("normalized_student_logits_AB",normalized_student_logits_AB)

            token_logits_mse_loss = F.mse_loss(normalized_teacher_logits_AB, normalized_student_logits_AB)
            # logger.info(f"token_logits_mse_loss,{token_logits_mse_loss.tolist()}")

            loss = args.embedding_loss_weight * embedding_mse  + args.answer_token_loss_weight * token_logits_mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # total_loss += loss.item()

            if global_rank == 0:
                for k, v in zip(['embedding_mse',  'token_logits_mse_loss', 'total'],
                                [embedding_mse,  token_logits_mse_loss, loss]):
                    loss_q[k].append(v.item())

            if global_rank == 0 and (step + 1) % args.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(
                    f"Epoch [{epoch + 1}/{args.epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {avg_loss:.6f}")
                avg = {k: sum(q) / len(q) for k, q in loss_q.items()}
                curr_lr = scheduler.get_last_lr()[0]
                gs = epoch * len(train_loader) + step + 1  # global step
                for k, v in avg.items():
                    writer.add_scalar(f'Loss/{k}', v, gs)
                    writer.flush()
                writer.add_scalar('LR', curr_lr, gs)
                writer.flush()

        if global_rank == 0 and (epoch + 1) % args.save_interval == 0:
            torch.save(
                student_model.additional_layer.state_dict(),
                os.path.join(args.output_path, f"additional_layer_{args.learning_rate}_epoch_{epoch + 1}.pth")
            )
            logger.info(f"new layer save: {args.output_path}")
        dist.barrier()

    dist.destroy_process_group()
