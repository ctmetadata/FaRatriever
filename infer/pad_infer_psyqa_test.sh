#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -n "$MLP_WORKER_NUM" ]; then
  NNODES="$MLP_WORKER_NUM"
  GPUS_PER_NODE=8
else
  NNODES=1
  GPUS_PER_NODE=1
fi

if [ -n "$MLP_ROLE_INDEX" ]; then
  NODE_RANK="$MLP_ROLE_INDEX"
else
  NODE_RANK=0
fi

if [ -n "$MLP_WORKER_0_HOST" ]; then
  MASTER_ADDR="$MLP_WORKER_0_HOST"
  MASTER_PORT="$MLP_WORKER_0_PORT"
else
  MASTER_ADDR=localhost
  MASTER_PORT=12345
fi

# NNODES=${MLP_WORKER_NUM:-1}
# GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=1
# NODE_RANK=${MLP_ROLE_INDEX:-0}

# MASTER_ADDR=${MLP_WORKER_0_HOST:-"localhost"}
# MASTER_PORT=${MLP_WORKER_0_PORT:-12345}

DISTRIBUTED_ARGS="--nnodes $NNODES --nproc-per-node $NPROC_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# declare -a dict_epch=("1160" "1741" "2321" "2900")

echo "GPUS_PER_NODE: $GPUS_PER_NODE"

start_idx=100
end_idx=149

res_file_idx=${start_idx}_${end_idx}
root_path=/vepfs/group04/user/chenteng/my_data_dialog/retreval_data/new_format/sft_data/psyqa/eval/student

inputfiles=""

echo $inputfiles

# epoch 4 497 ; 3 373 ; 2 248 ; 1 124
model_epoch=${epoch:-3}
base_model=${base_model:-""}
model_path=${model_path:-""}
lr=${lr:-"1e6"}

outputfile=""

echo $model_epoch
echo $model_path
echo $base_model
echo $outputfile

python -m torch.distributed.run $DISTRIBUTED_ARGS /vepfs/group04/user/chenteng/lahore/lahore_distill_student_luban/infer/ddp_infer_message_type_v2.py \
                --base_model $base_model \
                --model_path $model_path \
                --data_path "$inputfiles" \
                --batch_size 10 \
                --output_path "$outputfile" \
                --max_new_tokens 2 \
                --world_size $((NNODES * GPUS_PER_NODE)) \
                --node_rank $NODE_RANK \
                --local_size $GPUS_PER_NODE \
                --master_addr $MASTER_ADDR \
                --master_port $MASTER_PORT


