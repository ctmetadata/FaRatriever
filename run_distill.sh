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

#echo "WORLD_SIZE=$WORLD_SIZE  RANK=$RANK  MASTER=$MASTER_ADDR:$MASTER_PORT"


DISTRIBUTED_ARGS="--nnodes $NNODES --nproc-per-node $NPROC_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --base_model_name_or_path "" \
    --data_path "" \
    --batch_size 2 \
    --learning_rate 3e-4 \
    --epochs 3 \
    --temperature 4.0 \
    --embedding_loss_weight 0.3 \
    --logits_loss_weight 0.3 \
    --answer_token_loss_weight 0.4 \
    --log_interval 50 \
    --save_interval 1

