#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_SRC_DIR=${SCRIPT_DIR}

# add flux python package to PYTHONPATH
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_MODULE_LOADING=LAZY # EAGER if launch the consumer kernel before the producer kernel on host

# set default communication env vars
export BYTED_TORCH_BYTECCL=O0
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}

nproc_per_node=${ARNOLD_WORKER_GPU:=$(nvidia-smi --list-gpus | wc -l)}
nnodes=${ARNOLD_WORKER_NUM:=1}
if [ $ARNOLD_WORKER_NUM == 1 ]; then # single machine. use no NVSHMEM_REMOTE_TRANSPORT
  export NVSHMEM_REMOTE_TRANSPORT=none
fi
node_rank=${ARNOLD_ID:=0}
master_addr=${ARNOLD_WORKER_0_HOST:="127.0.0.1"}
if [ -z ${ARNOLD_WORKER_0_PORT} ]; then
  master_port="23456"
else
  master_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
fi

additional_args="--rdzv_endpoint=${master_addr}:${master_port}"

if [[ "$ARNOLD_DEVICE_TYPE" == *A100* ]] || [[ "$ARNOLD_DEVICE_TYPE" == *H800* ]]; then
  IB_HCA=mlx5
else
  IB_HCA=$ARNOLD_RDMA_DEVICE:1
fi

if [[ -n $NCCL_SOCKET_IFNAME ]]; then # check if NCCL_SOCKET_IFNAME exists
  ip link >/dev/null || sudo apt install iproute2
  ip link | grep $NCCL_SOCKET_IFNAME
  if [ $? -eq 0 ]; then
    echo "NCCL_SOCKET_IFNAME ${NCCL_SOCKET_IFNAME} exists"
  else
    echo "NCCL_SOCKET_IFNAME ${NCCL_SOCKET_IFNAME} not exist. unset NCCL_SOCKET_IFNAME"
    unset NCCL_SOCKET_IFNAME # let NCCL make the decisions
  fi
fi

if [ "$ARNOLD_RDMA_DEVICE" != "" ]; then
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:=0}
  export NCCL_IB_HCA=${NCCL_IB_HCA:=$IB_HCA}
  export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}

  ## env var setting for ARNOLD machines
  if [[ "$ARNOLD_DEVICE_TYPE" == *A100* ]] || [[ "$ARNOLD_DEVICE_TYPE" == *L20* ]]; then
    export NVSHMEM_IB_ENABLE_IBGDA=1
    export NVSHMEM_IBGDA_SUPPORT=1
    export NVSHMEM_IB_GID_INDEX=3
  fi
  if [[ "$ARNOLD_DEVICE_TYPE" == *PCI* ]]; then
    # for merlin PCIE machine
    export NVSHMEM_HCA_LIST=mlx5_0
  fi
else
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:=1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:=eth0}
fi

CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${additional_args} \
  ${FLUX_EXTRA_TORCHRUN_ARGS} $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret
