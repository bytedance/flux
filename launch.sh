#!/bin/bash
# libflux_cuda.so maybe installed under /usr/local/lib or ~/.local/lib/ by pip3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:~/.local/lib/
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

nproc_per_node=$(nvidia-smi --list-gpus | wc -l)
nnodes=1
node_rank=0
master_addr="127.0.0.1"
master_port="23456"
additional_args="--rdzv_endpoint=${master_addr}:${master_port}"
IB_HCA=mlx5


export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:=3}
export NVSHMEM_IB_GID_INDEX=3


CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${FLUX_EXTRA_TORCHRUN_ARGS} ${additional_args} $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret
