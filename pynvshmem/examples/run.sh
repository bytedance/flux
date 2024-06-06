#!/bin/bash
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
export PYNVSHMEM_ROOT=$(realpath ${BASEDIR}/..)
export NVSHMEM_ROOT=${NVSHMEM_ROOT:-$(realpath ${PYNVSHMEM_ROOT}/..)}

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1

export NVSHMEM_DEBUG=WARN
export NVSHMEM_BOOTSTRAP_MPI_PLUGIN=nvshmem_bootstrap_torch.so
export NVSHMEM_REMOTE_TRANSPORT=none

export PYTHONPATH=${PYNVSHMEM_ROOT}/build:${PYTHONPATH}
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH:${PYNVSHMEM_ROOT}/build:${NVSHMEM_ROOT}/build/src/lib:

cd ${PYNVSHMEM_ROOT}

torchrun \
  --node_rank=0 \
  --nproc_per_node=4 \
  --nnodes=1 \
  --rdzv_id=none \
  --master_addr=127.0.0.1 \
  --master_port=23456 \
  -t 0:3 \
  --redirects 1:3,2:3,3:3,4:3,5:3,6:3,7:3 \
  examples/run_example.py
