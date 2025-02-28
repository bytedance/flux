#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${SCRIPT_DIR}/../../..

workspace=${PROJECT_ROOT}/workspace/vec-add

# compile kernels
python3 ${PROJECT_ROOT}/python/flux_triton/tools/compile_aot.py \
  --workspace ${workspace} \
  --kernels ${PROJECT_ROOT}/python/flux_triton/tools/compile_aot.py:add_kernel \
  --library flux_triton_kernel \
  --build

pushd ${workspace}
# compile test bin
g++ ${PROJECT_ROOT}/test/tools/aot/add_kernel_test.cc \
  -I${workspace} \
  -I/usr/local/cuda/include \
  -L${workspace}/build \
  -L/usr/local/cuda/lib64 \
  -lflux_triton_kernel \
  -lcudart -lcuda \
  -o add_kernel_test

# run test
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(realpath ./build) ./add_kernel_test
popd
