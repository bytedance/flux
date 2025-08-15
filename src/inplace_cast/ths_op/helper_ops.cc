//===- helper_ops.cc ---------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include <c10/cuda/CUDAStream.h>
#include "flux/cuda/cuda_common.h"
#include "inplace_cast/ths_op/helper_ops.h"
#include "inplace_cast/inplace_cast.hpp"

namespace bytedance::flux::ths_op {

void
inplace_cast_fp32_to_bf16(torch::Tensor data) {
  int block_size = INPLACE_CAST_BLOCK_SIZE;
  size_t data_size = data.numel();

  size_t num_chunks =
      (data_size + block_size * INPLACE_CAST_TS - 1) / (block_size * INPLACE_CAST_TS);

  unsigned *flags;
  CUDA_CHECK(cudaMalloc(&flags, num_chunks * sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(flags, 0, num_chunks * sizeof(unsigned)));

  unsigned *chunk_counter;
  CUDA_CHECK(cudaMalloc(&chunk_counter, sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(chunk_counter, 0, sizeof(unsigned)));

  inplace_cast_fp32_to_bf16_impl(
      data.data_ptr(),
      data_size,
      flags,
      chunk_counter,
      c10::cuda::getCurrentCUDAStream(),
      INPLACE_CAST_NUM_BLOCKS,
      block_size);

  cudaFree(flags);
  cudaFree(chunk_counter);
}

}  // namespace bytedance::flux::ths_op
