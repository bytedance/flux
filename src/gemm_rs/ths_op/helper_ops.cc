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
#include <torch/all.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cuda_bf16.h>

#include "flux/ths_op/ths_op.h"
#include "gemm_rs/ths_op/helper_ops.h"
#include "gemm_rs/bsr_reduce.hpp"
#include "gemm_rs/ring_reduce.hpp"
#include "gemm_rs/padding_util.hpp"

namespace bytedance::flux::ths_op {
using torch::Tensor;

void
bsr_reduce(torch::Tensor input, torch::Tensor output, int block_h, int block_w) {
  TORCH_CHECK(input.dim() == 3, "input shape is not 3 (B, M, N)");
  bsr2dense_reduce(
      input.data_ptr(),
      output.data_ptr(),
      std::vector<int>(
          {static_cast<int>(input.size(0)),
           static_cast<int>(input.size(1)),
           static_cast<int>(input.size(2))}),
      block_h,
      block_w,
      from_torch_dtype(input.scalar_type()),
      c10::cuda::getCurrentCUDAStream());
}

std::pair<torch::Tensor, c10::optional<torch::Tensor>>
pad_m_to_TPxTile(
    torch::Tensor input, c10::optional<torch::Tensor> input_scale, int tp_size, int tile_size) {
  TORCH_CHECK(input.dim() == 2, "input shape is not 2 (M, N)");
  int m_dim = input.size(0);
  int n_dim = input.size(1);
  int m_padded = pad_to(m_dim, tp_size * tile_size);
  if (m_dim == m_padded) {
    return {input, input_scale};
  }
  torch::Tensor output = torch::empty({m_padded, n_dim}, input.options());
  c10::optional<torch::Tensor> padded_scale = input_scale;
  if (input_scale.has_value()) {
    FLUX_CHECK(input_scale->dim() == 2);
    FLUX_CHECK(input_scale->size(1) == 1);
    padded_scale = torch::empty({m_padded, input_scale->size(1)}, input_scale->options());
  }
  ::bytedance::flux::pad_m_to_TPxTile(
      input.data_ptr(),
      input_scale.has_value() ? input_scale->data_ptr() : nullptr,
      output.data_ptr(),
      padded_scale.has_value() ? padded_scale->data_ptr() : nullptr,
      m_dim,
      n_dim,
      tp_size,
      tile_size,
      from_torch_dtype(input.scalar_type()),
      from_torch_dtype(
          input_scale.has_value() ? input_scale->scalar_type() : c10::ScalarType::Float),
      c10::cuda::getCurrentCUDAStream());
  return {output, padded_scale};
}

void
ring_reduce(torch::Tensor input, torch::Tensor output, int dim, int rank) {
  int node_num = 1, chunk_size = 1;
  int world_size = input.size(dim);
  for (int i = 0; i < dim; ++i)
    node_num *= input.size(i);
  for (int i = dim + 1; i < input.dim(); ++i)
    chunk_size *= input.size(i);
  ::bytedance::flux::ring_reduce(
      input.data_ptr(),
      output.data_ptr(),
      rank,
      node_num,
      world_size,
      chunk_size,
      from_torch_dtype(input.scalar_type()),
      c10::cuda::getCurrentCUDAStream());
}
}  // namespace bytedance::flux::ths_op
