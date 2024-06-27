//===- helper_ops.cc ---------------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "c10/cuda/CUDAStream.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/helper_kernels.h"
#include <torch/python.h>

namespace bytedance::flux::ths_op {
using torch::Tensor;

bool
bitwise_check(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.dim() == B.dim(), "Tensor dimension not matching! A:", A.dim(), " vs B:", B.dim());
  return bitwise_check(from_torch_dtype(A.scalar_type()), A.data_ptr(), B.data_ptr(), A.numel());
}

void
uniform_initialize(torch::Tensor tensor, uint64_t seed, double min, double max) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  uniform_initialize(
      from_torch_dtype(tensor.scalar_type()),
      tensor.data_ptr(),
      tensor.numel(),
      seed,
      min,
      max,
      stream);
}

void
cudaipc_barrier_all_on_stream(
    cudaStream_t stream, int rank, std::vector<torch::Tensor> &sync_buffers) {
  std::vector<int32_t *> sync_buffer_ptrs;
  int world_size = sync_buffers.size();
  for (int i = 0; i < sync_buffers.size(); i++) {
    sync_buffer_ptrs.push_back(reinterpret_cast<int32_t *>(sync_buffers[i].data_ptr()));
  }
  cudaipc_barrier_all_on_stream_impl(stream, sync_buffer_ptrs.data(), rank, world_size);
}

}  // namespace bytedance::flux::ths_op
