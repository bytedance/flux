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

#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/all.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <mutex>
#include "cuda.h"
#include "cuda_fp16.h"
#include <cuda_bf16.h>
#include <torch/python.h>

#include "flux/ths_op/ths_op.h"
#include "reduce_scatter/ths_op/helper_ops.h"
#include "reduce_scatter/bsr_reduce.hpp"

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

static int _register_reduce_scatter_helper_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one(
      "reduce_scatter_helper_ops", [](py::module &m) { m.def("bsr_reduce", &bsr_reduce); });
  return 0;
}();

}  // namespace bytedance::flux::ths_op
