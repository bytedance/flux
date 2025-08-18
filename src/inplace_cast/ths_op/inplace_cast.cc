//===- inplace_cast.cc -------------------------------------------- C++ ---===//
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

#include "inplace_cast/ths_op/inplace_cast.h"

#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/logging_is_not_google_glog.h>

#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/utils.h"
#include "inplace_cast/inplace_cast.hpp"

namespace bytedance::flux::ths_op {
using torch::Tensor;

class InplaceCast::InplaceCastImpl {
 private:
  int32_t block_size;
  int32_t data_size;

  torch::Tensor flags;
  torch::Tensor chunk_counter;

 public:
  InplaceCastImpl(int data_size) : data_size(data_size) {
    block_size = INPLACE_CAST_BLOCK_SIZE;

    int num_chunks =
        (data_size + block_size * INPLACE_CAST_TS - 1) / (block_size * INPLACE_CAST_TS);

    flags = torch::empty(
        {num_chunks},
        at::TensorOptions()
            .dtype(c10::ScalarType::Int)
            .device(at::kCUDA)
            .device_index(c10::cuda::current_device()));

    chunk_counter = torch::empty(
        {1},
        at::TensorOptions()
            .dtype(c10::ScalarType::Int)
            .device(at::kCUDA)
            .device_index(c10::cuda::current_device()));
  }

  void
  reset() {
    this->flags.zero_();
    this->chunk_counter.zero_();
  }

  void
  from_fp32_to_bf16(torch::Tensor input) {
    int input_size = input.numel();
    FLUX_CHECK(input_size <= data_size) << "flags space not enough.";

    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();

    this->reset();

    bytedance::flux::inplace_cast_fp32_to_bf16_impl(
        input.data_ptr(),
        input_size,
        (unsigned *)flags.data_ptr(),
        (unsigned *)chunk_counter.data_ptr(),
        current_stream,
        INPLACE_CAST_NUM_BLOCKS,
        block_size);
  }
};

InplaceCast::InplaceCast(int32_t data_size) : impl(new InplaceCastImpl(data_size)) {}
InplaceCast::~InplaceCast() { delete impl; }
void
InplaceCast::from_fp32_to_bf16(torch::Tensor input) {
  impl->from_fp32_to_bf16(input);
}

}  // namespace bytedance::flux::ths_op
