//===- moe_utils.h ------------------------------------------------ C++ ---===//
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

#pragma once
#include <c10/core/ScalarType.h>
#include <torch/all.h>

namespace bytedance::flux::ths_op {

class TransportOp {
 public:
  TransportOp(int64_t rank, int64_t world_size, torch::Tensor recv_buffer);

  void copy_by_sm(
      torch::Tensor send_buffer, torch::Tensor transport_offsets, torch::Tensor transport_nbytes);

  void copy_by_ce(
      torch::Tensor send_buffer, torch::Tensor transport_offsets, torch::Tensor transport_nbytes);

 private:
  class TransportOpImpl;
  TransportOpImpl *impl_;
};

class All2AllOp {
 public:
  All2AllOp(int64_t rank, int64_t world_size, torch::Tensor recv_buffer);
  ~All2AllOp();

  void forward(c10::List<torch::Tensor> send_buffer);

 private:
  class All2AllOpImpl;
  All2AllOpImpl *impl_;
};
}  // namespace bytedance::flux::ths_op
