//===- isendrecv.h -------------------------------------------- C++ ---===//
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
#include <vector>
#include <c10/core/ScalarType.h>
#include <torch/all.h>
namespace bytedance::flux::ths_op {
class AsyncSendRecv {
 public:
  AsyncSendRecv(
      int64_t max_m,
      int64_t n_dim,
      int64_t rank,        // rank in pp
      int64_t world_size,  // world_size of pp
      at::ScalarType input_dtype,
      int64_t duplicate);

  ~AsyncSendRecv();

  torch::Tensor get_comm_buffer(int64_t comm_buff_id);
  torch::Tensor read_comm_buffer(
      int64_t tgt_rank, int64_t src_comm_buff_id, int64_t tgt_comm_buff_id);
  void write_comm_buffer(int64_t tgt_rank, int64_t src_comm_buff_id, int64_t tgt_comm_buff_id);
  void set_signal(int64_t tgt_rank, int64_t comm_buffer_id, int64_t value);
  void wait_signal_eq(int64_t comm_buffer_id, int64_t value);
  void reset_signal(int64_t comm_buffer_id);

 private:
  class AsyncSendRecvOpImpl;
  AsyncSendRecvOpImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
