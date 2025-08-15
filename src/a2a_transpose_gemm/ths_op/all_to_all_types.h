//===- all_to_all_types.h ----------------------------------------- C++ ---===//
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
#include "flux/ths_op/topo_utils.h"
#include <iostream>

namespace bytedance::flux {
// All2All for nvlink mode. for NVLINK machine, default is 0
// Ring1D for 1d-ring. for PCI-e machine without GPUs cross NUMA nodes use ring 1d
// Ring2D for 2d-ring. for PCI-e machine with GPUs cross NUMA nodes defaults to ring_2d
enum class A2ARingMode {
  All2All = 0,
  Ring1D = 1,
  Ring2D = 2,
};

namespace detail {
template <typename T, bool O = false>
using optionally_optional = std::conditional_t<O, std::optional<T>, T>;

template <bool O = false>
struct A2AOptionType {
  optionally_optional<bool, O> input_buffer_copied;
  optionally_optional<bool, O> use_cuda_core;
  optionally_optional<bool, O> fuse_sync;
  optionally_optional<bool, O> use_read;
  optionally_optional<bool, O> skip_barrier;
  optionally_optional<bool, O> return_comm_buf;
  optionally_optional<A2ARingMode, O> mode;
};

}  // namespace detail

using AllToAllOption = detail::A2AOptionType<false>;
using AllToAllOptionWithOptional = detail::A2AOptionType<true>;

inline A2ARingMode
get_default_a2a_ring_mode() {
  return A2ARingMode::All2All;
}

}  // namespace bytedance::flux
