//===- reduce_scatter_op.h ---------------------------------------- C++ ---===//
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
#include <optional>

namespace bytedance::flux {
// All2All for nvlink mode. for NVLINK machine, default is 0
// Ring1D for 1d-ring. for PCI-e machine without GPUs cross NUMA nodes use ring 1d
// Ring2D for 2d-ring. for PCI-e machine with GPUs cross NUMA nodes defaults to ring_2d
enum class RingMode {
  All2All = 0,
  Ring1D = 1,
  Ring2D = 2,
};
namespace detail {
template <typename T, bool O = false>
using optionally_optional = std::conditional_t<O, std::optional<T>, T>;

template <bool O = false>
struct ReduceScatterOptionType {
  optionally_optional<bool, O> use_barrier_queue;
  optionally_optional<bool, O> use_1d_ring;
  optionally_optional<bool, O> use_p2p_read;
  optionally_optional<bool, O> use_cudaMemcpyAsync;
  optionally_optional<bool, O> use_gemmk;
  optionally_optional<bool, O> per_tile_flags;
  optionally_optional<int, O> n_split;
  optionally_optional<int, O> num_blocks;
  optionally_optional<RingMode, O> ring_mode;
};

}  // namespace detail

using ReduceScatterOption = detail::ReduceScatterOptionType<false>;
using ReduceScatterOptionWithOptional = detail::ReduceScatterOptionType<true>;

inline RingMode
get_default_rs_ring_mode() {
  if (topo_utils::has_nvswitch()) {
    return RingMode::All2All;
  }
  static const int kNumaWorldSize = 4;

  if (topo_utils::has_heterogeneous_pcie()) {
    if (topo_utils::topo_numa_local_world_size() != kNumaWorldSize) {
      return RingMode::Ring1D;  // PCI-e ring mode with no optimization
    }
    return RingMode::Ring2D;
  }
  return RingMode::Ring1D;
}

}  // namespace bytedance::flux
