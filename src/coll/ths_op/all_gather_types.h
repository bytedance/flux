//===- all_gather_types.h ---------------------------------------- C++ ---===//
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
enum class AGRingMode {
  All2All = 0,
  Ring1D = 1,
  Ring2D = 2,
};
namespace detail {
template <typename T, bool O = false>
using optionally_optional = std::conditional_t<O, std::optional<T>, T>;

template <bool O = false>
struct AllGatherOptionType {
  optionally_optional<bool, O> input_buffer_copied;
  optionally_optional<bool, O> use_cuda_core_local;
  optionally_optional<bool, O> use_cuda_core_ag;
  optionally_optional<bool, O> fuse_sync;  // only valid when use_cuda_core_local=True
  optionally_optional<bool, O> use_read;
  optionally_optional<AGRingMode, O> mode;
};

}  // namespace detail

using AllGatherOption = detail::AllGatherOptionType<false>;
using AllGatherOptionWithOptional = detail::AllGatherOptionType<true>;

static const int kNumaWorldSize = 4;

inline AGRingMode
get_default_ag_ring_mode() {
  if (topo_utils::has_nvswitch()) {
    return AGRingMode::All2All;
  }

  if (topo_utils::has_heterogeneous_pcie()) {
    if (topo_utils::topo_numa_local_world_size() != kNumaWorldSize) {
      return AGRingMode::Ring1D;  // PCI-e ring mode with no optimization
    }
    return AGRingMode::Ring2D;
  }
  return AGRingMode::Ring1D;
}

}  // namespace bytedance::flux
