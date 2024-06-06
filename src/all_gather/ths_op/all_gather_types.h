//===- all_gather_types.cc ---------------------------------------- C++ ---===//
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

#pragma once
#include "flux/ths_op/topo_utils.h"
#include <iostream>

namespace bytedance::flux {
// All2All for nvlink mode. for NVLINK machine, default is 0
enum class AGRingMode {
  All2All = 0,
  Auto = -1,
};
static const int intra_numa_world_size = 4;

static AGRingMode
get_ring_mode(AGRingMode ring_mode) {
  if (ring_mode == AGRingMode::Auto) {  // auto detect. with nvlink use ring mode.
    if (!topo_utils::is_topo_properly_placed() || topo_utils::has_nvswitch()) {
      return AGRingMode::All2All;
    }
    return AGRingMode::All2All;
  }
  return ring_mode;
}

}  // namespace bytedance::flux
