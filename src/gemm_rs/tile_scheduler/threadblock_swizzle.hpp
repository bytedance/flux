//===- threadblock_swizzle.hpp ------------------------------------ C++ ---===//
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
struct ThreadblockSwizzleStreamKRankOffset : public ThreadblockSwizzleStreamK {
 public:
  int local_rank;
  int local_world_size;

  /// Constructor
  CUTLASS_HOST_DEVICE
  ThreadblockSwizzleStreamKRankOffset() {}

  ThreadblockSwizzleStreamKRankOffset(
      GemmUniversalMode const mode_,
      GemmCoord const problem_size_,
      GemmCoord const tile_size_,
      int const batch_split_,  /// Either (mode == GemmUniversalMode::kBatched) the batch count, or
                               /// (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1
                               /// defaults to StreamK, >1 emulates Split-K)
      int const sm_occupancy_,
      int const device_sms_,
      int const avail_sms_,
      size_t const element_A_bytes_,
      size_t const element_B_bytes_,
      size_t const element_C_bytes_,
      int const epilogue_acc_fragments_)
      : ThreadblockSwizzleStreamK(
            mode_,
            problem_size_,
            tile_size_,
            batch_split_,
            sm_occupancy_,
            device_sms_,
            avail_sms_,
            element_A_bytes_,
            element_B_bytes_,
            element_C_bytes_,
            epilogue_acc_fragments_) {
    const char *local_rank_str = std::getenv("LOCAL_RANK");
    const char *local_world_size_str = std::getenv("LOCAL_WORLD_SIZE");
    if (local_rank_str && local_world_size_str) {
      local_rank = atoi(local_rank_str);
      local_world_size = atoi(local_world_size_str);
    }
  }

  /// Obtains the calling threadblock's tiled coordinates for the given tile index
  CUTLASS_DEVICE
  GemmCoord
  get_tile_offset(int tile_idx) const {
    auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);
    int tiled_m = ThreadblockSwizzleStreamK::tiled_shape().m();
    // Guard for tiles that exceed the tiled_m
    if (coord.m() >= tiled_m)
      return coord;
    int m = (coord.m() + tiled_m / local_world_size * (local_rank)) % tiled_m;
    coord.m() = m;
    return coord;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
