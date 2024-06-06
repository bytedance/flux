//===- sm80_all_gather_gemm_threadblock_swizzle.hpp --------------- C++ ---===//
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "flux/utils.h"
#include "all_gather/all_gather_swizzle.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct AGThreadblockSwizzleStreamKRankOffset : public ThreadblockSwizzleStreamK {
 public:
  int nnodes;
  int node_id;
  int rank;
  int world_size;
  int local_rank;
  int local_world_size;
  int raster_order;
  int TILE_SIZE_M;
  int tile_m_offset;
  int tiled_m;
  int local_tiled_m;

  /// Constructor
  CUTLASS_HOST_DEVICE
  AGThreadblockSwizzleStreamKRankOffset() {}

  AGThreadblockSwizzleStreamKRankOffset(
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
      int const epilogue_acc_fragments_,
      int const raster_order_,
      int rank_,
      int world_size_,
      int nnodes_)
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
            epilogue_acc_fragments_),
        nnodes(nnodes_),
        rank(rank_),
        world_size(world_size_) {
    FLUX_CHECK(world_size % nnodes == 0);
    local_world_size = world_size / nnodes;
    local_rank = rank % local_world_size;
    node_id = rank / local_world_size;

    raster_order = raster_order_;
    TILE_SIZE_M = tile_size_.m();

    int M = ThreadblockSwizzleStreamK::problem_size.m();
    int M_per_rank = M / world_size;
    int M_start = M_per_rank * rank;
    tile_m_offset = (M_start + TILE_SIZE_M - 1) / TILE_SIZE_M;

    tiled_m = ThreadblockSwizzleStreamK::tiled_shape().m();
    local_tiled_m = tiled_m / nnodes;
  }

  /// Obtains the calling threadblock's tiled coordinates for the given tile index
  CUTLASS_DEVICE
  GemmCoord
  get_tile_offset(int tile_idx) const {
    auto coord = (raster_order == 0)
                     ? ThreadblockSwizzleStreamK::get_tile_offset_row_major(tile_idx)
                     : ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);

    if (nnodes > 1) {
      int local_id = coord.m() / local_tiled_m;
      int m = (coord.m() + tile_m_offset) % local_tiled_m;
      int new_m = (m + (node_id - local_id + nnodes) % nnodes * local_tiled_m) % tiled_m;

      coord.m() = new_m;
      return coord;
    } else {  // nnodes = 1
      int m = (coord.m() + tile_m_offset) % tiled_m;
      // int m = (coord.m() + tiled_m / local_world_size * (local_rank)) % tiled_m;
      coord.m() = m;
      return coord;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
