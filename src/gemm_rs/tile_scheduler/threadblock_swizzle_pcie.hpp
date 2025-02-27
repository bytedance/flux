//===- threadblock_swizzle_pcie.hpp ----------------------------- C++ ---===//
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

#include <cassert>
#include "cutlass/detail/helper_macros.hpp"
#include "flux/cuda/cuda_common_device.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/gemm_coord.h"
#include "gemm_rs/reduce_scatter_topos.hpp"
#include "flux/utils.h"
#include "flux/flux.h"
#include "flux/cuda/cuda_common.h"
#include "threadblock_swizzle_segment_util.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
bytedance::flux::SegmentInfo *segments_global_device = []() {
  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, sizeof(bytedance::flux::SegmentInfo) * 1000));
  // bytedance::flux::SegmentInfo *ptr_device;
  return (bytedance::flux::SegmentInfo *)ptr;
}();
}  // namespace

namespace cutlass {
namespace gemm {
namespace threadblock {

namespace {
CUTLASS_HOST_DEVICE
int
m_by_gemmk(int m, int tiled_size_m, int world_size) {
  int m_per_rank = m / world_size;
  int m_per_rank_fixed = (m_per_rank + tiled_size_m - 1) / tiled_size_m * tiled_size_m;
  return m_per_rank_fixed * world_size;
}
}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
struct ThreadblockSwizzlePcie {
 private:
  cutlass::gemm::GemmCoord tiled_shape;
  int world_size;
  bytedance::flux::SegmentInfo *segments;

 public:
  /// Constructor
  CUTLASS_HOST_DEVICE
  ThreadblockSwizzlePcie() {}

  CUTLASS_HOST_DEVICE
  ThreadblockSwizzlePcie(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::gemm::GemmCoord tile_size_,
      int world_size_,
      void *args_workspace_)
      : tiled_shape(
            (problem_size_.m() + tile_size_.m() - 1) / tile_size_.m(),
            (problem_size_.n() + tile_size_.n() - 1) / tile_size_.n(),
            1),
        world_size(world_size_),
        segments(reinterpret_cast<bytedance::flux::SegmentInfo *>(args_workspace_)) {}

  /// Obtains the calling threadblock's tiled coordinates for the given tile index
  [[nodiscard]] CUTLASS_DEVICE GemmCoord
  get_tile_offset(int tile_idx) const {
    int tiled_m = tiled_shape.m(), tiled_n = tiled_shape.n();
    if (tile_idx >= tiled_m * tiled_n) {
      return GemmCoord(INT_MAX, INT_MAX, 0);
    }
    int tile_m_origin = tile_idx / tiled_n;
    int tile_n_origin = tile_idx % tiled_n;

    int segment_id = 0;
    // TODO(houqi.1993) using warp level instruction to simplify
    for (; segment_id < world_size; segment_id++) {
      const auto &segment = segments[segment_id];
      bool in_segment = segment.size > 0 && tile_m_origin >= segment.tile_m_start_origin &&
                        tile_m_origin < segment.tile_m_start_origin + segment.size;
      if (in_segment) {
        break;
      }
    }

    const auto &segment_new = segments[segment_id];
    int tiled_m_offset = tile_m_origin - segment_new.tile_m_start_origin;
    // using custom-cohort
    int inner_idx = tiled_n * tiled_m_offset + tile_n_origin;
    int inner_m = inner_idx % segment_new.size;
    int inner_n = inner_idx / segment_new.size;
    auto coord = GemmCoord(segment_new.tile_m_start_new + inner_m, inner_n, 0);
    return coord;
  }
};  // namespace threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////

class ThreadblockSwizzleStreamKPcie : public ThreadblockSwizzleStreamK {
 private:
  ThreadblockSwizzlePcie custom_swizzle_;

 public:
  CUTLASS_HOST_DEVICE
  ThreadblockSwizzleStreamKPcie() {}

  ThreadblockSwizzleStreamKPcie(
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
      int rank_,
      int world_size_,
      int nnodes_,
      bool use_gemmk_,
      bool per_tile_flags_,
      bool use_1d_ring_,
      void *args_workspace_)
      : ThreadblockSwizzleStreamK(
            mode_,
            GemmCoord(
                use_gemmk_ ? m_by_gemmk(problem_size_.m(), tile_size_.m(), world_size_)
                           : problem_size_.m(),
                problem_size_.n(),
                problem_size_.k()),
            tile_size_,
            batch_split_,
            sm_occupancy_,
            device_sms_,
            avail_sms_,
            element_A_bytes_,
            element_B_bytes_,
            element_C_bytes_,
            epilogue_acc_fragments_),
        custom_swizzle_(
            GemmCoord(
                use_gemmk_ ? m_by_gemmk(problem_size_.m(), tile_size_.m(), world_size_)
                           : problem_size_.m(),
                problem_size_.n(),
                problem_size_.k()),
            tile_size_,
            world_size_,
            args_workspace_) {}

  /// Obtains the calling threadblock's tiled coordinates for the given tile index
  [[nodiscard]] CUTLASS_DEVICE GemmCoord
  get_tile_offset(int tile_idx) const {
    auto coord = custom_swizzle_.get_tile_offset(tile_idx);
    return coord;
  }
};

// forcing N=1 and leave threadblock swizzle efficiency to me
struct GemmIdentityThreadblockSwizzlePcie : public GemmHorizontalThreadblockSwizzle {
 private:
  ThreadblockSwizzlePcie custom_swizzle_;
  GemmCoord const tiled_shape;

 public:
  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzlePcie() = default;

  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzlePcie(
      GemmCoord const problem_size_,
      GemmCoord const tile_size_,
      GemmCoord const tiled_shape_,
      int rank_,
      int world_size_,
      int nnodes_,
      bool use_gemmk_,
      bool per_tile_flags_,
      bool use_1d_ring_,
      void *args_workspace_)
      : custom_swizzle_(
            GemmCoord(
                use_gemmk_ ? m_by_gemmk(problem_size_.m(), tile_size_.m(), world_size_)
                           : problem_size_.m(),
                problem_size_.n(),
                problem_size_.k()),
            tile_size_,
            world_size_,
            args_workspace_),
        tiled_shape(tiled_shape_) {}

  [[nodiscard]] CUTLASS_DEVICE GemmCoord
  get_tile_offset(int log_tile /** not used */) const {
    cutlass::gemm::GemmCoord coord{(int)blockIdx.y, (int)blockIdx.x, (int)blockIdx.z};  // along-N
    int tile_idx = coord.m() * tiled_shape.n() + coord.n();
    return custom_swizzle_.get_tile_offset(tile_idx);
  }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
