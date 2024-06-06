//===- tile_mapping.hpp ------------------------------------------- C++ ---===//
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
#include "flux/flux.h"
#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include "cutlass/detail/helper_macros.hpp"

namespace bytedance::flux::detail {

using namespace cute;

struct OutputRankSwizzler {
  static constexpr int kMaxMLogSwizzle = 3;
  static constexpr int kMaxM = 1 << kMaxMLogSwizzle;
  // RankShape: (kMaxM, kMaxM). The first dim is the current tile's local_rank.
  // The second dim is current rank's local_rank.
  using RankShape = Shape<Int<kMaxM>, Int<kMaxM>>;
  // Use swizzle to make the tile's local_rank mapped to different order on each rank.
  using SwizzleLayout =
      decltype(composition(Swizzle<kMaxMLogSwizzle, 0, kMaxMLogSwizzle>{}, Layout<RankShape>{}));
  // Squeeze out the second dim from an offset in RankShape
  // E.g: The order on rank #1 is [9, 8, 11, 10, 13, 12, 15, 14].
  // We use this layout to map it to [1, 0, 3, 2, 5, 4, 7, 6]
  using SqueezeLayout = Layout<RankShape, Stride<_1, _0>>;

  struct Params {
    int swizzle_idx;
  };

  Params params;

  static Params
  to_underlying_arguments(int swizzle_idx) {
    return {swizzle_idx};
  }

  CUTLASS_HOST_DEVICE
  OutputRankSwizzler() {}

  CUTLASS_HOST_DEVICE explicit OutputRankSwizzler(Params const &params) : params(params) {}

  CUTLASS_HOST_DEVICE int
  swizzle_rank(int rank) const {
    constexpr auto swizzle_layout = SwizzleLayout{};
    constexpr auto squeeze_layout = SqueezeLayout{};

    int rank_swizzled = squeeze_layout(swizzle_layout(rank, params.swizzle_idx));
    return rank_swizzled;
  }
};

/// Given a tile's work_idx_M, compute a swizzled values.
/// This is mainly used to make
struct WorkIdxMSwizzler {
  /// (problem_blocks_per_m, nnodes, local_world_size)
  using WorkIdxMLayout = cute::Layout<cute::Shape<int32_t, int32_t, int32_t>>;

  struct Params {
    WorkIdxMLayout work_idx_m_layout;
    typename OutputRankSwizzler::Params rank_swizzler_params;
  };

  WorkIdxMLayout work_idx_m_layout;
  OutputRankSwizzler rank_swizzler;

  CUTLASS_HOST_DEVICE
  WorkIdxMSwizzler() {}

  CUTLASS_HOST_DEVICE explicit WorkIdxMSwizzler(Params const &params_)
      : work_idx_m_layout(params_.work_idx_m_layout),
        rank_swizzler(params_.rank_swizzler_params) {}

  static Params
  to_underlying_arguments(int rank, int world_size, int nnodes, int problem_blocks_m) {
    // calculate offsets on M
    int local_world_size = world_size / nnodes;
    FLUX_CHECK(local_world_size * nnodes == world_size);
    FLUX_CHECK(local_world_size <= OutputRankSwizzler::kMaxM);  // Swizzle requirement
    FLUX_CHECK(
        local_world_size ==
        (local_world_size & -local_world_size));  // require to be power of 2 for swizzling
    int local_rank = rank % local_world_size;
    FLUX_CHECK(problem_blocks_m % world_size == 0);

    int problem_blocks_m_per_rank = problem_blocks_m / world_size;
    auto work_idx_m_layout = cute::make_layout(
        cute::make_shape(problem_blocks_m_per_rank, nnodes, local_world_size),
        cute::make_stride(
            _1{}, local_world_size * problem_blocks_m_per_rank, problem_blocks_m_per_rank));

    // If across node, we let tiles of the current rank to be calculated last,
    // because they will waiting for other local ranks to finish reduction
    int swizzle_idx = local_rank;
    return {work_idx_m_layout, OutputRankSwizzler::to_underlying_arguments(swizzle_idx)};
  }

  CUTLASS_HOST_DEVICE int
  swizzle_m(int m_idx) const {
    using namespace cute;
    auto [offset, node_idx, local_rank_tile] = work_idx_m_layout.get_flat_coord(m_idx);
    int local_rank_swizzled = rank_swizzler.swizzle_rank(local_rank_tile);
    int m_idx_swizzled = work_idx_m_layout(offset, node_idx, local_rank_swizzled);
    return m_idx_swizzled;
  }
};

}  // namespace bytedance::flux::detail
