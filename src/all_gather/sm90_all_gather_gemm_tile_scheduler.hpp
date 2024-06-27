//===- sm90_all_gather_gemm_tile_scheduler.hpp -------------------- C++ ---===//
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

#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"

#include "cute/util/debug.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/swizzle_layout.hpp"

#include "cutlass/barrier.h"
#include "flux/flux.h"
#include "flux/cuda/memory_utils.hpp"

#include "all_gather/all_gather_swizzle.hpp"

namespace cutlass::gemm::kernel::detail {
using SystemBarrier = cutlass::detail::SystemBarrier;
///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
class Sm90AGKernelTileScheduler : public PersistentTileSchedulerSm90 {
 public:
  using Base = PersistentTileSchedulerSm90;
  using WorkTileInfo = Base::WorkTileInfo;

  struct Arguments : Base::Arguments {
    int nnodes = 1;
    int rank = 0;
    int world_size = 1;
    int local_world_size = 1;
    int local_rank = 0;
    SystemBarrier::T *ptr_barrier = nullptr;
  };

  struct Params : public Base::Params {
    using WorkIdxMLayout = cute::Layout<
        cute::Shape<int32_t, int32_t, cute::Int<2>>,
        cute::Stride<cute::Int<1>, int32_t, cute::Int<0>>>;

    SystemBarrier::T *ptr_barrier = nullptr;
    int problem_blocks_m_offset = 0;
    int problem_blocks_m = 0;

    int nnodes = 1;
    int node_id = 0;
  };

  // Sink scheduler params as a member
  Params scheduler_params;

  //
  // Methods
  //

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      [[maybe_unused]] KernelHardwareInfo const &hw_info,
      Arguments const &arguments,
      [[maybe_unused]] void *workspace = nullptr,
      [[maybe_unused]] int epilogue_subtile = 1) {
    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Base::Params base_params;
    base_params.initialize(
        problem_blocks,
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        arguments.raster_order);

    int nnodes = arguments.nnodes;
    int rank = arguments.rank;
    int world_size = arguments.world_size;
    int local_world_size = arguments.local_world_size;
    FLUX_CHECK(local_world_size * nnodes == world_size);
    // int local_rank = arguments.local_rank;
    int node_id = rank / local_world_size;
    // calculate offsets on M
    int problem_blocks_m = problem_blocks.x;
    FLUX_CHECK(local_world_size != 0);

    // int problem_blocks_m_offset = problem_blocks_m / local_world_size * local_rank;
    int M_per_rank = static_cast<int>(get<0>(problem_shape_mnkl)) / world_size;
    int M_start = M_per_rank * rank;
    int problem_blocks_m_offset = (M_start + get<0>(tile_shape) - 1) / get<0>(tile_shape);

    return {
        base_params,
        arguments.ptr_barrier,
        problem_blocks_m_offset,
        problem_blocks_m,
        nnodes,
        node_id};
  }

  CUTLASS_HOST_DEVICE
  Sm90AGKernelTileScheduler() {}

  CUTLASS_DEVICE explicit Sm90AGKernelTileScheduler(Params const &params_)
      : Base(params_), scheduler_params(params_) {}

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    auto work_tile_info = Base::get_current_work();
    if (work_tile_info.is_valid()) {
      int new_M_idx;
      auto M_idx = work_tile_info.M_idx;

      if (scheduler_params.nnodes > bytedance::flux::kNodes) {
        int local_block_m = scheduler_params.problem_blocks_m / scheduler_params.nnodes;
        int local_id = M_idx / local_block_m;

        auto m = (M_idx + scheduler_params.problem_blocks_m_offset) % local_block_m;
        new_M_idx = (m + (scheduler_params.node_id - local_id + scheduler_params.nnodes) %
                             scheduler_params.nnodes * local_block_m) %
                    scheduler_params.problem_blocks_m;
      } else if (scheduler_params.nnodes > 1) {
        int local_block_m = scheduler_params.problem_blocks_m / scheduler_params.nnodes;
        int local_id = M_idx / local_block_m;

        auto m = (M_idx + scheduler_params.problem_blocks_m_offset) % local_block_m;
        new_M_idx =
            (m + bytedance::flux::nodes_wizzle[scheduler_params.nnodes / 2]
                         .swizzle[scheduler_params.node_id * scheduler_params.nnodes + local_id] *
                     local_block_m) %
            scheduler_params.problem_blocks_m;
      } else {
        new_M_idx =
            (M_idx + scheduler_params.problem_blocks_m_offset) % scheduler_params.problem_blocks_m;
      }
      work_tile_info.M_idx = new_M_idx;
    }

    return work_tile_info;
  }
};

}  // namespace cutlass::gemm::kernel::detail
