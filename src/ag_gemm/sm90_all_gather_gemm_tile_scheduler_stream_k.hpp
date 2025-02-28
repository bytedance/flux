//===- sm90_all_gather_gemm_tile_scheduler_stream_k.hpp ----------- C++ ---===//
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
// Modified based on cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp

#pragma once

#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel::detail {

// Persistent Thread Block (TB) scheduler leveraging stream-K decomposition
template <class TileShape, class ClusterShape>
class Sm90AGKernelPersistentTileSchedulerSm90StreamK
    : public PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape> {
  //
  // Data members
  //

 private:
  using Base = PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;

 public:
  static constexpr bool IsDynamicPersistent = false;

  using BarrierType = typename Base::BarrierType;

  using StreamKWorkTileInfo = typename Base::WorkTileInfo;
  using StreamKArguments = typename Base::Arguments;
  using StreamKBaseParams = typename Base::Params;

  struct WorkTileInfo : StreamKWorkTileInfo {
    // M_idx before ag swizzle
    int32_t M_idx_ori = 0;
    CUTLASS_HOST_DEVICE
    StreamKWorkTileInfo
    get_original_work_tile_info() const {
      return StreamKWorkTileInfo{
          .M_idx = this->M_idx_ori,
          .N_idx = this->N_idx,
          .K_idx = this->K_idx,
          .L_idx = this->L_idx,
          .k_tile_count = this->k_tile_count,
          .k_tile_remaining = this->k_tile_remaining,
          .is_separate_reduction = this->is_separate_reduction};
    }
  };

  struct Arguments : StreamKArguments {
    // for ag
    int nnodes = 1;
    int rank = -1;
    int world_size = 1;
    int local_world_size = 1;
    int local_rank = 0;
  };

  struct SwizzleParams {
    int problem_blocks_m_offset = 0;
    int problem_blocks_m = 0;
    int nnodes = 1;
    int node_id = 0;
  };

  struct Params : StreamKBaseParams {
    SwizzleParams swizzle_params;
  };

  SwizzleParams swizzle_params;

  //
  // Methods
  //

  template <class ProblemShape>
  static Params
  to_underlying_arguments(
      ProblemShape problem_shape,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo const &hw_info,
      Arguments const &args,
      void *workspace,
      const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    dim3 problem_blocks =
        Base::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    uint32_t k_tile_per_output_tile = cute::size(
        cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    StreamKBaseParams base_params;
    base_params.initialize(
        problem_blocks,
        k_tile_per_output_tile,
        to_gemm_coord(cluster_shape),
        hw_info,
        args.splits,
        args.max_swizzle_size,
        args.raster_order,
        args.reduction_mode,
        args.decomposition_mode,
        workspace,
        epilogue_subtile);

    int nnodes = args.nnodes;
    int rank = args.rank;
    int world_size = args.world_size;
    int local_world_size = args.local_world_size;
    FLUX_CHECK(local_world_size * nnodes == world_size);
    int node_id = rank / local_world_size;
    // calculate offsets on M
    int problem_blocks_m = problem_blocks.x;
    FLUX_CHECK(local_world_size != 0);

    int M_per_rank = static_cast<int>(get<0>(problem_shape_mnkl)) / world_size;
    int M_start = M_per_rank * rank;
    int problem_blocks_m_offset = (M_start + get<0>(tile_shape) - 1) / get<0>(tile_shape);
    return {
        base_params, SwizzleParams{problem_blocks_m_offset, problem_blocks_m, nnodes, node_id}};
  }

  CUTLASS_HOST_DEVICE
  Sm90AGKernelPersistentTileSchedulerSm90StreamK() {};

  CUTLASS_HOST_DEVICE
  Sm90AGKernelPersistentTileSchedulerSm90StreamK(Params const &params_)
      : Base(params_), swizzle_params(params_.swizzle_params) {}

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() {
    auto base_work_tile_info = Base::get_current_work();
    // Apply ag swizzle. And record origin M_idx, will be used in fixup.
    WorkTileInfo work_tile_info = {base_work_tile_info, base_work_tile_info.M_idx};
    work_tile_info.M_idx = ag_swizzle_m(swizzle_params, work_tile_info);
    return work_tile_info;
  }

  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto
  fetch_next_work(WorkTileInfo work_tile_info) {
    // sk tile may be divided into multiple runs
    if (continue_current_work(work_tile_info)) {
      // Also need to apply ag swizzle here. Because `continue_current_work` may call the
      // `assign_work` if current sk tile has remaining k tile, then the original(no ag swizzle)
      // M/N/L idx will be set again.
      work_tile_info.M_idx_ori = work_tile_info.M_idx;
      work_tile_info.M_idx = ag_swizzle_m(swizzle_params, work_tile_info);
      return cute::make_tuple(work_tile_info, true);
    }

    Base::advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  // Returns the initial work tile info that will be computed over
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape) {
    return get_current_work();
  }

  // Performs the reduction across splits for a given output tile.
  template <class FrgTensorC>
  CUTLASS_DEVICE static void
  fixup(
      Params const &params,
      WorkTileInfo const &work_tile_info,
      FrgTensorC &accumulators,
      uint32_t num_barriers,
      uint32_t barrier_idx) {
    // Why use the original M_idx(no ag swizzle)?
    // Streamk scheduler use the first few output tiles(assuming there are X) as sk tiles, and
    // only allocate workspace(barrier and partial sum) for these sk tiles. In fixup, it calculate
    // tile_idx using M_idx and N_idx, then use tile_idx to calculate offset of workspace. There is
    // an assumption that the value of tile_idx is in \[0, X). if we use M_idx after ag swizzle,
    // it will be out of bound.
    const StreamKWorkTileInfo ori_work_tile_info = work_tile_info.get_original_work_tile_info();
    Base::fixup(params, ori_work_tile_info, accumulators, num_barriers, barrier_idx);
  }

 private:
  CUTLASS_DEVICE
  static int32_t
  ag_swizzle_m(SwizzleParams const &params, WorkTileInfo const &work_tile_info) {
    int new_M_idx;
    int M_idx = work_tile_info.M_idx;
    if (work_tile_info.is_valid()) {
      if (params.nnodes > bytedance::flux::kNodes) {
        int local_block_m = params.problem_blocks_m / params.nnodes;
        int local_id = M_idx / local_block_m;

        auto m = (M_idx + params.problem_blocks_m_offset) % local_block_m;
        new_M_idx =
            (m + (params.node_id - local_id + params.nnodes) % params.nnodes * local_block_m) %
            params.problem_blocks_m;
      } else if (params.nnodes > 1) {
        int local_block_m = params.problem_blocks_m / params.nnodes;
        int local_id = M_idx / local_block_m;

        auto m = (M_idx + params.problem_blocks_m_offset) % local_block_m;
        new_M_idx = (m + bytedance::flux::nodes_swizzle[params.nnodes / 2]
                                 .swizzle[params.node_id * params.nnodes + local_id] *
                             local_block_m) %
                    params.problem_blocks_m;
      } else {
        new_M_idx = (M_idx + params.problem_blocks_m_offset) % params.problem_blocks_m;
      }
      return new_M_idx;
    }
    return M_idx;
  }
};

}  // namespace cutlass::gemm::kernel::detail