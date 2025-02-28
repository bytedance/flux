//===- sm90_tile_scheduler_reduce_scatter.hpp --------------------- C++ ---===//
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
#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"

#include "./tile_mappings.hpp"

namespace cutlass::gemm::kernel::detail {

using bytedance::flux::detail::WorkIdxMSwizzler;

///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
class PersistentTileSchedulerSm90ReduceScatter {
  using UnderlyingScheduler = PersistentTileSchedulerSm90;

 private:
  UnderlyingScheduler underlying_scheduler_;
  WorkIdxMSwizzler swizzler_;

 public:
  using WorkTileInfo = typename UnderlyingScheduler::WorkTileInfo;
  using RasterOrder = typename UnderlyingScheduler::RasterOrder;
  using RasterOrderOptions = typename UnderlyingScheduler::RasterOrderOptions;
  static constexpr bool IsDynamicPersistent = false;

 public:
  struct Arguments : UnderlyingScheduler::Arguments {
    int rank = 0;
    int world_size = 1;
    int nnodes = 1;
  };

  struct Params : UnderlyingScheduler::Params {
    typename WorkIdxMSwizzler::Params swizzler;
  };

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      [[maybe_unused]] KernelHardwareInfo const &hw_info,
      Arguments const &arguments,
      [[maybe_unused]] void *workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {
    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    UnderlyingScheduler::Params scheduler_params;
    scheduler_params.initialize(
        problem_blocks,
        to_gemm_coord(cluster_shape),
        hw_info,
        arguments.max_swizzle_size,
        arguments.raster_order);

    WorkIdxMSwizzler::Params swizzler_params;
    int problem_blocks_m = round_up(
        problem_blocks.x, (1 << scheduler_params.log_swizzle_size_) * cute::get<0>(cluster_shape));

    swizzler_params = WorkIdxMSwizzler::to_underlying_arguments(
        arguments.rank, arguments.world_size, arguments.nnodes, problem_blocks_m);

    return {scheduler_params, swizzler_params};
  }

  CUTLASS_HOST_DEVICE
  static bool
  can_implement(Arguments const &args) {
    return UnderlyingScheduler::can_implement(args);
  }

  CUTLASS_HOST_DEVICE
  PersistentTileSchedulerSm90ReduceScatter() {}

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90ReduceScatter(Params const &params_)
      : underlying_scheduler_(params_), swizzler_(params_.swizzler) {}

  // Returns the initial work tile info that will be computed over
  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    return get_current_work();
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    auto work_tile_info = underlying_scheduler_.get_current_work();
    if (work_tile_info.is_valid()) {
      work_tile_info.M_idx = swizzler_.swizzle_m(work_tile_info.M_idx);
    }

    return work_tile_info;
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx) const {
    return underlying_scheduler_.get_current_work_for_linear_idx(linear_idx);
  }

  CUTLASS_DEVICE
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    underlying_scheduler_.advance_to_next_work(advance_count);
  }

  // Computes the linear index within a batch given M and N tile offsets within the batch.
  // This essentially inverts the mapping performed in get_work_idx_m_and_n
  template <class... Args>
  static CUTLASS_DEVICE uint64_t
  get_linear_idx_from_m_and_n(Args... args) {
    return UnderlyingScheduler::get_linear_idx_from_m_and_n(args...);
  }

  template <class... Args>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(Args... args) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(args...);
  }

  // Reloaded interface that receives WorkTileInfo to deduce next work.
  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto
  fetch_next_work(WorkTileInfo work_tile_info) {
    if (continue_current_work(work_tile_info)) {
      return cute::make_tuple(work_tile_info, true);
    }

    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  CUTLASS_DEVICE
  static auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
    return UnderlyingScheduler::work_tile_to_cta_coord(work_tile_info);
  }

  CUTLASS_DEVICE
  static auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info, dim3 block_id_in_cluster) {
    return UnderlyingScheduler::work_tile_to_cta_coord(work_tile_info, block_id_in_cluster);
  }

  // Given the inputs, computes the physical grid we should launch.
  template <class... Args>
  CUTLASS_HOST_DEVICE static dim3
  get_grid_shape(Args... args) {
    return UnderlyingScheduler::get_grid_shape(args...);
  }

  // Convert CTA-level work tile info to cluster-level tile coord
  CUTLASS_DEVICE
  auto
  work_tile_to_cluster_coord_mnkl(WorkTileInfo work_tile_info) const {
    return underlying_scheduler_.work_tile_to_cluster_coord_mnkl(work_tile_info);
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the basic tile scheduler, this is always true.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const &, Params const &) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const &) {
    return true;
  }

  // Performs the reduction across splits for a given output tile. Since this scheduler does
  // not split output tiles, no reduction is needed.
  template <class FrgTensorC>
  CUTLASS_DEVICE static void
  fixup(Params const &, WorkTileInfo const &, FrgTensorC &, uint32_t, uint32_t) {}

  // Performs the reduction across splits for a given output tile. No fixup is required for
  // work units returned by this scheduler.
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  fixup(WorkTileInfo const &, FrgTensorC &, uint32_t, uint32_t) const {}

  // Returns whether the current WorkTileInfo passed in should continue to be used. Since
  // this scheduler only schedules work in units of single, full output tiles, the WorkTileInfo
  // passed in should not be used after having been processed.
  CUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo &) {
    return false;
  }

  template <class ProblemShapeMNKL, class TileShape, class Shape>
  CUTLASS_DEVICE auto
  get_k_tile_iterator(
      WorkTileInfo const &work_tile_info,
      ProblemShapeMNKL problem_shape_MNKL,
      TileShape tile_shape,
      Shape) {
    auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape));
    return cute::make_coord_iterator(k_tiles);
  }

  template <class ProblemShape, class TileShape>
  CUTLASS_HOST_DEVICE static int
  get_work_k_tile_count(
      WorkTileInfo const &work_tile_info, ProblemShape problem_shape, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const &) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_DEVICE
  static bool
  need_separate_reduction(Params const &params) {
    return false;
  }

  CUTLASS_DEVICE
  bool
  is_work_tile_for_reduction(WorkTileInfo const &work_tile_info, Params const &params) {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE void
  separate_reduction(Args... args) {
    underlying_scheduler_.separate_reduction(args...);
  }

  // Shares the accumulator set with peers in the global workspace
  template <class... Args>
  CUTLASS_DEVICE static void
  share(Args... args) {
    UnderlyingScheduler::share(args...);
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const &work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool
  requires_separate_reduction(Params const &params) {
    return false;
  }

  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  template <class... Args>
  static CUTLASS_DEVICE cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(Args... args) {
    return UnderlyingScheduler::get_work_idx_m_and_n(args...);
  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(
      Arguments const &,
      ProblemShape,
      KernelHardwareInfo const &,
      uint32_t,
      const uint32_t = 1,
      uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(
      Arguments const &,
      void *,
      cudaStream_t,
      ProblemShape,
      KernelHardwareInfo const &,
      uint32_t,
      const uint32_t = 1,
      uint32_t = 1,
      CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }
};
}  // namespace cutlass::gemm::kernel::detail
