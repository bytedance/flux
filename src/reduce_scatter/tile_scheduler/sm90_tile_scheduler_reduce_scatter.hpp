//===- sm90_tile_scheduler_reduce_scatter.hpp --------------------- C++ ---===//
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
#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"

#include "./tile_mappings.hpp"

namespace cutlass::gemm::kernel::detail {

using bytedance::flux::detail::WorkIdxMSwizzler;

///////////////////////////////////////////////////////////////////////////////

// Persistent Thread Block (TB) scheduler
class PersistentTileSchedulerSm90ReduceScatter : public PersistentTileSchedulerSm90 {
 public:
  using Base = PersistentTileSchedulerSm90;
  using WorkTileInfo = Base::WorkTileInfo;

  struct Arguments : Base::Arguments {
    int rank = 0;
    int world_size = 1;
    int nnodes = 1;
  };

  struct Params : public Base::Params {
    typename WorkIdxMSwizzler::Params swizzler;
  };

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
      [[maybe_unused]] const uint32_t epilogue_subtile = 1) {
    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Base::Params scheduler_params;
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

  WorkIdxMSwizzler swizzler;

  CUTLASS_HOST_DEVICE
  PersistentTileSchedulerSm90ReduceScatter() {}

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90ReduceScatter(Params const &params_)
      : Base(params_), swizzler(params_.swizzler) {}

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    auto work_tile_info = Base::get_current_work();
    if (work_tile_info.is_valid()) {
      work_tile_info.M_idx = swizzler.swizzle_m(work_tile_info.M_idx);
    }

    return work_tile_info;
  }
};

template <class TileShape, class ClusterShape>
class PersistentTileSchedulerSm90ReduceScatterStreamK
    : public PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape> {
 public:
  using UnderlyingScheduler = PersistentTileSchedulerSm90;
  using Base = PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;
  using WorkTileInfo = typename Base::WorkTileInfo;

  struct Arguments : public Base::Arguments {
    using Base::Arguments::Arguments;
    int rank = 0;
    int world_size = 1;
    int nnodes = 1;
  };

  struct Params : public Base::Params {
    typename WorkIdxMSwizzler::Params swizzler;
  };

  template <class ProblemShape>
  static Params
  to_underlying_arguments(
      ProblemShape problem_shape,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      KernelHardwareInfo const &hw_info,
      Arguments const &args,
      [[maybe_unused]] void *workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1) {
    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    dim3 problem_blocks = UnderlyingScheduler::get_tiled_cta_shape_mnl(
        problem_shape_mnkl, tile_shape, cluster_shape);

    typename Base::Params scheduler_params = Base::to_underlying_arguments(
        problem_shape, tile_shape, cluster_shape, hw_info, args, workspace);

    WorkIdxMSwizzler::Params swizzler_params;
    int problem_blocks_m = round_up(
        problem_blocks.x, (1 << scheduler_params.log_swizzle_size_) * cute::get<0>(cluster_shape));

    swizzler_params = WorkIdxMSwizzler::to_underlying_arguments(
        args.rank, args.world_size, args.nnodes, problem_blocks_m);
    return {scheduler_params, swizzler_params};
  }

  WorkIdxMSwizzler swizzler;

  CUTLASS_HOST_DEVICE
  PersistentTileSchedulerSm90ReduceScatterStreamK() {};

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90ReduceScatterStreamK(Params const &params_)
      : Base(params_), swizzler(params_.swizzler) {}

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() const {
    auto work_tile_info = Base::get_current_work();
    // if (work_tile_info.is_valid_tile) {
    // FIXME: swizzle may cause hang
    // work_tile_info.M_idx = swizzler.swizzle_m(work_tile_info.M_idx);
    // }
    return work_tile_info;
  }
};

}  // namespace cutlass::gemm::kernel::detail
