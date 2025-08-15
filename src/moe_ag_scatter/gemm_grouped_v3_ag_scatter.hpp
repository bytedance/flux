//===- gemm_grouped_v3_ag_scatter.hpp -------------------------- C++ ------===//
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
#include <memory>
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/int_tuple.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "flux/cuda/cutlass_v3_builder.hpp"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/args/moe_ag_scatter.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_grouped_v3_impl.hpp"
#include "./dispatch_policy.hpp"
#include "./sm90_gemm_array_warpspecialized_cooperative.hpp"
#include "./sm90_gemm_array_warpspecialized_pingpong.hpp"
#include "./sm90_mma_warpspecialized.hpp"
#include "./scatter_epilogue_array.hpp"
#include "./gather_tensor.hpp"
#include "./sm90_ag_scatter_fetcher.hpp"
#include "./sm90_tile_scheduler_group_ag_scatter.hpp"
#include "moe_ag_scatter/sort_util.h"

namespace bytedance {
namespace flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV3AGScatter_Kernel : public GemmGroupedV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmGroupedV3BaseKernel<GemmMetaT, GemmHParamsT>;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static_assert(
      cute::
          is_same_v<cutlass::layout::RowMajor, decltype(to_cutlass_layout_c(meta.gemm_layout()))>,
      "output must be row-major");
  static constexpr auto gemm_v3_hparams = to_gemm_v3_hparams(hparams.impl_spec());
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  static_assert(meta.comm_op() == _AGScatter{}, "requires _AGScatter{}");

  using GatherA = IndexedGatherArray<int32_t>;
  using ScatterD = IndexedGather<int32_t>;
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using AGFetcherA = Sm90AgScatterFetcher;

  auto
  tile_scheduler() const {
    return make_declval<cutlass::gemm::GroupSchedulerAGScatter>();
  }

  auto
  collective_mma() const {
    using namespace cutlass_v3_builder;
    auto base_params = base_mainloop_params(meta, hparams);
    using CollectiveEpilogue = identity_t<decltype(this->collective_epilogue())>;
    constexpr int epi_smem_size = sizeof(typename CollectiveEpilogue::SharedStorage);

    auto get_default_schedule = []() {
      if constexpr (gemm_v3_hparams.kernel_schedule() == _Cooperative{}) {
        return TypeWrapper<cutlass::gemm::KernelCpAsyncWarpSpecializedCooperative>{};
      } else if constexpr (gemm_v3_hparams.kernel_schedule() == _PingPong{}) {
        return TypeWrapper<cutlass::gemm::KernelCpAsyncWarpSpecializedPingpong>{};
      } else {
        static_assert(
            cutlass::detail::dependent_false<decltype(gemm_v3_hparams.kernel_schedule())>);
      }
    };

    auto old_params = default_mainloop_params(
        meta.impl(_GemmV3{}), hparams, get_default_schedule(), cute::Int<epi_smem_size>{});

    constexpr int Stages = old_params.stages();
    using ClusterShape = decltype(old_params.cluster_shape());

    auto get_array_schedule = []() {
      if constexpr (gemm_v3_hparams.kernel_schedule() == _Cooperative{}) {
        return make_declval<cutlass::gemm::KernelPtrArrayCpAsyncWarpSpecializedCooperative>();
      } else if constexpr (gemm_v3_hparams.kernel_schedule() == _PingPong{}) {
        return make_declval<cutlass::gemm::KernelPtrArrayCpAsyncWarpSpecializedPingpong>();
      } else {
        static_assert(
            cutlass::detail::dependent_false<decltype(gemm_v3_hparams.kernel_schedule())>);
      }
    };
    using NewDispatchPolicy = cutlass::gemm::MainloopSm90ArrayCpAsyncGmmaWarpSpecialized<
        Stages,
        ClusterShape,
        decltype(get_array_schedule())>;
    auto params = old_params.dispatch_policy(TypeWrapper<NewDispatchPolicy>{})
                      .gmem_tiled_copy_b(TypeWrapper<SM90_TMA_LOAD_MULTICAST>{});

    using Mma = cutlass::gemm::collective::Sm90GemmArrayAgScatterMma<
        params.stages(),
        decltype(params.cluster_shape()),
        decltype(params.tile_shape()),
        decltype(params.kernel_schedule()),
        decltype(base_params.element_a()),
        decltype(base_params.stride_a()),
        decltype(base_params.element_b()),
        decltype(base_params.stride_b()),
        decltype(params.tiled_mma()),
        decltype(params.gmem_tiled_copy_a()),
        decltype(params.smem_layout_atom_a()),
        decltype(params.smem_copy_atom_a()),
        decltype(params.transform_a()),
        decltype(params.gmem_tiled_copy_b()),
        decltype(params.smem_layout_atom_b()),
        decltype(params.smem_copy_atom_b()),
        decltype(params.transform_b()),
        to_gemm_v3_meta(meta.impl_spec()).fast_accum()>;
    return make_declval<Mma>();
  }

  auto
  collective_epilogue() const {
    using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;
    using DefaultCollectiveEpilogue =
        identity_t<decltype(this->template default_collective_epilogue<EpilogueSchedule>())>;
    using StrideC = typename DefaultCollectiveEpilogue::StrideC;
    using StrideD = typename DefaultCollectiveEpilogue::StrideD;
    using ThreadEpilogueOp = typename DefaultCollectiveEpilogue::ThreadEpilogueOp;
    using Epilogue = cutlass::epilogue::collective::
        EpilogueScatterArray<StrideC, StrideD, ThreadEpilogueOp, EpilogueSchedule, ScatterD>;
    return make_declval<
        cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>>();
  }

  auto
  gemm_kernel() const {
    using CollectiveMma = decltype(this->collective_mma());
    using CollectiveEpilogue = decltype(this->collective_epilogue());
    using TileSchedulerTag = decltype(this->tile_scheduler());

    if constexpr (gemm_v3_hparams.kernel_schedule() == _Cooperative{}) {
      using GemmKernel = cutlass::gemm::kernel::Sm90AGScatterGemmArrayUniversalCooperative<
          typename Base::ProblemShape,
          CollectiveMma,
          CollectiveEpilogue,
          TileSchedulerTag,
          GatherA,
          AGFetcherA>;
      return make_declval<GemmKernel>();
    } else if constexpr (gemm_v3_hparams.kernel_schedule() == _PingPong{}) {
      using GemmKernel = cutlass::gemm::kernel::Sm90AGScatterGemmArrayUniversalPingpong<
          typename Base::ProblemShape,
          CollectiveMma,
          CollectiveEpilogue,
          TileSchedulerTag,
          GatherA,
          AGFetcherA>;
      return make_declval<GemmKernel>();
    } else {
      static_assert(cutlass::detail::dependent_false<decltype(gemm_v3_hparams.kernel_schedule())>);
    }
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmGroupedV3AGScatter_Device
    : public GemmGroupedV3BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmGroupedV3AGScatter_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using KernelBuilder = GemmGroupedV3AGScatter_Kernel<GemmMetaT, GemmHParamsT>;
  using Base =
      GemmGroupedV3BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmGroupedV3AGScatter_Device>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedV3AGScatter_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  using GatherA = typename KernelBuilder::GatherA;
  using ScatterD = typename KernelBuilder::ScatterD;
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using AGFetcherA = typename KernelBuilder::AGFetcherA;

  // Preparing args from host to device workspace
  std::size_t
  get_args_workspace_size(std::any const &unify_args) const override {
    const auto &args = std::any_cast<GemmGroupedAgScatterArguments>(unify_args);
    std::size_t workspace_size =
        static_cast<Base const *>(this)->get_args_workspace_size_impl(args);
    // gather_A
    workspace_size = make_align(workspace_size + sizeof(int32_t const *) * args.problem_count);
    // scatter_D
    workspace_size = make_align(workspace_size + sizeof(int32_t const *) * args.problem_count);
    // problem_schedule
    workspace_size = make_align(workspace_size + sizeof(ProblemSchedule) * args.problem_count);
    return workspace_size;
  }

  std::size_t
  get_barrier_workspace_size(std::any const &unify_args) const override {
    const auto &args = std::any_cast<GemmGroupedAgScatterArguments>(unify_args);
    return pad_to(args.dist_env.world_size * sizeof(int), 128);
    // auto [tile_M, tile_N, tile_K] = hparams.tile_shape();
    // std::size_t nflags = 2 * args.ntokens * cute::ceil_div(args.h, tile_K);
    // return pad_to(sizeof(int) * nflags, 128);
  }

  void
  initialize_args_workspace(
      std::any const &unify_args, void *args_workspace, void *stream) const override {
    const auto &args = std::any_cast<GemmGroupedAgScatterArguments>(unify_args);
    std::vector<uint8_t> host_workspace =
        static_cast<Base const &>(*this).get_host_workspace_buffer(args);
    std::size_t base_size = host_workspace.size();
    host_workspace.resize(this->get_args_workspace_size(unify_args));

    std::size_t workspace_offset = base_size;
    uint8_t *workspace_ptr = host_workspace.data();
    std::size_t sizeof_ptrs = sizeof(int32_t const *) * args.problem_count;
    // gather_A
    memcpy(workspace_ptr + workspace_offset, args.gather_A, sizeof_ptrs);
    workspace_offset = make_align(workspace_offset + sizeof_ptrs);
    // scatter_D
    memcpy(workspace_ptr + workspace_offset, args.scatter_D, sizeof_ptrs);
    workspace_offset = make_align(workspace_offset + sizeof_ptrs);
    // problem_schedule
    memcpy(
        workspace_ptr + workspace_offset,
        args.problem_schedule,
        sizeof(ProblemSchedule) * args.problem_count);
    workspace_offset = make_align(workspace_offset + sizeof_ptrs);
    this->initialize_args_workspace_with_buffer(host_workspace, args_workspace, stream);
  }

  auto
  parse_extra_args_from_args_workspace(
      GemmGroupedAgScatterArguments const &args, void *args_workspace) const {
    std::size_t workspace_offset =
        static_cast<Base const &>(*this).get_args_workspace_size_impl(args);
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(args_workspace);
    // gather_A
    auto ptr_gather_A = reinterpret_cast<int32_t const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(int32_t const *) * args.problem_count);
    // scatter_D
    auto ptr_scatter_D = reinterpret_cast<int32_t const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(int32_t const *) * args.problem_count);
    // problem_schedule
    auto ptr_problem_schedule =
        reinterpret_cast<ProblemSchedule const *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(ProblemSchedule) * args.problem_count);
    FLUX_CHECK(workspace_offset == this->get_args_workspace_size(args));
    return cute::make_tuple(ptr_gather_A, ptr_scatter_D, ptr_problem_schedule);
  }

  auto
  to_gemm_args_impl(GemmGroupedAgScatterArguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmKernel = typename Gemm::GemmKernel;
    using GemmArguments = typename Gemm::Arguments;
    using UnderlyingProblemShape = typename Base::ProblemShape::UnderlyingProblemShape;
    using EpilogueParams = typename Gemm::EpilogueOutputOp::Params;
    using ElementAccumulator = typename Base::ElementAccumulator;

    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and
    // wish to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id) -
        args.sm_margin;
    // problem_sizes on host can be None
    auto problem_sizes_host = static_cast<UnderlyingProblemShape *>(args.problem_sizes);

    auto
        [problem_sizes_device,
         ptr_A,
         ptr_Stride_A,
         ptr_B,
         ptr_Stride_B,
         ptr_C,
         ptr_Stride_C,
         ptr_D,
         ptr_Stride_D,
         ptr_alpha] = this->parse_common_gemm_args_from_workspace(args, args_workspace);

    // parameters for the epilogue
    EpilogueParams epi_params;
    if (args.ptr_alpha == nullptr) {
      epi_params = EpilogueParams(ElementAccumulator(args.alpha), ElementAccumulator(args.beta));
    } else {
      FLUX_CHECK(args.beta == 0.0f) << args.beta << "must be 0.0 if ptr_alpha is set";
      epi_params = EpilogueParams(ptr_alpha);
    }

    auto [ptr_gather_A, ptr_scatter_D, ptr_problem_schedule] =
        this->parse_extra_args_from_args_workspace(args, args_workspace);

    auto gather_A = GatherA(ptr_gather_A);
    auto ag_fetcher_A = AGFetcherA(
        args.dist_env,
        static_cast<ProblemSchedule const *>(ptr_problem_schedule),
        static_cast<int *>(args.barrier_ptr));

    using TileScheduler = typename Gemm::GemmKernel::TileScheduler;
    auto scheduler = typename TileScheduler::Arguments{};
    if constexpr (hparams.raster_order() == _RasterAlongM{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    } else if constexpr (hparams.raster_order() == _RasterAlongN{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    }

    auto gemm_args =
        GemmArguments{/*mode=*/cutlass::gemm::GemmUniversalMode::kGrouped,
                      {args.problem_count, problem_sizes_device, problem_sizes_host},
                      /*MMA*/
                      {ptr_A, ptr_Stride_A, ptr_B, ptr_Stride_B},
                      /*Epilogue*/
                      {epi_params, ptr_C, ptr_Stride_C, ptr_D, ptr_Stride_D, ptr_scatter_D},
                      hw_info,
                      /*scheduler=*/scheduler,
                      /*GatherA=*/gather_A,
                      /*AGFetcherA=*/ag_fetcher_A};

    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &unified_args, void *args_workspace) const {
    auto const &args = std::any_cast<GemmGroupedAgScatterArguments>(unified_args);
    return this->to_gemm_args_impl(args, args_workspace);
  }
};
}  // namespace flux
}  // namespace bytedance
