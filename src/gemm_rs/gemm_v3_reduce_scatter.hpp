//===- gemm_v3_reduce_scatter.hpp --------------------------------- C++ ---===//
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
#include <stdexcept>
#include <type_traits>
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cute/layout.hpp"
#include "flux/cuda/cutlass_v3_builder.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/args/gemm_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_v3_impl.hpp"

#include "gemm_rs/tile_scheduler/sm90_tile_scheduler_reduce_scatter.hpp"
#include "gemm_rs/epilogue_nvshmem_reduce_scatter.hpp"
#include "gemm_rs/epilogue_reduce_scatter.hpp"
#include "gemm_rs/epilogue_vectorized_reduce_scatter.hpp"
#include "gemm_rs/sm90_epilogue_evt.hpp"
#include "gemm_rs/sm90_reduce_scatter_utils.hpp"
#include "gemm_rs/sm90_gemm_tma_warpspecialized_cooperative_reduce_scatter.hpp"
#include "gemm_rs/sm90_gemm_tma_warpspecialized_pingpong_reduce_scatter.hpp"

namespace cutlass::gemm {
struct PersistentSchedulerReduceScatter {};
struct StreamKSchedulerReduceScatter {};
}  // namespace cutlass::gemm

namespace cutlass::gemm::kernel::detail {
template <class ArchTag, class TileShape, class ClusterShape>
struct TileSchedulerSelector<PersistentSchedulerReduceScatter, ArchTag, TileShape, ClusterShape> {
  using Scheduler = PersistentTileSchedulerSm90ReduceScatter;
};
}  // namespace cutlass::gemm::kernel::detail

namespace bytedance::flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmV3ReduceScatter_Kernel : public GemmV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static_assert(meta.comm_op() == _ReduceScatter{}, "requires _ReduceScatter{}");
  static constexpr auto rs_meta = to_reduce_scatter_meta(meta.comm_spec());

  auto
  tile_scheduler() const {
    if constexpr (meta.arch() != _Sm90{}) {
      return make_declval<cutlass::gemm::PersistentScheduler>();
    } else {
      if constexpr (hparams.gemm_kind() == _GemmStreamK{}) {
        return make_declval<cutlass::gemm::StreamKSchedulerReduceScatter>();
      } else {
        return make_declval<cutlass::gemm::PersistentSchedulerReduceScatter>();
      }
    }
  }

  auto
  sm80_collective_epilogue() const {
    auto dispatch_epilogue = [](auto... args) {
      if constexpr (rs_meta.comm_kind() == _IntraNode{}) {
        using Epilogue =
            cutlass::epilogue::collective::EpilogueReduceScatterVectorized<decltype(args)...>;
        return make_declval<Epilogue>();
      } else {
        using Epilogue =
            cutlass::epilogue::collective::EpilogueReduceScatterVectorizedNvshmemLocalReduce<
                decltype(args)...>;
        return make_declval<Epilogue>();
      }
    };

    cutlass_v3_builder::Sm80EpilogueParams params =
        cutlass_v3_builder::default_epilogue_params(meta, hparams);
    return dispatch_epilogue(
        params.stride_c(),
        params.stride_d(),
        params.thread_epilogue_op(),
        params.smem_layout(),
        params.copy_atom_r2s(),
        params.tiled_copy_s2r(),
        params.copy_atom_r2g());
  }

  auto
  sm90_collective_epilogue() const {
    cutlass_v3_builder::Sm90EpilogueParams params =
        cutlass_v3_builder::default_epilogue_params(meta, hparams);
    using OldDispatchPolicy = decltype(params.dispatch_policy());
    // reduce DispatchPolicy's stageD to 1
    using DispatchPolicy = cutlass::epilogue::Sm90TmaWarpSpecialized<
        /*StagesC=*/1,
        /*StagesD=*/1,
        OldDispatchPolicy::FragmentSize,
        OldDispatchPolicy::ReuseSmemC,
        OldDispatchPolicy::DelayTmaStore>;
    auto compose_fusion_callbacks = [this, &params]() {
      constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
      using namespace cutlass::epilogue::fusion;

      using ElementC = decltype(params.element_c());
      using ElementCNonVoid = decltype(params.element_c_unvoid());
      using ElementD = decltype(params.element_d());
      using ElementCompute = ElementD;
      using ElementAccumulator = decltype(params.element_accumulator());

      auto select_evt_d = [this, &params]() {
        if constexpr (this->is_s8_gemm) {
          return this->s8_gemm_evt_d(params);
        } else {
          using EVT_Compute0 = Sm90EVT<
              Sm90Compute<
                  cutlass::multiplies,
                  ElementD,
                  ElementCompute,
                  RoundStyle>,                          // alpha * acc
              Sm90ScalarBroadcast<ElementAccumulator>,  // alpha
              Sm90AccFetch                              // acc
              >;
          if constexpr (cute::is_void_v<ElementC>) {
            return make_declval<EVT_Compute0>();
          } else {
            using EVT_Compute1 = Sm90EVT<  // D
                Sm90Compute<
                    cutlass::multiply_add,
                    ElementD,
                    ElementCompute,
                    RoundStyle>,                          // beta * C + (alpha * acc)
                Sm90ScalarBroadcast<ElementAccumulator>,  // beta
                Sm90SrcFetch<ElementCNonVoid>,            // C
                EVT_Compute0>;
            return make_declval<EVT_Compute1>();
          }
        }
      };

      using EVT_D = decltype(select_evt_d());
      auto select_evt_final = []() {
        using AuxStoreType = Sm90AuxStoreReduceScatter<
            DispatchPolicy::StagesD,
            decltype(params.tile_shape()),
            decltype(params.epilogue_tile_mn()),
            ElementD,
            RoundStyle,
            decltype(params.stride_d()),
            decltype(params.smem_layout_atom_d()),
            decltype(params.copy_op_r2s()),
            rs_meta.comm_kind()>;
        return make_declval<Sm90EVT<AuxStoreType, EVT_D>>();
      };

      using CustomEVT = decltype(select_evt_final());

      using FusionCallbacks = typename cutlass::epilogue::collective::detail::CallbacksBuilder<
          DispatchPolicy,
          CustomEVT,
          decltype(params.tile_shape()),
          decltype(params.epilogue_tile_mn()),
          ElementAccumulator>::Callbacks;
      return make_declval<FusionCallbacks>();
    };
    using FusionCallbacks = decltype(compose_fusion_callbacks());
    auto new_params = params.dispatch_policy(TypeWrapper<DispatchPolicy>{})
                          .fusion_callbacks(TypeWrapper<FusionCallbacks>{});
    return cutlass_v3_builder::build_collective_epilogue(new_params);
  }

  auto
  collective_epilogue() const {
    if constexpr (meta.arch() == _Sm80{}) {
      return this->sm80_collective_epilogue();
    } else if constexpr (meta.arch() == _Sm90{}) {
      return this->sm90_collective_epilogue();
    } else {
      static_assert(cutlass::detail::dependent_false<decltype(meta.arch())>, "unsupported arch");
    }
  }

  auto
  gemm_kernel() const {
    using CollectiveEpilogue = identity_t<decltype(this->collective_epilogue())>;
    using CollectiveMma = decltype(this->default_collective_mma(
        cute::Int<sizeof(typename CollectiveEpilogue::SharedStorage)>{}));
    using TileScheduler = decltype(this->tile_scheduler());
    if constexpr (meta.arch() == _Sm80{}) {
      return make_declval<cutlass::gemm::kernel::GemmUniversal<
          typename Base::ProblemShape,
          CollectiveMma,
          CollectiveEpilogue,
          TileScheduler>>();
    } else if constexpr (meta.arch() == _Sm90{}) {
      using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;
      using SmemLayoutAtom = typename CollectiveEpilogue::SmemLayoutAtomD;
      using Element = decltype(to_cutlass_element(dt_conf.d()));
      using StrideMNL = typename CollectiveEpilogue::StrideD;
      using ScheduleType = decltype(to_gemm_v3_hparams(hparams.impl_spec()).kernel_schedule());

      constexpr auto get_dma_stages = []() {
        if constexpr (Base::is_s8_gemm) {
          // pingpong has smaller EpilogueTile than warp specilized
          // (64, 32) for 16bit output, (64, 64) for 8bit output
          // more dma stages are better than larger EpilogueTile
          if constexpr (to_gemm_v3_hparams(hparams.impl_spec()).kernel_schedule() == _PingPong{}) {
            constexpr int32_t mma_N = cute::get<1>(hparams.tile_shape());
            constexpr int32_t epi_N = cute::get<1>(EpilogueTile{});
            return mma_N / epi_N;
          } else {
            return 1;
          }
        } else {
          return 1;
        }
      };

      // need tuning?
      // Currently the dma stage is only optimized for PingPong and is calculated by dividing MMA_N
      // by Epi_N. The performance is close to optimal under some shape of s8 gemm. But the dma
      // stage of Cooperative is still one, which is not optimal and can be tuned. In addition, the
      // increase in the number of dma stages may affect the number of stages in mainloop due to
      // the use of shared memory. Better performance can be achieved through tuning(as one of gemm
      // config).
      constexpr int32_t StagesDma = get_dma_stages();
      using ReduceScatterDma = Sm90ReduceScatterDma<
          StagesDma,
          decltype(hparams.tile_shape()),
          EpilogueTile,
          SmemLayoutAtom,
          Element,
          StrideMNL,
          rs_meta.comm_kind()(),
          rs_meta.fuse_reduction()()>;
      return make_declval<cutlass::gemm::kernel::Sm90RSGemmUniversal<
          typename Base::ProblemShape,
          CollectiveMma,
          CollectiveEpilogue,
          TileScheduler,
          ReduceScatterDma>>();
    } else {
      static_assert(cutlass::detail::dependent_false<decltype(meta.arch())>, "unsupported arch");
    }
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV3ReduceScatter_Device
    : public GemmV3BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmV3ReduceScatter_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using Base = GemmV3BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmV3ReduceScatter_Device>;
  using KernelBuilder = GemmV3ReduceScatter_Kernel<GemmMetaT, GemmHParamsT>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3ReduceScatter_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr auto rs_meta = to_reduce_scatter_meta(meta.comm_spec());

  auto
  to_s8_gemm_args_impl(GemmReduceScatterArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmKernel = GemmKernelT;
    using GemmArguments = typename Gemm::Arguments;

    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));
    using ElementCNonVoid = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;
    using ElementScale = float;
    static constexpr bool has_bias = not cute::is_void_v<ElementC>;

    auto ptr_A = static_cast<ElementA const *>(args.input);
    auto ptr_B = static_cast<ElementB const *>(args.weight);
    auto ptr_C = static_cast<ElementC const *>(args.bias);
    auto ptr_scatter_D = reinterpret_cast<ElementD **>(args.output_scatter_ptrs);
    auto ptr_D = static_cast<ElementD const *>(ptr_scatter_D[args.rank]);
    auto ptr_barrier = reinterpret_cast<int **>(args.barrier_ptrs);
    auto ptr_scale_A = static_cast<ElementScale const *>(args.scaleA);
    auto ptr_scale_B = static_cast<ElementScale const *>(args.scaleB);
    auto beta = static_cast<ElementCNonVoid>(args.beta);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, 1));
    // bias shape: [1, n], stride: [0, 1]
    cute::tuple<int64_t, cute::Int<1>, int64_t> stride_C = {
        cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}};
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, 1));
    using Epilogue = typename GemmKernel::CollectiveEpilogue;
    typename GemmKernel::EpilogueArguments epilogue{
        {}, has_bias ? ptr_C : nullptr, stride_C, ptr_D, stride_D};

    if constexpr (has_bias) {
      epilogue.thread = {
          // unary op : Aux store D
          {
              // ternary op : beta * C + (scale_a * scale_b * Acc)
              {beta},   // beta
              {ptr_C},  // bias
              {
                  // binary op : (scale_a * scale_b) * Acc
                  {
                      // scale_a * scale_b
                      {ptr_scale_A},                   // scale_a
                      {ptr_scale_B},                   // scale_b
                      {}                               // binary args : multiplies
                  },                                   // end binary op
                  {},                                  // Acc
                  {}                                   // binary args : multiplies
              },                                       // end binary op
              {}                                       // ternary args : multiply_add
          },                                           // end ternary op
          {.barrier_ptr_aux = ptr_barrier[args.rank]}  // unary args : aux store D
      };  // end unary op
    } else {
      epilogue.thread = {
          // unary op : Aux store D
          {
              // (scale_a * scale_b) * Acc
              {
                  {ptr_scale_A},                       // scale_a
                  {ptr_scale_B},                       // scale_b
                  {}                                   // binary args : multiplies
              },                                       // end binary op
              {},                                      // Acc
              {}                                       // binary args : multiplies
          },                                           // end binary op
          {.barrier_ptr_aux = ptr_barrier[args.rank]}  // unary args : aux store D
      };  // end unary op
    }

    using TileScheduler = typename GemmKernel::TileScheduler;
    using TileSchedulerTag = decltype(KernelBuilder().tile_scheduler());
    auto scheduler = typename TileScheduler::Arguments{};

    auto [m_tile_size, n_tile_size, _] = hparams.tile_shape();
    int min_tile_count =
        min(cute::ceil_div(args.m, m_tile_size), cute::ceil_div(args.n, n_tile_size));
    if constexpr (
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::PersistentSchedulerReduceScatter> or
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::StreamKSchedulerReduceScatter>) {
      int local_world_size = args.world_size / args.nnodes;
      int m_per_local_rank = args.m / local_world_size;
      int m_tiles = cute::ceil_div(m_per_local_rank, m_tile_size);

      if constexpr (hparams.raster_order() == _RasterAlongN{}) {
        scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
      } else if constexpr (hparams.raster_order() == _RasterAlongM{}) {
        scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
      }
      // not set max_swizzle_size for better performance
      // if constexpr (rs_meta.comm_kind() == _AcrossNode{}) {
      //   if constexpr (hparams.gemm_kind() != _GemmStreamK{}) {
      //     scheduler.max_swizzle_size = cute::min(m_tiles, min_tile_count);
      //   }
      // } else if constexpr (hparams.gemm_kind() != _GemmStreamK{}) {
      //   scheduler.max_swizzle_size = min_tile_count;
      // }
      scheduler.rank = args.rank;
      scheduler.world_size = args.world_size;
      scheduler.nnodes = args.nnodes;
    }

    auto gemm_args = GemmArguments{};
    gemm_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    gemm_args.problem_shape = {args.m, args.n, args.k};
    gemm_args.mainloop = {ptr_A, stride_A, ptr_B, stride_B};
    gemm_args.epilogue = cute::move(epilogue);
    gemm_args.scheduler = cute::move(scheduler);
    if constexpr (meta.arch() == _Sm90{}) {
      // rs_dma args
      gemm_args.rs_dma = typename GemmKernel::ReduceScatterDmaArguments{
          .output_scatter_ptrs = ptr_scatter_D,
          .stride = stride_D,
          .rank = args.rank,
          .world_size = args.world_size,
          .nnodes = args.nnodes,
          .local_reduce_buffer = args.reduce_buffer_ptrs[args.rank],
          .barrier_ptrs = ptr_barrier};
    }
    return gemm_args;
  }

  auto
  to_gemm_args_impl(GemmReduceScatterArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    using GemmKernel = GemmKernelT;
    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));
    constexpr bool has_bias = not cute::is_void_v<ElementC>;
    using ElementCNonVoid = cute::conditional_t<has_bias, ElementC, ElementD>;

    auto ptr_A = static_cast<ElementA const *>(args.input);
    auto ptr_B = static_cast<ElementB const *>(args.weight);
    auto ptr_C = static_cast<ElementCNonVoid const *>(args.bias);
    auto ptr_scatter_D = reinterpret_cast<ElementD **>(args.output_scatter_ptrs);
    auto ptr_D = static_cast<ElementD const *>(ptr_scatter_D[args.rank]);
    auto ptr_barrier = reinterpret_cast<int **>(args.barrier_ptrs);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, cute::make_shape(args.m, args.n, 1));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, 1));

    auto get_epilogue_args = [&]() {
      if constexpr (meta.arch() == _Sm90{}) {
        typename GemmKernel::EpilogueArguments epilogue{
            {}, has_bias ? ptr_C : nullptr, stride_C, ptr_D, stride_D};
        if constexpr (has_bias) {
          epilogue.thread = {
              // unary op: aux store D
              {
                  // ternary op : beta * C + (alpha * acc)
                  {{args.beta}},  // leaf op+args : beta
                  {},             // leaf op+args : C
                  {
                      // binary op : alpha * acc
                      {{args.alpha}},  // leaf op+args : alpha
                      {},              // leaf op+args : acc
                      {}               // binary args : multiplies
                  },                   // end binary op
                  {}                   // ternary args : multiply_add
              },
              {.barrier_ptr_aux = ptr_barrier[args.rank]}  // unary args: aux store D
          };
        } else {
          epilogue.thread = {
              // unary op: aux store D
              {
                  // binary op : alpha * acc
                  {{args.alpha}},  // leaf op+args : alpha
                  {},              // leaf op+args : acc
                  {}               // binary args : multiplies
              },
              {.barrier_ptr_aux = ptr_barrier[args.rank]}  // unary args: aux store D
          };
        }
        return epilogue;
      } else {
        typename GemmKernel::EpilogueArguments epilogue{
            args.rank,
            args.world_size,
            {args.alpha, args.beta},
            ptr_C,
            stride_C,
            ptr_scatter_D,
            stride_D,
            ptr_barrier};
        return epilogue;
      }
    };
    auto epilogue = get_epilogue_args();

    using TileScheduler = typename GemmKernel::TileScheduler;
    using TileSchedulerTag = decltype(KernelBuilder().tile_scheduler());
    auto scheduler = typename TileScheduler::Arguments{};

    auto [m_tile_size, n_tile_size, _] = hparams.tile_shape();
    int min_tile_count =
        min(cute::ceil_div(args.m, m_tile_size), cute::ceil_div(args.n, n_tile_size));
    if constexpr (
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::PersistentSchedulerReduceScatter> or
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::StreamKSchedulerReduceScatter>) {
      int local_world_size = args.world_size / args.nnodes;
      int m_per_local_rank = args.m / local_world_size;
      int m_tiles = cute::ceil_div(m_per_local_rank, m_tile_size);

      if constexpr (rs_meta.comm_kind() == _AcrossNode{}) {
        if constexpr (hparams.raster_order() == _RasterAlongN{}) {
          scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
        } else {
          scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
        }
        if constexpr (hparams.gemm_kind() != _GemmStreamK{}) {
          scheduler.max_swizzle_size = cute::min(m_tiles, min_tile_count);
        }
      } else if constexpr (hparams.gemm_kind() != _GemmStreamK{}) {
        if constexpr (hparams.raster_order() == _RasterAlongN{}) {
          scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
        } else {
          scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
        }
        scheduler.max_swizzle_size = min_tile_count;
      }
      scheduler.rank = args.rank;
      scheduler.world_size = args.world_size;
      scheduler.nnodes = args.nnodes;
    }

    auto gemm_args = GemmArguments{};
    gemm_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    gemm_args.problem_shape = {args.m, args.n, args.k};
    gemm_args.mainloop = {ptr_A, stride_A, ptr_B, stride_B};
    gemm_args.epilogue = cute::move(epilogue);
    gemm_args.scheduler = cute::move(scheduler);
    if constexpr (meta.arch() == _Sm90{}) {
      // rs_dma args
      gemm_args.rs_dma = typename GemmKernel::ReduceScatterDmaArguments{
          .output_scatter_ptrs = ptr_scatter_D,
          .stride = stride_D,
          .rank = args.rank,
          .world_size = args.world_size,
          .nnodes = args.nnodes,
          .local_reduce_buffer = args.reduce_buffer_ptrs[args.rank],
          .barrier_ptrs = ptr_barrier};
    }
    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &args) const {
    if constexpr (this->is_s8_gemm) {
      return to_s8_gemm_args_impl(std::any_cast<GemmReduceScatterArguments>(args));
    } else {
      return to_gemm_args_impl(std::any_cast<GemmReduceScatterArguments>(args));
    }
  }

  std::size_t
  get_barrier_workspace_size(std::any const &var_args) const override {
    auto align_buffer = [](size_t size) { return (size + 127) / 128 * 128; };
    const auto &args = std::any_cast<GemmReduceScatterArguments>(var_args);
    if constexpr (not(rs_meta.comm_kind() == _AcrossNode{} and meta.arch() == _Sm80{})) {
      auto [tile_m, tile_n, tile_k] = hparams.tile_shape();
      // for each tile, one flag for finished writing to local
      // another for finished fetching&reducing from other rank
      std::size_t nflags = 2 * cute::ceil_div(args.m, tile_m) * cute::ceil_div(args.n, tile_n);
      return align_buffer(nflags * sizeof(int));
    } else {
      return 0;
    }
  }
};
}  // namespace bytedance::flux
