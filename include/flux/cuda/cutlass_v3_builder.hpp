//===- cutlass_v3_builder.hpp ------------------------------------- C++ ---===//
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

/// adapt cutlass sm90_builder.inc and sm90_gemm_builder.inc to our format
#pragma once
#include <type_traits>
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_operator_base_default_impl.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cute/numeric/integral_constant.hpp"
#include "cute/layout.hpp"
#include "cute/int_tuple.hpp"
#include "cute/config.hpp"
#include "cute/util/type_traits.hpp"
#include <cute/tensor.hpp>
#include "cute/algorithm/functional.hpp"
#include "cute/numeric/int.hpp"

#include "cute/arch/mma_sm80.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/atom/mma_atom.hpp"
#include <cute/arch/copy.hpp>
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cutlass/functional.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/helper_macros.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace bytedance::flux {
namespace cutlass_v3_builder {
using namespace cute;

namespace detail {
// for a class template which is an empty type (no runtime storage),
// this mixin provides functions to pack its template parameters
// into a cute::tuple and unpack from cute::tuple
template <template <class...> class DerivedTpl>
struct ToTypesTupleMixin {
 public:
  template <class... Ts>
  static constexpr auto
  to_tuple(DerivedTpl<Ts...>) {
    return cute::tuple<TypeWrapper<Ts>...>{};
  }

  template <class... Ts>
  static constexpr auto
  from_tuple(cute::tuple<TypeWrapper<Ts>...>) {
    return DerivedTpl<Ts...>();
  }
};
}  // namespace detail

/////////////////////////////////////////////////////
// Params types
// collect customizable types to build CollectiveMMa
// and CollectiveEpilogue
/////////////////////////////////////////////////////

// This Macro define field() and update_field functions for
// Params types, field() usually used with decltype, and
// update_field replace a field with new type and returns
// a new Params object
#if defined(DEFINE_PARAMS_FIELD)
static_assert(false, "redefinition of DEFINE_PARAMS_FIELD");
#endif
#define DEFINE_PARAMS_FIELD(FIELD, TYPE, INDEX)                              \
  auto FIELD() const { return make_declval<TYPE>(); }                        \
  template <class NewType>                                                   \
  constexpr auto FIELD(TypeWrapper<NewType> type_val) const {                \
    return from_tuple(tuple_replace_item<INDEX>(to_tuple(*this), type_val)); \
  }

template <
    class ElementA,
    class GmemLayoutA,
    class ElementB,
    class GmemLayoutB,
    class ElementAccumulator>
struct BaseMainloopParams : detail::ToTypesTupleMixin<BaseMainloopParams> {
  DEFINE_PARAMS_FIELD(element_a, ElementA, 0)
  DEFINE_PARAMS_FIELD(gmem_layout_a, GmemLayoutA, 1)
  DEFINE_PARAMS_FIELD(element_b, ElementB, 2)
  DEFINE_PARAMS_FIELD(gmem_layout_b, GmemLayoutB, 3)
  DEFINE_PARAMS_FIELD(element_accumulator, ElementAccumulator, 4)

  auto
  stride_a() const {
    return make_declval<cutlass::gemm::TagToStrideA_t<GmemLayoutA>>();
  }
  auto
  stride_b() const {
    return make_declval<cutlass::gemm::TagToStrideB_t<GmemLayoutB>>();
  }
  constexpr int
  alignment_a() const {
    return 128 / cutlass::sizeof_bits_v<ElementA>;
  }
  constexpr int
  alignment_b() const {
    return 128 / cutlass::sizeof_bits_v<ElementB>;
  }
};

namespace detail {
template <class T>
struct is_base_mainloop_params : cute::false_type {};
template <class... Ts>
struct is_base_mainloop_params<BaseMainloopParams<Ts...>> : cute::true_type {};
}  // namespace detail

template <
    class BaseParams,
    class DispatchPolicy,
    class TileShape,
    class TiledMma,
    class GmemTiledCopyA,
    class SmemLayoutAtomA,
    class SmemCopyAtomA,
    class TransformA,
    class GmemTiledCopyB,
    class SmemLayoutAtomB,
    class SmemCopyAtomB,
    class TransformB>
struct MainloopParams : BaseParams, detail::ToTypesTupleMixin<MainloopParams> {
  static_assert(
      detail::is_base_mainloop_params<BaseParams>::value,
      "requires is_base_mainloop_params<BaseParams>");
  using detail::ToTypesTupleMixin<MainloopParams>::from_tuple;
  using detail::ToTypesTupleMixin<MainloopParams>::to_tuple;
  DEFINE_PARAMS_FIELD(base_params, BaseParams, 0)
  DEFINE_PARAMS_FIELD(dispatch_policy, DispatchPolicy, 1)
  DEFINE_PARAMS_FIELD(tile_shape, TileShape, 2)
  DEFINE_PARAMS_FIELD(tiled_mma, TiledMma, 3)
  DEFINE_PARAMS_FIELD(gmem_tiled_copy_a, GmemTiledCopyA, 4)
  DEFINE_PARAMS_FIELD(smem_layout_atom_a, SmemLayoutAtomA, 5)
  DEFINE_PARAMS_FIELD(smem_copy_atom_a, SmemCopyAtomA, 6)
  DEFINE_PARAMS_FIELD(transform_a, TransformA, 7)
  DEFINE_PARAMS_FIELD(gmem_tiled_copy_b, GmemTiledCopyB, 8)
  DEFINE_PARAMS_FIELD(smem_layout_atom_b, SmemLayoutAtomB, 9)
  DEFINE_PARAMS_FIELD(smem_copy_atom_b, SmemCopyAtomB, 10)
  DEFINE_PARAMS_FIELD(transform_b, TransformB, 11)
  constexpr int
  stages() const {
    return DispatchPolicy::Stages;
  }
  constexpr auto
  cluster_shape() const {
    return make_declval<typename DispatchPolicy::ClusterShape>();
  }
  constexpr auto
  kernel_schedule() const {
    return make_declval<typename DispatchPolicy::Schedule>();
  }
};

template <
    class ElementC,
    class GmemLayoutC,
    class ElementD,
    class GmemLayoutD,
    class ElementAccumulator>
struct BaseEpilogueParams : detail::ToTypesTupleMixin<BaseEpilogueParams> {
  DEFINE_PARAMS_FIELD(element_c, ElementC, 0)
  DEFINE_PARAMS_FIELD(gmem_layout_c, GmemLayoutC, 1)
  DEFINE_PARAMS_FIELD(element_d, ElementD, 2)
  DEFINE_PARAMS_FIELD(gmem_layout_d, GmemLayoutD, 3)
  DEFINE_PARAMS_FIELD(element_accumulator, ElementAccumulator, 4)
  auto
  element_c_unvoid() const {
    return make_declval<cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>>();
  }
  auto
  stride_c() const {
    return make_declval<cutlass::gemm::TagToStrideC_t<GmemLayoutC>>();
  }
  auto
  stride_d() const {
    return make_declval<cutlass::gemm::TagToStrideC_t<GmemLayoutD>>();
  }
  constexpr int
  alignment_c() const {
    return 128 / cutlass::sizeof_bits_v<decltype(this->element_c_unvoid())>;
  }
  constexpr int
  alignment_d() const {
    return 128 / cutlass::sizeof_bits_v<ElementD>;
  }
  constexpr bool
  has_bias() const {
    return not cute::is_void_v<ElementC>;
  }
};

namespace detail {
template <class T>
struct is_base_epilogue_params : cute::false_type {};
template <class... Ts>
struct is_base_epilogue_params<BaseEpilogueParams<Ts...>> : cute::true_type {};
}  // namespace detail

template <
    class BaseParams,
    class ThreadEpilogueOp,
    class SmemLayout,
    class CopyAtomR2S,
    class TiledCopyS2R,
    class CopyAtomR2G>
struct Sm80EpilogueParams : BaseParams, detail::ToTypesTupleMixin<Sm80EpilogueParams> {
  static_assert(
      detail::is_base_epilogue_params<BaseParams>::value,
      "requires is_base_epilogue_params<BaseParams>");
  using detail::ToTypesTupleMixin<Sm80EpilogueParams>::from_tuple;
  using detail::ToTypesTupleMixin<Sm80EpilogueParams>::to_tuple;
  DEFINE_PARAMS_FIELD(base_params, BaseParams, 0)
  DEFINE_PARAMS_FIELD(thread_epilogue_op, ThreadEpilogueOp, 1)
  DEFINE_PARAMS_FIELD(smem_layout, SmemLayout, 2)
  DEFINE_PARAMS_FIELD(copy_atom_r2s, CopyAtomR2S, 3)
  DEFINE_PARAMS_FIELD(tiled_copy_s2r, TiledCopyS2R, 4)
  DEFINE_PARAMS_FIELD(copy_atom_r2g, CopyAtomR2G, 5)
};

template <
    class BaseParams,
    class DispatchPolicy,
    class TileShape,
    class EpilogueTileMN,
    class FusionCallbacks,
    class CopyOpG2S,
    class SmemLayoutAtomC,
    class CopyOpS2R,
    class CopyOpS2G,
    class SmemLayoutAtomD,
    class CopyOpR2S>
struct Sm90EpilogueParams : BaseParams, detail::ToTypesTupleMixin<Sm90EpilogueParams> {
  static_assert(
      detail::is_base_epilogue_params<BaseParams>::value,
      "requires is_base_epilogue_params<BaseParams>");
  using detail::ToTypesTupleMixin<Sm90EpilogueParams>::from_tuple;
  using detail::ToTypesTupleMixin<Sm90EpilogueParams>::to_tuple;
  DEFINE_PARAMS_FIELD(base_params, BaseParams, 0)
  DEFINE_PARAMS_FIELD(dispatch_policy, DispatchPolicy, 1)
  DEFINE_PARAMS_FIELD(tile_shape, TileShape, 2)
  DEFINE_PARAMS_FIELD(epilogue_tile_mn, EpilogueTileMN, 3)
  DEFINE_PARAMS_FIELD(fusion_callbacks, FusionCallbacks, 4)
  DEFINE_PARAMS_FIELD(copy_op_g2s, CopyOpG2S, 5)
  DEFINE_PARAMS_FIELD(smem_layout_atom_c, SmemLayoutAtomC, 6)
  DEFINE_PARAMS_FIELD(copy_op_s2r, CopyOpS2R, 7)
  DEFINE_PARAMS_FIELD(copy_op_s2g, CopyOpS2G, 8)
  DEFINE_PARAMS_FIELD(smem_layout_atom_d, SmemLayoutAtomD, 9)
  DEFINE_PARAMS_FIELD(copy_op_r2s, CopyOpR2S, 10)
};

#undef DEFINE_PARAMS_FIELD

/////////////////////////////////////////////////////
// Functions computing default Params given
// GemmMeta and GemmHParams. By modifying fields of
// the default Params, we can achieve customization
/////////////////////////////////////////////////////

template <class... Ts, class... Us>
auto
base_mainloop_params(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using GmemLayoutA = decltype(to_cutlass_layout_a(meta.gemm_layout()));
  using GmemLayoutB = decltype(to_cutlass_layout_b(meta.gemm_layout()));
  using ElementAccumulator = decltype(to_cutlass_element(dt_conf.acc()));

  return BaseMainloopParams<
      ElementA,
      cute::conditional_t<is_grouped_gemm_impl(meta.impl()), GmemLayoutA *, GmemLayoutA>,
      ElementB,
      cute::conditional_t<is_grouped_gemm_impl(meta.impl()), GmemLayoutB *, GmemLayoutB>,
      ElementAccumulator>{};
}

namespace detail {

template <class... Ts, class... Us>
constexpr auto
select_mma_atom(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  static_assert(
      meta.arch() == _Sm80{} || meta.arch() == _Sm89{}, "only select mma_atom for sm80 or sm89");
  static_assert(
      meta.gemm_layout() == _RRR{} or meta.gemm_layout() == _RCR{}, "only support TN|TT");

  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static_assert(dt_conf.acc() == _FP32{}, "Expect accumulator to be float");
  static_assert(dt_conf.a() == dt_conf.b(), "requires dtype a == b");

  if constexpr (dt_conf.a() == _FP16{}) {
    return make_declval<MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>();
  } else if constexpr (dt_conf.a() == _BF16{}) {
    return make_declval<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>();
  } else {
    static_assert(cutlass::detail::dependent_false<decltype(dt_conf.a())>, "unsupported dtype");
  }
}

template <class... Ts, class... Us>
auto
default_sm80_mainloop_params(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  BaseMainloopParams base_params = base_mainloop_params(meta, hparams);

  using MmaAtom = decltype(select_mma_atom(meta, hparams));
  using TiledMma = TiledMMA<MmaAtom, Layout<Shape<_4, _2, _1>>, Tile<_64, _64, _16>>;

  using TileShape = decltype(hparams.tile_shape());
  static_assert(get<2>(TileShape{}) == 32, "require K-dim of TileShape to be 32");

  using ElementA = decltype(base_params.element_a());
  using ElementB = decltype(base_params.element_b());
  using GmemLayoutA = decltype(base_params.gmem_layout_a());
  using SmemLayoutAtomA =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
  using GmemTiledCopyA = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementA>{},
      Layout<Shape<_64, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _8>>{}));
  using TransformA = cute::identity;

  using GmemLayoutB = decltype(base_params.gmem_layout_b());
  using SmemLayoutAtomB =
      decltype(composition(Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;
  using GmemTiledCopyB = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementB>{},
      Layout<Shape<_64, _4>, Stride<_4, _1>>{},
      Layout<Shape<_1, _8>>{}));
  using TransformB = cute::identity;

  using DispatchPolicy =
      cutlass::gemm::MainloopSm80CpAsyncUnpredicated</*Stages_=*/hparams.mainloop_stage()>;

  return MainloopParams<
      decltype(base_params),
      DispatchPolicy,
      TileShape,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      TransformA,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      TransformB>{};
}

template <class... Ts, class... Us>
auto
default_kernel_schedule(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  namespace builder = cutlass::gemm::collective;
  constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  constexpr auto v3_meta = to_gemm_v3_meta(meta.impl_spec());
  constexpr auto v3_hparams = to_gemm_v3_hparams(hparams.impl_spec());

  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  constexpr bool is_input_fp8 = builder::detail::is_input_fp8<ElementA, ElementB>();
  if constexpr (is_grouped_gemm_impl(meta.impl())) {
    return make_declval<cute::conditional_t<
        is_input_fp8,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>>();
  } else {
    return make_declval<cute::conditional_t<
        is_input_fp8 and v3_meta.fast_accum(),
        cute::conditional_t<
            v3_hparams.kernel_schedule() == _PingPong{},
            cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
            cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>,
        cute::conditional_t<
            v3_hparams.kernel_schedule() == _PingPong{},
            cutlass::gemm::KernelTmaWarpSpecializedPingpong,
            cutlass::gemm::KernelTmaWarpSpecializedCooperative>>>();
  }
}
}  // namespace detail

template <
    class GemmMetaT,
    class GemmHParamsT,
    class KernelSchedule_ = void,
    int EpiSmemSize = 0,
    __CUTE_REQUIRES(is_gemm_meta_v<GemmMetaT> and is_gemm_hparams_v<GemmHParamsT>)>
auto
default_mainloop_params(
    GemmMetaT meta,
    GemmHParamsT hparams,
    TypeWrapper<KernelSchedule_> kernel_schedule = TypeWrapper<KernelSchedule_>{},
    cute::Int<EpiSmemSize> epi_smem_size = _0{}) {
  if constexpr (meta.arch() == _Sm80{}) {
    return detail::default_sm80_mainloop_params(meta, hparams);
  } else {
    auto dt_conf = make_gemm_dtype_config(meta.dtype());
    auto base_params = base_mainloop_params(meta, hparams);
    auto v3_params = to_gemm_v3_hparams(hparams.impl_spec());

    using KernelSchedule = cute::conditional_t<
        cute::is_void_v<KernelSchedule_>,
        decltype(detail::default_kernel_schedule(meta, hparams)),
        KernelSchedule_>;

    using StageCountType = cute::conditional_t<
        hparams.mainloop_stage() == 0,
        cutlass::gemm::collective::StageCountAutoCarveout<EpiSmemSize>,
        cutlass::gemm::collective::StageCount<hparams.mainloop_stage()>>;

    using Mma = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        decltype(base_params.element_a()),
        decltype(base_params.gmem_layout_a()),
        base_params.alignment_a(),
        decltype(base_params.element_b()),
        decltype(base_params.gmem_layout_b()),
        base_params.alignment_b(),
        decltype(base_params.element_accumulator()),
        decltype(hparams.tile_shape()),
        decltype(v3_params.cluster_shape()),
        StageCountType,
        KernelSchedule>::CollectiveOp;

    return MainloopParams<
        decltype(base_params),
        typename Mma::DispatchPolicy,
        typename Mma::TileShape,
        typename Mma::TiledMma,
        typename Mma::GmemTiledCopyA,
        typename Mma::SmemLayoutAtomA,
        typename Mma::SmemCopyAtomA,
        typename Mma::TransformA,
        typename Mma::GmemTiledCopyB,
        typename Mma::SmemLayoutAtomB,
        typename Mma::SmemCopyAtomB,
        typename Mma::TransformB>();
  }
}

template <class... Ts, class... Us>
auto
base_epilogue_params(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  using ElementC = decltype(to_cutlass_element(dt_conf.c()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));
  using GmemLayoutC = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using GmemLayoutD = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using ElementAccumulator = decltype(to_cutlass_element(dt_conf.acc()));
  return BaseEpilogueParams<ElementC, GmemLayoutC, ElementD, GmemLayoutD, ElementAccumulator>{};
}

namespace detail {
template <class... Ts, class... Us>
auto
default_sm80_epilogue_params(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  BaseEpilogueParams base_params = base_epilogue_params(meta, hparams);
  using TileShapeS2R = Shape<_16, _128>;
  using SmemLayoutAtom =
      decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, _8>, Stride<_8, _1>>{}));
  using SwizzledSmemLayout = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_256, _128>{}));
  using PreSwizzleLayout = Layout<Shape<_256, _128>, Stride<_128, _1>>;
  auto select_smem_layout = []() {
    auto meta = GemmMeta<Ts...>{};
    if constexpr (meta.comm_op() == _ReduceScatter{}) {
      if constexpr (to_reduce_scatter_meta(meta.comm_spec()).comm_kind() == _AcrossNode{}) {
        return make_declval<PreSwizzleLayout>();
      } else {
        return make_declval<SwizzledSmemLayout>();
      }
    } else {
      return make_declval<SwizzledSmemLayout>();
    }
  };
  using SmemLayout = decltype(select_smem_layout());
  using ElementD = decltype(base_params.element_d());
  using ElementAccumulator = decltype(base_params.element_accumulator());

  using CopyAtomR2S = Copy_Atom<DefaultCopy, ElementAccumulator>;
  using TiledCopyS2R = TiledCopy<
      Copy_Atom<DefaultCopy, ElementAccumulator>,
      Layout<Shape<_32, _8>, Stride<_8, _1>>,
      TileShapeS2R>;
  using CopyAtomR2G = Copy_Atom<DefaultCopy, ElementD>;
  using ThreadEpilogueOp = cutlass::epilogue::thread::
      LinearCombination<ElementD, 8, ElementAccumulator, ElementAccumulator>;
  return Sm80EpilogueParams<
      decltype(base_params),
      ThreadEpilogueOp,
      SmemLayout,
      CopyAtomR2S,
      TiledCopyS2R,
      CopyAtomR2G>{};
}

template <class... Ts, class... Us>
auto
default_epilogue_schedule(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  constexpr auto v3_hparams = to_gemm_v3_hparams(hparams.impl_spec());
  if constexpr (v3_hparams.kernel_schedule() == _PingPong{}) {
    return make_declval<cutlass::epilogue::TmaWarpSpecialized>();
  } else {
    return make_declval<cutlass::epilogue::TmaWarpSpecializedCooperative>();
  }
}

template <class... Ts, class... Us>
auto
default_sm90_epilogue_params(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  using namespace cutlass::epilogue;
  BaseEpilogueParams base_params = base_epilogue_params(meta, hparams);
  using ElementC = decltype(base_params.element_c());
  using ElementD = decltype(base_params.element_d());
  using ElementCUnVoid = decltype(base_params.element_c_unvoid());
  using ElementAccumulator = decltype(base_params.element_accumulator());

  auto select_evt_d = []() {
    constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using namespace cutlass::epilogue::fusion;

    using ElementCompute = ElementD;
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
              RoundStyle>,  // beta * C + (alpha * acc)
          cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAccumulator>,  // beta
          cutlass::epilogue::fusion::Sm90SrcFetch<ElementCUnVoid>,             // C
          EVT_Compute0>;
      return make_declval<EVT_Compute1>();
    }
  };

  using EVT_D = decltype(select_evt_d());
  using EpilogueSchedule = decltype(default_epilogue_schedule(meta, hparams));

  auto v3_hparams = to_gemm_v3_hparams(hparams.impl_spec());
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      decltype(hparams.tile_shape()),
      decltype(v3_hparams.cluster_shape()),
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      /*ElementCompute=*/ElementD,
      decltype(base_params.element_c()),
      decltype(base_params.gmem_layout_c()),
      base_params.alignment_c(),
      decltype(base_params.element_d()),
      decltype(base_params.gmem_layout_d()),
      base_params.alignment_d(),
      EpilogueSchedule,
      EVT_D>::CollectiveOp;

  return Sm90EpilogueParams<
      decltype(base_params),
      typename CollectiveEpilogue::DispatchPolicy,
      decltype(hparams.tile_shape()),
      typename CollectiveEpilogue::EpilogueTile,
      typename CollectiveEpilogue::FusionCallbacks,
      typename CollectiveEpilogue::CopyOpG2S,
      typename CollectiveEpilogue::SmemLayoutAtomC,
      typename CollectiveEpilogue::CopyOpS2R,
      typename CollectiveEpilogue::CopyOpS2G,
      typename CollectiveEpilogue::SmemLayoutAtomD,
      typename CollectiveEpilogue::CopyOpR2S>();
}
}  // namespace detail

template <class... Ts, class... Us>
auto
default_epilogue_params(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  if constexpr (meta.arch() == _Sm80{}) {
    return detail::default_sm80_epilogue_params(meta, hparams);
  } else {
    return detail::default_sm90_epilogue_params(meta, hparams);
  }
}

/////////////////////////////////////////////////////
// Helper functions.
// converting Params to CollectivaMma and
// CollectiveEpilogue
/////////////////////////////////////////////////////
template <class... Ts>
auto
build_collective_mainloop(MainloopParams<Ts...> params) {
  using Mma = cutlass::gemm::collective::CollectiveMma<
      decltype(params.dispatch_policy()),
      decltype(params.tile_shape()),
      decltype(params.element_a()),
      decltype(params.stride_a()),
      decltype(params.element_b()),
      decltype(params.stride_b()),
      decltype(params.tiled_mma()),
      decltype(params.gmem_tiled_copy_a()),
      decltype(params.smem_layout_atom_a()),
      decltype(params.smem_copy_atom_a()),
      decltype(params.transform_a()),
      decltype(params.gmem_tiled_copy_b()),
      decltype(params.smem_layout_atom_b()),
      decltype(params.smem_copy_atom_b()),
      decltype(params.transform_b())>;
  return make_declval<Mma>();
}

template <class... Ts>
auto
build_collective_epilogue(Sm80EpilogueParams<Ts...> params) {
  using Epilogue = cutlass::epilogue::collective::Epilogue<
      decltype(params.stride_c()),
      decltype(params.stride_d()),
      decltype(params.thread_epilogue_op()),
      decltype(params.smem_layout()),
      decltype(params.copy_atom_r2s()),
      decltype(params.tiled_copy_s2r()),
      decltype(params.copy_atom_r2g())>;
  return make_declval<Epilogue>();
}

template <class... Ts>
auto
build_collective_epilogue(Sm90EpilogueParams<Ts...> params) {
  using Epilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      decltype(params.dispatch_policy()),
      decltype(params.tile_shape()),
      decltype(params.epilogue_tile_mn()),
      decltype(params.element_c()),
      decltype(params.stride_c()),
      decltype(params.element_d()),
      decltype(params.stride_d()),
      decltype(params.fusion_callbacks()),
      decltype(params.copy_op_g2s()),
      decltype(params.smem_layout_atom_c()),
      decltype(params.copy_op_s2r()),
      decltype(params.copy_op_s2g()),
      decltype(params.smem_layout_atom_d()),
      decltype(params.copy_op_r2s())>;
  return make_declval<Epilogue>();
}

}  // namespace cutlass_v3_builder
}  // namespace bytedance::flux
