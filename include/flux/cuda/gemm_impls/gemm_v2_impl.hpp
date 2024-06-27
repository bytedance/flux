//===- gemm_v2_impl.hpp ------------------------------------------- C++ ---===//
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

// This file should be included before any other cutlass device headers.
// The order of cutlass headers is carefully adjusted
#pragma once
#include <type_traits>
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/gemm/kernel/default_gemm_with_absmax.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/cuda/cuda_common.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/cuda/gemm_impls/gemm_operator_base_default_impl.hpp"

#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/config.hpp"
#include "cute/container/tuple.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/trace.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/default_gemm_with_absmax.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/epilogue/threadblock/fusion/visitor_load.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"

namespace bytedance::flux {
template <class Tuple>
constexpr auto
to_gemm_shape(Tuple tuple) {
  return cutlass::gemm::
      GemmShape<cute::size<0>(tuple), cute::size<1>(tuple), cute::size<2>(tuple)>();
}

namespace gemm_v2_impl {
namespace detail {
template <class AlwaysVoid, template <class...> class Op, class... Args>
struct detector : public std::false_type {};
template <template <class...> class Op, class... Args>
struct detector<std::void_t<Op<Args...>>, Op, Args...> : public std::true_type {};
template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<void, Op, Args...>;

template <typename T>
using has_custom_gemm_device_ = decltype(&T::custom_gemm_device);

template <typename T, typename... TArgs>
using has_custom_evt_d_ = decltype(&T::template custom_evt_d<TArgs...>);
}  // namespace detail

template <typename T>
constexpr bool has_custom_gemm_device =
    detail::is_detected<detail::has_custom_gemm_device_, T>::value;
template <typename T, typename... TArgs>
constexpr bool has_custom_evt_d =
    detail::is_detected<detail::has_custom_evt_d_, T, TArgs...>::value;

template <class TBSwizzle, class AlignmentC, class EVT>
struct KernelParams {
  auto
  tb_swizzle() {
    return make_declval<TBSwizzle>();
  }
  constexpr int
  alignment_c() {
    return AlignmentC{};
  }
  auto
  evt() {
    return make_declval<EVT>();
  }
};

}  // namespace gemm_v2_impl

using SystemBarrier = cutlass::Barrier;

template <class GemmMetaT, class GemmHParamsT, class DerivedImpl>
struct GemmV2Impl
    : public GemmOperatorBaseDefaultImplMixin<GemmV2Impl<GemmMetaT, GemmHParamsT, DerivedImpl>> {
 public:
  using Base = GemmOperatorBaseDefaultImplMixin<GemmV2Impl>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2Impl)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  using ArchTag = decltype(to_cutlass_archtag(meta.arch()));
  using OpClass = cutlass::arch::OpClassTensorOp;

  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  static constexpr int AlignmentA = 128 / cute::sizeof_bits_v<ElementA>;
  static constexpr int AlignmentB = 128 / cute::sizeof_bits_v<ElementB>;
  using GmemLayoutA = decltype(to_cutlass_layout_a(meta.gemm_layout()));
  using GmemLayoutB = decltype(to_cutlass_layout_b(meta.gemm_layout()));

  using ElementC = decltype(to_cutlass_element(dt_conf.c()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));
  using ElementAccumulator = decltype(to_cutlass_element(dt_conf.acc()));
  using ElementCNoVoid = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;
  static constexpr bool has_bias = not cute::is_void_v<ElementC>;
  using GmemLayoutC = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using GmemLayoutD = GmemLayoutC;
  using StrideC = cutlass::gemm::TagToStrideC_t<GmemLayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<GmemLayoutD>;

  using TileShape = decltype(hparams.tile_shape());
  using ThreadblockShape = decltype(to_gemm_shape(TileShape{}));
  static constexpr auto gemm_v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
  using WarpShape = decltype(to_gemm_shape(gemm_v2_hparams.warp_shape()));
  using InstructionShape = decltype(to_gemm_shape(gemm_v2_hparams.instruction_shape()));
  static constexpr int EVTEpilogueStages = 1;

  static constexpr bool is_fp8_gemm = is_fp8_dtype(dt_conf.a()) && is_fp8_dtype(dt_conf.b());
  static constexpr bool is_sm89 = (meta.arch() == _Sm89{});

  template <class... Ts>
  auto
  output_tile_thread_map(gemm_v2_impl::KernelParams<Ts...> params) const {
    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementCNoVoid,
        params.alignment_c(),
        EVTEpilogueStages>;
    return make_declval<OutputTileThreadMap>();
  }

  template <class... Ts>
  auto
  evt_d(gemm_v2_impl::KernelParams<Ts...> params) const {
    if constexpr (gemm_v2_impl::has_custom_evt_d<DerivedImpl, Ts...>) {
      // if Derived has defined evt_d then CRTP it
      return static_cast<DerivedImpl const *>(this)->custom_evt_d(params);
    } else {
      return this->default_evt_d(params);
    }
  }

  template <class... Ts>
  auto
  default_evt_d(gemm_v2_impl::KernelParams<Ts...> params) const {
    using namespace cutlass::epilogue::threadblock;
    using ElementCompute = ElementD;
    using EVT_Compute0 = Sm80EVT<
        VisitorCompute<
            cutlass::multiplies,
            ElementD,
            ElementCompute,
            cutlass::FloatRoundStyle::round_to_nearest>,  // alpha * acc
        VisitorScalarBroadcast<ElementAccumulator>,       // alpha
        VisitorAccFetch                                   // acc
        >;
    if constexpr (cute::is_void_v<ElementC>) {
      return make_declval<EVT_Compute0>();
    } else {
      using OutputTileThreadMap = decltype(this->output_tile_thread_map(params));
      // NOTE: Cutlass 2.x evt does not have alternative to Sm90SrcFetch that
      // fetches the C tensor of the epilogue. So we need to do AuxLoad for C
      using C = VisitorAuxLoad<
          OutputTileThreadMap,
          ElementCNoVoid,
          cute::Stride<int64_t, cute::_1, int64_t>  // StrideMNL
          >;
      using EVT_Compute1 = Sm80EVT<  // D
          VisitorCompute<
              cutlass::multiply_add,
              ElementD,
              ElementCompute,
              cutlass::FloatRoundStyle::round_to_nearest>,  // beta * C + (alpha * acc)
          VisitorScalarBroadcast<ElementAccumulator>,       // beta
          C,                                                // C
          EVT_Compute0>;
      return make_declval<EVT_Compute1>();
    }
  }

  auto
  default_kernel_params() const {
    if constexpr (cute::is_same_v<ArchTag, cutlass::arch::Sm89> && this->is_fp8_gemm) {
      using SM89TBSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

      // using SM89AlignmentC = cute::min(8, 128 / cutlass::sizeof_bits_v<ElementCNoVoid>);
      using SM89AlignmentC_Type = cute::Int<8>;

      using SM89Epilogue = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
          cutlass::epilogue::thread::Identity,  // maybe not need this, so use Identity
          ElementCNoVoid,
          ElementCNoVoid,
          SM89AlignmentC_Type{},
          ElementAccumulator,
          ElementAccumulator>;

      return gemm_v2_impl::KernelParams<SM89TBSwizzle, SM89AlignmentC_Type, SM89Epilogue>();
    } else {
      using TBSwizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

      using AlignmentC_Type = cute::Int<128 / cute::sizeof_bits_v<ElementCNoVoid>>;

      using namespace cutlass::epilogue::threadblock;
      auto kparams = gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, void>();
      using OutputTileThreadMap = decltype(this->output_tile_thread_map(kparams));
      using EVT_D = decltype(this->evt_d(kparams));
      using StoreD = VisitorAuxStore<
          OutputTileThreadMap,
          ElementD,
          cutlass::FloatRoundStyle::round_to_nearest,
          cute::Stride<int64_t, cute::_1, int64_t>>;
      using EVT = Sm80EVT<StoreD, EVT_D>;
      return gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, EVT>();
    }
  }

  template <class... Ts>
  auto
  default_gemm_kernel(gemm_v2_impl::KernelParams<Ts...> params) const {
    /*
    Ada FP8 GEMM.

    In addition to using FP8 Tensor Core instructions, the Ada FP8 GEMM uses a distinct epilogue
    that enables additional scaling of operands/outputs, storing a pre-activation-function output
    tensor (called the "auxiliary" output), and computing the absolute maximum value of the
    outputs.

    Pseudocode for this epilogue is as follows:

    Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
    D = activation(Aux)

    if Aux is fp8 type:
        abs_max_output = max( abs(aux) | (for every aux in Aux))
        Aux = scale_aux * Aux
    endif

    if D is fp8 type:
        abs_max_output = max( abs(d) | (for every d in D))
        D = scale_d * D
    endif

    Parameter Aux is optionally stored to global memory
    */

    using Operator = cute::conditional_t<
        to_gemm_v2_meta(meta.impl_spec()).fast_accum(),
        cutlass::arch::OpMultiplyAddFastAccum,
        cutlass::arch::OpMultiplyAdd>;
    if constexpr (cute::is_same_v<ArchTag, cutlass::arch::Sm89> && this->is_fp8_gemm) {
      using SM89Impl = cutlass::gemm::kernel::DefaultGemmWithAbsMax<
          ElementA,
          GmemLayoutA,
          cutlass::ComplexTransform::kNone,
          AlignmentA,
          ElementB,
          GmemLayoutB,
          cutlass::ComplexTransform::kNone,
          AlignmentB,
          ElementCNoVoid,
          GmemLayoutC,
          ElementAccumulator,
          OpClass,
          ArchTag,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          decltype(params.evt()),
          decltype(params.tb_swizzle()),
          hparams.mainloop_stage(),
          Operator>;
      return make_declval<typename SM89Impl::GemmKernel>();
    } else {
      using ElementCompute = ElementD;

      using Impl = cutlass::gemm::kernel::DefaultGemmWithVisitor<
          ElementA,
          GmemLayoutA,
          cutlass::ComplexTransform::kNone,
          AlignmentA,
          ElementB,
          GmemLayoutB,
          cutlass::ComplexTransform::kNone,
          AlignmentB,
          ElementCNoVoid,
          GmemLayoutC,
          params.alignment_c(),
          ElementAccumulator,
          ElementCompute,
          OpClass,
          ArchTag,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          decltype(params.evt()),
          decltype(params.tb_swizzle()),
          hparams.mainloop_stage(),
          cutlass::arch::OpMultiplyAdd,
          EVTEpilogueStages>;
      return make_declval<typename Impl::GemmKernel>();
    }
  }

  auto
  default_gemm_device() const {
    using GemmKernel = decltype(static_cast<DerivedImpl const *>(this)->gemm_kernel());

    return make_declval<cutlass::gemm::device::GemmUniversalBase<
        FluxGemmKernel<GemmMetaT, GemmHParamsT, GemmKernel>>>();
  }

 public:
  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    if constexpr (gemm_v2_impl::has_custom_gemm_device<DerivedImpl>) {
      // if Derived has defined gemm_device then CRTP it
      return static_cast<DerivedImpl const *>(this)->custom_gemm_device();
    } else {
      return this->default_gemm_device();
    }
  }

  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    return static_cast<DerivedImpl const *>(this)->to_gemm_args(args, args_workspace);
  }

 protected:
  int
  get_stride_b(int n, int k) const {
    if constexpr (cute::is_same_v<GmemLayoutB, cutlass::layout::RowMajor>) {
      return n;
    } else {
      static_assert(
          cute::is_same_v<GmemLayoutB, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return k;
    }
  }

  int
  get_stride_c(int m, int n) const {
    if constexpr (cute::is_same_v<GmemLayoutC, cutlass::layout::RowMajor>) {
      return n;
    } else {
      static_assert(
          cute::is_same_v<GmemLayoutC, cutlass::layout::ColumnMajor>, "requires ColumnMajor.");
      return m;
    }
  }

  // used for comm ops that doesn't have customized evt
  template <class PtrC, class StrideC, class PtrD, class StrideD, class Alpha, class Beta>
  auto
  default_get_callback_args(
      PtrC ptr_C, StrideC stride_C, PtrD ptr_D, StrideD stride_D, Alpha alpha, Beta beta) const {
    using EVT = identity_t<decltype(this->default_kernel_params().evt())>;
    if constexpr (has_bias) {
      return typename EVT::Arguments{
          // unary op: aux store D
          {
              // ternary op : beta * C + (alpha * acc)
              {{beta}},                        // leaf op+args : beta
              {ptr_C, ElementC{0}, stride_C},  // leaf op+args : C
              {
                  // binary op : alpha * acc
                  {{alpha}},  // leaf op+args : alpha
                  {},         // leaf op+args : acc
                  {}          // binary args : multiplies
              },              // end binary op
              {}              // ternary args : multiply_add
          },
          {ptr_D, stride_D}  // unary args: aux store D
      };
    } else {
      return typename EVT::Arguments{
          // unary op: aux store D
          {
              {{alpha}},  // leaf op+args : alpha
              {},         // leaf op+args : acc
              {}          // binary args : multiplies
          },
          {ptr_D, stride_D}  // unary args: aux store D
      };
    }
  }
};

}  // namespace bytedance::flux
