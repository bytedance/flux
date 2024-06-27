//===- gemm_grouped_impl.hpp -------------------------------------- C++ ---===//
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
#include <type_traits>
#include "cute/config.hpp"
#include "cute/tensor.hpp"
#include "cute/util/debug.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/trace.h"
#include "cutlass/util/packed_stride.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/cuda/gemm_impls/gemm_operator_base_default_impl.hpp"

namespace bytedance {
namespace flux {

template <class Tuple>
constexpr auto
to_gemm_shape(Tuple tuple) {
  return cutlass::gemm::
      GemmShape<cute::size<0>(tuple), cute::size<1>(tuple), cute::size<2>(tuple)>();
}

using cutlass::Status;
// D = alpha * (A @ B) + beta * C
template <class GemmMetaT, class GemmHParamsT, class DerivedImpl>
struct GemmGroupedImpl : public GemmOperatorBaseDefaultImplMixin<
                             GemmGroupedImpl<GemmMetaT, GemmHParamsT, DerivedImpl>> {
 public:
  using Base = GemmOperatorBaseDefaultImplMixin<GemmGroupedImpl>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedImpl)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});

  // Parse template parameters
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementC = decltype(to_cutlass_element(dt_conf.c()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));

  using ArchTag = decltype(to_cutlass_archtag(meta.arch()));
  using ProblemShape = cute::Shape<int, int, int>;

  using LayoutA = decltype(to_cutlass_layout_a(meta.gemm_layout()));
  static constexpr int32_t kAlignmentA = 8;

  using LayoutB = decltype(to_cutlass_layout_b(meta.gemm_layout()));
  static constexpr int32_t kAlignmentB = 8;

  using ElementOutput = ElementD;
  using LayoutOutput = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using ElementAccumulator = decltype(to_cutlass_element(dt_conf.acc()));

  /// compose cutlass grouped gemm
  // using MmaOp = typename CustomCollectiveMmaBuilder<Element, ArchTag>::Mma;
  using MmaOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;  // TODO(houqi.1993)
  // bfloat16 configuration
  // Threadblock-level tile size (concept: GemmShape)
  using ShapeMmaThreadBlock = decltype(to_gemm_shape(hparams.tile_shape()));
  // Warp-level tile size (concept: GemmShape)
  static constexpr auto cutlass2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
  using ShapeMmaWarp = decltype(to_gemm_shape(cutlass2_hparams.warp_shape()));
  // Instruction-level tile size (concept: GemmShape)
  using ShapeMmaOp = decltype(to_gemm_shape(cutlass2_hparams.instruction_shape()));

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator>;
  // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
  // This parameter is passed in at present to match the APIs of other kernels. The parameter
  // is unused within the kernel.
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
  static constexpr int32_t NumStages = 4;
  using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

  auto
  default_gemm_kernel() const {
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        kAlignmentA,
        ElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        kAlignmentB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MmaOp,
        SmArch,
        ShapeMmaThreadBlock,
        ShapeMmaWarp,
        ShapeMmaOp,
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        GroupScheduleMode::kDeviceOnly>::GemmKernel;
    return make_declval<GemmKernel>();
  }

 public:
  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    using GemmKernel = decltype(static_cast<DerivedImpl const *>(this)->gemm_kernel());
    return make_declval<
        cutlass::gemm::device::GemmGrouped<FluxGemmKernel<GemmMetaT, GemmHParamsT, GemmKernel>>>();
  }

  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    return static_cast<DerivedImpl const *>(this)->to_gemm_args(args);
  }
};

}  // namespace flux
}  // namespace bytedance
