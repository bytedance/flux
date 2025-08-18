//===- cutlass_blockscale_gemm_impl.cu ---------------------------- C++ ---===//
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

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

#include "comm_none/cutlass_blockscale_gemm_impl.h"
#include "flux/args/comm_none.h"
#include "flux/cuda/cuda_common.h"

namespace bytedance {
namespace flux {
namespace {
using namespace cute;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
struct CutlassBlockScaleGemmImpl {
  // A matrix configuration
  using ElementA = cutlass::float_e4m3_t;     // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::float_e4m3_t;        // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                    // matrix in units of elements (up to 16 bytes)

  // C matrix configuration
  using ElementC = void;                          // Element type for C and D matrix operands
  using ElementCNonVoid = cutlass::float_e4m3_t;  // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::ColumnMajor;   // Layout type for C and D matrix operands
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementCNonVoid>::value;  // Memory access granularity/alignment
                                                           // of C matrix in units of elements (up
                                                           // to 16 bytes)

  // D matrix configuration
  using ElementD = cutlass::float_e4m3_t;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  // Auxiliary matrix configuration and other fusion types
  using ElementAux = ElementCNonVoid;
  using LayoutAux = LayoutC;
  using ElementAmax = float;
  using ElementBias = float;

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ElementBlockScale = float;   // Element type for blockscaling during accumulation
  using ElementCompute = float;      // Element type for epilogue computation
  using ArchTag =
      cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = Shape<_128, _128, _128>;             // Threadblock-level tile size
  using ClusterShape = Shape<_1, _2, _1>;                // Shape of the threadblocks in a cluster

  static constexpr int ScaleMsPerTile = 128;
  static constexpr int ScaleGranularityM = size<0>(TileShape{}) / ScaleMsPerTile;

  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<ScaleGranularityM>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
      LayoutAux,
      cutlass::epilogue::thread::Identity,
      ElementD,
      ElementCompute,
      ElementAux,
      ElementAmax,
      ElementBias,
      ElementCNonVoid>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      EpilogueSchedule,
      FusionOperation>::CollectiveOp;

  using CollectiveMainloopWithBlockWiseScaling =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          LayoutA,
          AlignmentA,
          ElementB,
          LayoutB,
          AlignmentB,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloopWithBlockWiseScaling,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename Gemm::Arguments
  args_from_flux(const BlockScaleGemmArguments &args) {
    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_C = static_cast<ElementCNonVoid const *>(args.C);
    auto ptr_D = static_cast<ElementD *>(args.D);
    auto ptr_blockscale_A = static_cast<ElementBlockScale const *>(args.blockscale_A);
    auto ptr_blockscale_B = static_cast<ElementBlockScale const *>(args.blockscale_B);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, args.l));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, args.l));
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, cute::make_shape(args.m, args.n, args.l));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, args.l));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {args.m, args.n, args.k, args.l},
        {ptr_A,
         stride_A,
         ptr_B,
         stride_B,
         args.mma_promotion_interval,
         ptr_blockscale_A,
         ptr_blockscale_B},
        {{},  // epilogue.thread
         ptr_C,
         stride_C,
         ptr_D,
         stride_D}};

    auto &fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = args.alpha;
    fusion_args.beta = args.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.scale_a = args.scale_a;
    fusion_args.scale_b = args.scale_b;
    fusion_args.scale_c = args.scale_c;
    fusion_args.scale_a_ptr = nullptr;
    fusion_args.scale_b_ptr = nullptr;
    fusion_args.scale_c_ptr = nullptr;

    // ignored if tensor types are not fp8
    fusion_args.scale_d = args.scale_d;
    fusion_args.scale_aux = args.scale_aux;
    fusion_args.scale_d_ptr = nullptr;
    fusion_args.scale_aux_ptr = nullptr;

    // leaving/setting these as nullptr disables the fusion at runtime
    fusion_args.bias_ptr = nullptr;

    fusion_args.aux_ptr = nullptr;
    fusion_args.amax_aux_ptr = nullptr;
    fusion_args.amax_D_ptr = nullptr;

    return arguments;
  }

  static void
  run(const BlockScaleGemmArguments &flux_args, void *workspace, cudaStream_t stream) {
    Gemm gemm;
    auto arguments = args_from_flux(flux_args);
    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());
  }

  static size_t
  get_workspace_size(const BlockScaleGemmArguments &flux_args) {
    auto arguments = args_from_flux(flux_args);
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    return workspace_size;
  }
};
#endif
}  // namespace

void
CutlassBlockScaleGemm::run(
    const BlockScaleGemmArguments &flux_args, void *workspace, cudaStream_t stream) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  CutlassBlockScaleGemmImpl::run(flux_args, workspace, stream);
#else
  assert(0);
#endif
}

size_t
CutlassBlockScaleGemm::get_workspace_size(const BlockScaleGemmArguments &flux_args) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  return CutlassBlockScaleGemmImpl::get_workspace_size(flux_args);
#else
  assert(0);
  return 0;
#endif
}

}  // namespace flux
}  // namespace bytedance
