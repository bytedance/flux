// copied from CUTLASS v3.5.0 include/cutlass/gemm/kernel/default_gemm_grouped.h
// clang-format off
/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/layout/permute.h"

#include "ag_scatter_gemm_grouped_with_absmax.h"
#include "cutlass/epilogue/threadblock/default_epilogue_with_absmax.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {
// clang-format on
template <template <typename T> class ActivationFunctor,
          typename ElementOutput_,    ///< Data type used to load and store tensors
          typename ElementAuxOutput_, ///< Data type used to store auxiliary output
          int Count,                  ///< Number of elements computed per operation
                                      ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                      ///< but we use 64 or 32 sometimes when there are not enough data to store
          typename ElementAccumulator_ = ElementOutput_, ///< Accumulator data type
          typename ElementCompute_ = ElementOutput_,     ///< Data type used to compute linear combination
          ScaleType::Kind Scale = ScaleType::Default,    ///< Control Alpha and Beta scaling
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          bool IsHeavy = false>
class LinearCombinationGenericWithScalingAndAbsMaxAlwaysScaleD
    : public LinearCombinationGenericWithScalingAndAbsMax<ActivationFunctor,
                                                          ElementOutput_,
                                                          ElementAuxOutput_,
                                                          Count,
                                                          ElementAccumulator_,
                                                          ElementCompute_,
                                                          Scale,
                                                          Round,
                                                          IsHeavy> {
  using Base = LinearCombinationGenericWithScalingAndAbsMax<ActivationFunctor,
                                                            ElementOutput_,
                                                            ElementAuxOutput_,
                                                            Count,
                                                            ElementAccumulator_,
                                                            ElementCompute_,
                                                            Scale,
                                                            Round,
                                                            IsHeavy>;

public:
  using typename Base::Params;
  CUTLASS_HOST_DEVICE LinearCombinationGenericWithScalingAndAbsMaxAlwaysScaleD(Params const &params) : Base(params) {}
  static bool const kIsScalingAndAmaxOutputNeeded = true;
};
} // namespace thread
} // namespace epilogue
// clang-format off
namespace gemm {
namespace kernel {


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_ = GroupScheduleMode::kDeviceOnly,
    /// Operation performed by GEMM
    typename Operator = typename device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA_, ElementB_, ElementC_,
        ElementAccumulator>::Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    ///
    typename Enable = void
    >
struct DefaultAGScatterGemmGroupedWithAbsMax;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Permute result D
    typename PermuteDLayout
>
struct DefaultAGScatterGemmGroupedWithAbsMax<
  ElementA,
  LayoutA,
  ComplexTransform::kNone,   // transform A
  kAlignmentA,
  ElementB,
  LayoutB,
  ComplexTransform::kNone,   // transform B
  kAlignmentB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  GroupScheduleMode_,
  Operator,
  SharedMemoryClear,
  PermuteDLayout,
  typename platform::enable_if< ! cutlass::is_complex<ElementAccumulator>::value>::type
> {

  // If true, we must construct a 'transposed-and-exchanged' Mma operator.
  static bool const kInternalTranspose = platform::is_same<LayoutC, layout::ColumnMajor>::value;

  using MapArguments = kernel::detail::MapArguments<
    ElementA,
    LayoutA,
    ComplexTransform::kNone,
    kAlignmentA,
    ElementB,
    LayoutB,
    ComplexTransform::kNone,
    kAlignmentB,
    LayoutC,
    kInternalTranspose
  >;

  // Define the default GEMM kernel
  using DefaultGemmKernel = typename kernel::DefaultGemm<
    typename MapArguments::ElementA,
    typename MapArguments::LayoutA,
    MapArguments::kAlignmentA,
    typename MapArguments::ElementB,
    typename MapArguments::LayoutB,
    MapArguments::kAlignmentB,
    ElementC,
    typename MapArguments::LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear,
    true, /*GatherA*/
    false, /*GatherB*/
    true, /*ScatterD*/
    PermuteDLayout
  >::GemmKernel;

  // Define epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithAbsMax<
    typename DefaultGemmKernel::Epilogue::Shape,
    typename DefaultGemmKernel::Epilogue::WarpMmaOperator,
    DefaultGemmKernel::Epilogue::kPartitionsK,
    ElementC,
    typename EpilogueOutputOp::ElementAuxOutput,
    ElementC,
    EpilogueOutputOp,
    DefaultGemmKernel::Epilogue::kElementsPerAccess,
    true
  >::Epilogue;

    /// Define the kernel in terms of the default kernel
  using GemmKernel = kernel::AGScatterGemmGroupedWithAbsMax<
    typename DefaultGemmKernel::Mma,
    Epilogue,
    ThreadblockSwizzle,
    GroupScheduleMode_,
    kInternalTranspose
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
// clang-format on
