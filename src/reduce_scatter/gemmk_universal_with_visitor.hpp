//===- gemmk_universal_with_visitor.hpp --------------------------- C++ ---===//
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor_streamk.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

#include "gemmk_universal_with_visitor_streamk.h"

namespace cutlass {
namespace gemm {
namespace kernel {

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
    /// Access granularity of C matrix in unit of elements
    int kAlignmentC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Element type for epilogue computation
    typename ElementEpilogue,
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
    typename FusionCallbacks,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Number of stages used in the pipelined epilogue
    int EpilogueStages = 1>
struct GemmkWithVisitor {
 public:
  using GemmBase = typename DefaultGemmUniversal<
      ElementA_,
      LayoutA_,
      TransformA,
      kAlignmentA,
      ElementB_,
      LayoutB_,
      TransformB,
      kAlignmentB,
      ElementC_,
      LayoutC_,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      epilogue::thread::
          LinearCombination<ElementC_, kAlignmentC, ElementAccumulator, ElementEpilogue>,
      ThreadblockSwizzle,
      Stages,
      Operator>::GemmKernel;

  // Define epilogue
  using Epilogue = cutlass::epilogue::threadblock::
      EpilogueWithVisitorCallbacks<typename GemmBase::Epilogue, FusionCallbacks, EpilogueStages>;

  /// GemmWithVisitor without StreamkFeature member type
  template <class SwizzleT, class Enable = void>
  class GemmStreamk : public GemmWithEpilogueVisitor<typename GemmBase::Mma, Epilogue, SwizzleT> {
  };

  /// GemmWIthVisitor with StreamkFeature member type
  template <class SwizzleT>
  class GemmStreamk<SwizzleT, typename SwizzleT::StreamkFeature>
      : public GemmkWithEpilogueVisitorStreamk<typename GemmBase::Mma, Epilogue, SwizzleT> {};

  /// Select kernel by ThreadblockSwizzle's support for StreamkFeature
  using GemmKernel = GemmStreamk<ThreadblockSwizzle>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
