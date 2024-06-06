//===- sm80_gemm_universal_with_visitor.hpp ----------------------- C++ ---===//
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
// Some code from NVIDIA cutlass project
// Original license as follows
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor_streamk.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

#include "./sm80_all_gather_gemm.hpp"

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
struct Sm80GemmWithVisitor {
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
  class Sm80GemmStreamk
      : public GemmWithEpilogueVisitor<typename GemmBase::Mma, Epilogue, SwizzleT> {};

  /// GemmWIthVisitor with StreamkFeature member type
  template <class SwizzleT>
  class Sm80GemmStreamk<SwizzleT, typename SwizzleT::StreamkFeature>
      : public Sm80AGGemmWithEpilogueVisitorStreamk<typename GemmBase::Mma, Epilogue, SwizzleT> {};

  /// Select kernel by ThreadblockSwizzle's support for StreamkFeature
  using GemmKernel = Sm80GemmStreamk<ThreadblockSwizzle>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass