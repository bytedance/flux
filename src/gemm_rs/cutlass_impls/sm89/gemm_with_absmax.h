// clang-format off
/***************************************************************************************************
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Gemm kernel with an epilogue that computes the absolute maximum value of the output
    and a pre-activation-function auxiliary output. The auxiliary output is also (optionally)
    stored to global memory.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/layout/layout.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"
#include "cutlass/gemm/kernel/params_universal_base.h"

#include "cutlass/trace.h"

#include "gemm_rs/reduce_scatter_kernel.hpp"
#include "flux/cuda/cuda_common_device.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

// Gemm that computes the absolute maximum value of the output and a pre-activation-function
// auxiliary output.
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct GemmRSWithAbsMax {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(
    128 / sizeof_bits<ElementA>::value,
    128 / sizeof_bits<ElementB>::value
  );

  //
  // Structures
  //

  /// Argument structure
  struct Arguments : UniversalArgumentsBase
  {
    //
    // Data members
    //

    typename EpilogueOutputOp::Params epilogue;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C;
    void * ptr_D;
    void * ptr_Aux;

    void * ptr_Vector;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_Vector;

    typename LayoutA::Stride::Index lda;
    typename LayoutB::Stride::Index ldb;
    typename LayoutC::Stride::Index ldc;
    typename LayoutC::Stride::Index ldd;
    typename LayoutC::Stride::Index ldaux;
    typename LayoutC::Stride::Index ldr;

    ////////// for GEMM + ReduceScatter ///////
    typename bytedance::flux::ReduceScatterOpBase<ElementC>::Arguments rs_args;
    ////////// for GEMM + ReduceScatter ///////

    //
    // Methods
    //

    Arguments():
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr),
      ptr_Aux(nullptr)
    {}

    /// Constructs an arguments structure with ldaux
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      void * ptr_Aux,
      void * ptr_Vector,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      int64_t batch_stride_Vector,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldc,
      typename LayoutC::Stride::Index ldd,
      typename LayoutC::Stride::Index ldr,
      typename LayoutC::Stride::Index ldaux)
    :
      UniversalArgumentsBase(mode, problem_size, batch_count, batch_stride_D),
      epilogue(epilogue),
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D), ptr_Aux(ptr_Aux),
      ptr_Vector(ptr_Vector),
      batch_stride_A(batch_stride_A),
      batch_stride_B(batch_stride_B),
      batch_stride_C(batch_stride_C),
      batch_stride_Vector(batch_stride_Vector),
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd), ldaux(ldaux), ldr(ldr)
    {
    }

    /// Constructs an Arguments structure without ldaux.
    /// These parameters are overridden with D batch stride and ldd.
    Arguments(
      GemmUniversalMode mode,
      GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      void * ptr_Aux,
      void * ptr_Vector,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      int64_t batch_stride_Vector,
      typename LayoutA::Stride::Index lda,
      typename LayoutB::Stride::Index ldb,
      typename LayoutC::Stride::Index ldc,
      typename LayoutC::Stride::Index ldd,
      typename LayoutC::Stride::Index ldr)
    : Arguments(mode, problem_size, batch_count, epilogue, ptr_A, ptr_B, ptr_C, ptr_D, ptr_Aux, ptr_Vector,
               batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D, batch_stride_Vector,
               lda, ldb, ldc, ldd, ldr, ldd)
    {
    }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);

      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.batch_stride_A, args.batch_stride_B);

      return args;
    }
  };


  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params : UniversalParamsBase<
    ThreadblockSwizzle,
    ThreadblockShape,
    ElementA,
    ElementB,
    ElementC,
    LayoutA,
    LayoutB>
  {
    using ParamsBase = UniversalParamsBase<
      ThreadblockSwizzle,
      ThreadblockShape,
      ElementA,
      ElementB,
      ElementC,
      LayoutA,
      LayoutB>;

    //
    // Data members
    //

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::AuxOutputTileIterator::Params params_Aux;

    typename EpilogueOutputOp::Params output_op;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;
    void * ptr_Aux;

    void * ptr_Vector;
    typename LayoutC::Stride::Index ldr;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_Vector;

    ////////// for GEMM + ReduceScatter ///////
    typename bytedance::flux::ReduceScatterOpBase<ElementC>::Arguments rs_args;
    int* barrier_ptr;
    ////////// for GEMM + ReduceScatter ///////

    //
    // Host dispatch API
    //

    /// Default constructor
    Params() = default;

    /// Constructor
    Params(
      Arguments const &args,  /// GEMM application arguments
      int device_sms,         /// Number of SMs on the device
      int sm_occupancy)       /// Kernel SM occupancy (in thread blocks)
    :
      ParamsBase(args, device_sms, sm_occupancy),
      params_A(args.lda),
      params_B(args.ldb),
      params_C(args.ldc),
      params_D(args.ldd),
      params_Aux(args.ldaux),
      output_op(args.epilogue),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      ptr_Aux(args.ptr_Aux),
      ptr_Vector(args.ptr_Vector),
      ldr(args.ldr),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      batch_stride_Vector(args.batch_stride_Vector),
      rs_args(args.rs_args),
      barrier_ptr(rs_args.barrier_ptrs[rs_args.rank])
    {
      constexpr int kM = ThreadblockShape::kM;
      if (rs_args.use_gemmk) {
        int m = rs_args.m;
        int m_per_rank = m / rs_args.world_size;
        int tiled_m_per_rank = (m_per_rank + kM - 1) / kM;
        ParamsBase::grid_tiled_shape.m() = tiled_m_per_rank * rs_args.world_size;
      }
    }

    /// Lightweight update given a subset of arguments.
    CUTLASS_HOST_DEVICE
    void update(Arguments const &args)
    {
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;
      ptr_Aux = args.ptr_Aux;

      ptr_Vector = args.ptr_Vector;
      ldr = args.ldr;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      this->batch_stride_D = args.batch_stride_D;
      batch_stride_Vector = args.batch_stride_Vector;

      output_op = args.epilogue;

      rs_args = args.rs_args;
    }
  };


  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

public:

  //
  // Host dispatch API
  //

  /// Determines whether kernel satisfies alignment
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size) {

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutA, layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::RowMajorInterleaved<32>>::value
            || platform::is_same<LayoutB, layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajorInterleaved<32>>::value
            || platform::is_same<LayoutC, layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

public:

  //
  // Device-only API
  //

  // Factory invocation
  CUTLASS_DEVICE
  static void invoke(
    Params const &params,
    SharedStorage &shared_storage)
  {
    GemmRSWithAbsMax op;
    op(params, shared_storage);
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    // clang-format on
    ThreadblockSwizzle threadblock_swizzle(params.problem_size,
                                           {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
                                           params.grid_tiled_shape,
                                           params.rs_args.rank,
                                           params.rs_args.world_size,
                                           params.rs_args.nnodes,
                                           params.rs_args.use_gemmk,
                                           params.rs_args.per_tile_flags,
                                           params.rs_args.use_1d_ring,
                                           params.rs_args.args_workspace);
    // clang-format off

    cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    //
    // Fetch pointers based on mode.
    //
    if (params.mode == GemmUniversalMode::kGemm ||
      params.mode == GemmUniversalMode::kGemmSplitKParallel) {

      if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {

        problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
      }

      offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
      ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_A = static_cast<ElementA * const *>(params.ptr_A)[threadblock_tile_offset.k()];
      ptr_B = static_cast<ElementB * const *>(params.ptr_B)[threadblock_tile_offset.k()];
    }

    __syncthreads();

    /// for gemmk adjustment
    int offset_m = 0;
    constexpr int kM = ThreadblockShape::kM;
    int m_begin = threadblock_tile_offset.m() * kM;
    const auto& rs_args = params.rs_args;
    if (rs_args.use_gemmk) {
      int m = params.problem_size.m();
      int m_per_rank = m / rs_args.world_size;
      int tiled_m_per_rank = (m_per_rank + kM - 1) / kM;
      int segment = threadblock_tile_offset.m() / tiled_m_per_rank;
      int offset_per_segment = m_per_rank - tiled_m_per_rank * kM;
      offset_m = segment * offset_per_segment;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM + offset_m,
      offset_k,
    };

    cutlass::MatrixCoord tb_offset_B{
      offset_k,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();

    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations,
      accumulators,
      iterator_A,
      iterator_B,
      accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM + offset_m,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C);
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);
    typename Epilogue::ElementAuxOutput *ptr_Aux = static_cast<typename Epilogue::ElementAuxOutput *>(params.ptr_Aux);
    typename Epilogue::ElementVector *ptr_Vector = static_cast<typename Epilogue::ElementVector *>(params.ptr_Vector);

    //
    // Fetch pointers based on mode.
    //

    //
    // Special path when split-K not enabled.
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() == 1) {

      // Tile iterators loading from source tensors.
      typename Epilogue::OutputTileIterator iterator_C(
        params.params_C,
        ptr_C,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      // Tile iterator writing to destination tensor.
      typename Epilogue::OutputTileIterator iterator_D(
        params.params_D,
        ptr_D,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      // Tile iterator writing to auxiliary tensor.
      typename Epilogue::AuxOutputTileIterator iterator_Aux(
        params.params_Aux,
        ptr_Aux,
        params.problem_size.mn(),
        thread_idx,
        threadblock_offset
      );

      // Construct the epilogue
      Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx,
        warp_idx,
        lane_idx);

      // Move to appropriate location for this output tile
      if (ptr_Vector) {
        ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
      }

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op,
               ptr_Vector,
               iterator_D,
               accumulators,
               iterator_C,
               iterator_Aux,
               params.problem_size.mn(),
               threadblock_offset);

      set_barrier_ptr(params, threadblock_tile_offset, thread_idx); // set barrier ptr
      return;
    }

    //
    // Slower path when split-K or batching is needed
    //

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == GemmUniversalMode::kGemm) {

      // If performing a reduction via split-K, fetch the initial synchronization
      if (params.grid_tiled_shape.k() > 1) {

        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      }
    }
    else if (params.mode == GemmUniversalMode::kGemmSplitKParallel) {
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C += threadblock_tile_offset.k() * params.batch_stride_C;
      ptr_D += threadblock_tile_offset.k() * params.batch_stride_D;
      if (ptr_Aux) {
        ptr_Aux += threadblock_tile_offset.k() * params.batch_stride_D;
      }
      if (ptr_Vector) {
        ptr_Vector += threadblock_tile_offset.k() * params.batch_stride_Vector;
      }
    }
    else if (params.mode == GemmUniversalMode::kArray) {
      ptr_C = static_cast<ElementC * const *>(params.ptr_C)[threadblock_tile_offset.k()];
      ptr_D = static_cast<ElementC * const *>(params.ptr_D)[threadblock_tile_offset.k()];
      if (ptr_Aux) {
        ptr_Aux = static_cast<typename Epilogue::ElementAuxOutput * const *>(params.ptr_Aux)[threadblock_tile_offset.k()];
      }
      if (ptr_Vector) {
        ptr_Vector = static_cast<typename Epilogue::ElementVector * const *>(params.ptr_Vector)[threadblock_tile_offset.k()];
      }
    }

    // Tile iterators loading from source tensors.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to auxiliary destination tensor.
    typename Epilogue::AuxOutputTileIterator iterator_Aux(
      params.params_Aux,
      // Only the final block writes the auxiliary tensor
      ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
          (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
          ? nullptr
          : ptr_Aux,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if ((params.mode == GemmUniversalMode::kGemm) && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }

    // Move to appropriate location for this output tile
    if (ptr_Vector) {
      ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op,
             // Only the final block uses Vector
             ((params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) &&
              (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
                 ? nullptr
                 : ptr_Vector,
             iterator_D,
             accumulators,
             iterator_C,
             iterator_Aux,
             params.problem_size.mn(),
             threadblock_offset);
  set_barrier_ptr(params, threadblock_tile_offset, thread_idx);

  //////////// TODO(houqi.1993) ////// DONT'T modify epilogue. but just add code here.
  ////////////

    //
    // Release the semaphore
    //

    if ((params.mode == GemmUniversalMode::kGemm)  && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }
  }

  // clang-format on
  // runs this only with thread_idx == 0
  CUTLASS_DEVICE
  void set_barrier_ptr(const Params &params, const cutlass::gemm::GemmCoord &threadblock_tile_offset, int thread_idx) {
    const auto rs_args = params.rs_args;
    const auto &problem_size = params.problem_size;
    constexpr int kM = Mma::Shape::kM;
    constexpr int kN = Mma::Shape::kN;
    int tiled_m = (problem_size.m() + kM - 1) / kM;
    int tiled_n = (problem_size.n() + kN - 1) / kN;
    int tile_idx = tiled_n * threadblock_tile_offset.m() + threadblock_tile_offset.n();
    int rank = rs_args.rank;
    int world_size = rs_args.world_size;
    int * barrier_ptr = params.barrier_ptr;

    using Barrier = cutlass::Barrier;
    using namespace bytedance::flux;
    int num_tiles = tiled_m * tiled_n;
    if (rs_args.per_tile_flags) {
      auto flag = PerTileFlagsWrapper(barrier_ptr, num_tiles);
      if (rs_args.use_barrier_queue) {
        auto work_queue_flag = BarrierWorkQeueuFlagsWrapper(flag.extra_ptr(0), world_size);
        int tiled_m_per_rank = tiled_m / world_size;
        int m_segment = (tile_idx / tiled_n) / tiled_m_per_rank;
        // lock area start
        int head_value = atomicAdd(work_queue_flag.epilogue_done_ptr(m_segment), 1);
        atomic_store_dev(flag.epilogue_queue_ptr(head_value), tile_idx + 1);
      }
      Barrier::arrive_inc(flag.epilogue_ptr(tile_idx), thread_idx, 0, 1);
    }
    else {
      __syncthreads();
      auto wrapper = PerRankFlagsWrapper(barrier_ptr, world_size * rs_args.n_split);
      if (thread_idx == 0) {
        int m = problem_size.m();
        int tiled_n = (problem_size.n() + kN - 1) / kN;
        int segment_idx = tile_idx / tiled_n;
        int m_per_rank = m / world_size;
        if (rs_args.use_gemmk) {
          int segments_per_rank = (m_per_rank + kM - 1) / kM; // gemmk only
          int counter = atomicAdd(wrapper.counter_ptr(segment_idx), 1);
          if (counter == tiled_n * segments_per_rank - 1) {
            atomic_add_release_dev(wrapper.gemm_done_ptr(segment_idx), 1);
          }
        }
        else {
          int m_start = segment_idx * kM, m_end = std::min((segment_idx + 1) * kM, m) - 1;
          int segment_start = m_start / m_per_rank, segment_end = m_end / m_per_rank;
          for (int i = segment_start; i <= segment_end; i++) {
            // tiled per rank
            int tiled_m_start = i * m_per_rank / kM;
            int tiled_m_end = ((i + 1) * m_per_rank - 1) / kM;
            int m_seg_size = tiled_m_end - tiled_m_start + 1;
            int counter = atomicAdd(wrapper.counter_ptr(i), 1);
            if (counter == tiled_n * m_seg_size - 1) {
              atomic_add_release_dev(wrapper.gemm_done_ptr(i), 1);
            }
          }
        }
      }
    }
  }
  // clang-format off
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
// clang-format on
