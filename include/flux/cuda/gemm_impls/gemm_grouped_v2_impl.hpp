//===- gemm_grouped_v2_impl.hpp ------------------------------------ C++ ---===//
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
#include <cstdint>
#include <type_traits>
#include "cute/config.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/tensor.hpp"
#include "cute/util/debug.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass_impls/gemm_grouped_with_absmax.h"
#include "cutlass_impls/default_gemm_grouped_with_absmax.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/trace.h"
#include "cutlass/util/packed_stride.hpp"
#include "flux/args/comm_none.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/cuda/gemm_impls/gemm_operator_base_default_impl.hpp"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"

namespace bytedance {
namespace flux {

template <class Tuple>
constexpr auto
to_gemm_shape(Tuple tuple) {
  return cutlass::gemm::
      GemmShape<cute::size<0>(tuple), cute::size<1>(tuple), cute::size<2>(tuple)>();
}

template <class SizeT, class AlignT = int>
SizeT
make_align(SizeT size, AlignT align = 128) {
  return cutlass::round_nearest(size, align);
}

template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV2BaseKernel {
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static_assert(meta.arch() == _Sm80{} or meta.arch() == _Sm89{}, "requires either SM80 or SM89");
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  using ArchTag = decltype(to_cutlass_archtag(meta.arch()));
  using OpClass = cutlass::arch::OpClassTensorOp;
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementC = decltype(to_cutlass_element(dt_conf.c()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));
  using ElementAccumulator = decltype(to_cutlass_element(dt_conf.acc()));
  using ElementCNonVoid = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;
  // static constexpr int32_t kAlignmentA = 8;
  // static constexpr int32_t kAlignmentB = 8;
  static constexpr int kAlignmentA = 128 / cute::sizeof_bits_v<ElementA>;
  static constexpr int kAlignmentB = 128 / cute::sizeof_bits_v<ElementB>;
  using GmemLayoutA = decltype(to_cutlass_layout_a(meta.gemm_layout()));
  using GmemLayoutB = decltype(to_cutlass_layout_b(meta.gemm_layout()));
  using GmemLayoutC = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using GmemLayoutD = GmemLayoutC;
  using ThreadblockShape = decltype(to_gemm_shape(hparams.tile_shape()));
  static constexpr auto v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
  using WarpShape = decltype(to_gemm_shape(v2_hparams.warp_shape()));
  using InstructionShape = decltype(to_gemm_shape(v2_hparams.instruction_shape()));
  static constexpr int32_t NumStages = 4;

  static constexpr bool is_fp8_gemm = is_fp8_dtype(dt_conf.a()) && is_fp8_dtype(dt_conf.b());

  auto
  default_epilogue() const {
    if constexpr (is_fp8_gemm) {
      using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
          cutlass::epilogue::thread::Identity,  // maybe not need this, so use Identity
          ElementCNonVoid,
          ElementCNonVoid,
          cute::Int<128 / cutlass::sizeof_bits<ElementCNonVoid>::value>{},
          ElementAccumulator,
          ElementAccumulator>;
      return make_declval<EpilogueOp>();
    } else {
      using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
          ElementD,
          128 / cutlass::sizeof_bits<ElementCNonVoid>::value,
          ElementAccumulator,
          ElementAccumulator>;

      return make_declval<EpilogueOp>();
    }
  }

  auto
  default_gemm_kernel() const {
    using Epilogue = decltype(this->default_epilogue());

    /// TODO: we probably need tile scheduler support
    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

    /// TODO: add host mode
    if constexpr (cute::is_same_v<ArchTag, cutlass::arch::Sm89> && this->is_fp8_gemm) {
      using Operator = cute::conditional_t<
          to_gemm_v2_meta(meta.impl_spec()).fast_accum(),
          cutlass::arch::OpMultiplyAddFastAccum,
          cutlass::arch::OpMultiplyAdd>;
      using SM89Impl = typename cutlass::gemm::kernel::DefaultGemmGroupedWithAbsMax<
          ElementA,
          GmemLayoutA,
          cutlass::ComplexTransform::kNone,
          kAlignmentA,
          ElementB,
          GmemLayoutB,
          cutlass::ComplexTransform::kNone,
          kAlignmentB,
          ElementD,
          GmemLayoutD,
          ElementAccumulator,
          OpClass,
          ArchTag,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          Epilogue,
          SwizzleThreadBlock,
          NumStages,
          GroupScheduleMode::kDeviceOnly,
          Operator>;
      return make_declval<typename SM89Impl::GemmKernel>();
    } else {
      // Note: GemmGrouped kernels other than FP8 does not have specitialization for SM89 ArchTag,
      // so we should use SM80 ArchTag instead, and rely on compiler flag to indicate the GPU arch.
      using Impl = typename cutlass::gemm::kernel::DefaultGemmGrouped<
          ElementA,
          GmemLayoutA,
          cutlass::ComplexTransform::kNone,
          kAlignmentA,
          ElementB,
          GmemLayoutB,
          cutlass::ComplexTransform::kNone,
          kAlignmentB,
          ElementD,
          GmemLayoutD,
          ElementAccumulator,
          OpClass,
          cutlass::arch::Sm80,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          Epilogue,
          SwizzleThreadBlock,
          NumStages,
          GroupScheduleMode::kDeviceOnly>;
      return make_declval<typename Impl::GemmKernel>();
    }
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT, class DerivedImpl>
struct GemmGroupedV2BaseDevice
    : public GemmOperatorBaseDefaultImplMixin<
          GemmGroupedV2BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, DerivedImpl>>,
      public GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT> {
 public:
  using Base = GemmOperatorBaseDefaultImplMixin<GemmGroupedV2BaseDevice>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedV2BaseDevice)

  using KernelBuilder = GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});

  using KernelBuilder::is_fp8_gemm;
  using typename KernelBuilder::ElementA;
  using typename KernelBuilder::ElementAccumulator;
  using typename KernelBuilder::ElementB;
  using typename KernelBuilder::ElementC;
  using typename KernelBuilder::ElementCNonVoid;
  using typename KernelBuilder::ElementD;
  using typename KernelBuilder::GmemLayoutA;
  using typename KernelBuilder::GmemLayoutB;
  using typename KernelBuilder::GmemLayoutC;
  using typename KernelBuilder::GmemLayoutD;
  using typename KernelBuilder::ThreadblockShape;

  auto
  default_gemm_device() const {
    return make_declval<cutlass::gemm::device::GemmGrouped<GemmKernelT>>();
  }

 public:
  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    return this->default_gemm_device();
  }

  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    return static_cast<DerivedImpl const *>(this)->to_gemm_args(args, args_workspace);
  }

  std::size_t
  get_args_workspace_size_impl(GemmGroupedV2Arguments const &args) const {
    std::size_t workspace_size = 0;
    workspace_size = make_align(
        workspace_size + args.problem_count * sizeof(cutlass::gemm::GemmCoord),
        128);  // problem sizes
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(void *));  // A
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(void *));  // B
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(void *));  // C
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(void *));  // D
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(void *));  // ptr_Aux
    workspace_size =
        make_align(workspace_size + args.problem_count * sizeof(void *));  // ptr_Vector
    workspace_size =
        make_align(workspace_size + args.problem_count * sizeof(void *));  // ptr_scaleA
    workspace_size =
        make_align(workspace_size + args.problem_count * sizeof(void *));  // ptr_scaleB
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(int64_t));  // lda
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(int64_t));  // ldb
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(int64_t));  // ldc
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(int64_t));  // ldd
    workspace_size = make_align(workspace_size + args.problem_count * sizeof(int64_t));  // ldr
    return workspace_size;
  }

  std::vector<uint8_t>
  get_host_workspace_buffer(GemmGroupedV2Arguments const &args) const {
    std::size_t workspace_size = this->get_args_workspace_size_impl(args);  // buffer size
    std::vector<uint8_t> workspace_host(workspace_size);                    // host
    uint8_t *workspace_ptr = workspace_host.data();
    std::size_t workspace_offset = 0;

    std::size_t sizeof_problem_sizes = args.problem_count * sizeof(cutlass::gemm::GemmCoord);
    memcpy(
        workspace_ptr + workspace_offset,
        args.problem_sizes,
        sizeof_problem_sizes);  // copy problem sizes
    workspace_offset = make_align(workspace_offset + sizeof_problem_sizes, 128);

    // prepare lead dimension array data
    auto problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord *>(args.problem_sizes);
    std::vector<int64_t> lda(args.problem_count);
    std::vector<int64_t> ldb(args.problem_count);
    std::vector<int64_t> ldc(args.problem_count);
    std::vector<int64_t> ldd(args.problem_count);
    std::vector<int64_t> ldr(args.problem_count);
    for (int i = 0; i < args.problem_count; ++i) {
      auto M = problem_sizes[i].m();
      auto N = problem_sizes[i].n();
      auto K = problem_sizes[i].k();

      lda[i] = GmemLayoutA::packed({M, K}).stride(0);
      ldb[i] = GmemLayoutB::packed({K, N}).stride(0);
      ldc[i] = GmemLayoutC::packed({M, N}).stride(0);
      ldd[i] = GmemLayoutD::packed({M, N}).stride(0);
      ldr[i] = 0;  // Leading dimension of vector. This must be 0
    }

    /// Copy gemm arguments
    std::size_t sizeof_ptr = args.problem_count * sizeof(void *);
    std::size_t sizeof_ld = args.problem_count * sizeof(int64_t);
    /// copy A/B/C/D
    memcpy(workspace_ptr + workspace_offset, args.ptr_A, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    memcpy(workspace_ptr + workspace_offset, args.ptr_B, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    memcpy(workspace_ptr + workspace_offset, args.ptr_C, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    memcpy(workspace_ptr + workspace_offset, args.ptr_D, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    memcpy(workspace_ptr + workspace_offset, args.ptr_Aux, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    memcpy(workspace_ptr + workspace_offset, args.ptr_Vector, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    memcpy(workspace_ptr + workspace_offset, args.scaleA, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    memcpy(workspace_ptr + workspace_offset, args.scaleB, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    /// copy lead dimension arrays
    memcpy(workspace_ptr + workspace_offset, lda.data(), sizeof_ld);
    workspace_offset = make_align(workspace_offset + sizeof_ld);
    memcpy(workspace_ptr + workspace_offset, ldb.data(), sizeof_ld);
    workspace_offset = make_align(workspace_offset + sizeof_ld);
    memcpy(workspace_ptr + workspace_offset, ldc.data(), sizeof_ld);
    workspace_offset = make_align(workspace_offset + sizeof_ld);
    memcpy(workspace_ptr + workspace_offset, ldd.data(), sizeof_ld);
    workspace_offset = make_align(workspace_offset + sizeof_ld);
    memcpy(workspace_ptr + workspace_offset, ldr.data(), sizeof_ld);
    workspace_offset = make_align(workspace_offset + sizeof_ld);

    FLUX_CHECK_EQ(workspace_offset, workspace_size);
    return workspace_host;
  }

  void
  initialize_args_workspace_with_buffer(
      std::vector<uint8_t> const &host_workspace, void *args_workspace, void *stream) const {
    auto cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaMemcpyAsync(
        args_workspace,
        host_workspace.data(),
        host_workspace.size(),
        cudaMemcpyHostToDevice,
        cu_stream));
    cudaStreamSynchronize(cu_stream);
  }

  void
  initialize_args_workspace_impl(
      GemmGroupedV2Arguments const &args, void *args_workspace, void *stream) const {
    std::vector<uint8_t> host_workspace = this->get_host_workspace_buffer(args);
    this->initialize_args_workspace_with_buffer(host_workspace, args_workspace, stream);
  }

  auto
  parse_group_gemm_args_from_workspace(
      GemmGroupedV2Arguments const &args, void *args_workspace) const {
    std::size_t workspace_size = this->get_args_workspace_size_impl(args);
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(args_workspace);
    std::size_t workspace_offset = 0;

    // problem_sizes
    auto dev_problem_sizes =
        reinterpret_cast<cutlass::gemm::GemmCoord *>(workspace_ptr + workspace_offset);
    workspace_offset =
        make_align(workspace_offset + args.problem_count * sizeof(cutlass::gemm::GemmCoord), 128);
    // A/B/C/D
    auto dev_ptr_A = reinterpret_cast<ElementA **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));
    auto dev_ptr_B = reinterpret_cast<ElementB **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));
    auto dev_ptr_C = reinterpret_cast<ElementCNonVoid **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));
    auto dev_ptr_D = reinterpret_cast<ElementD **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));
    auto dev_ptr_Aux = reinterpret_cast<void **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));
    auto dev_ptr_Vector = reinterpret_cast<void **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));
    auto dev_scaleA = reinterpret_cast<void **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));
    auto dev_scaleB = reinterpret_cast<void **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(void *));

    // lda/ldb/ldc/ldd/ldr
    auto dev_lda = reinterpret_cast<int64_t *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(int64_t));
    auto dev_ldb = reinterpret_cast<int64_t *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(int64_t));
    auto dev_ldc = reinterpret_cast<int64_t *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(int64_t));
    auto dev_ldd = reinterpret_cast<int64_t *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(int64_t));
    auto dev_ldr = reinterpret_cast<int64_t *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + args.problem_count * sizeof(int64_t));

    FLUX_CHECK_EQ(workspace_offset, workspace_size);
    return cute::make_tuple(
        dev_problem_sizes,
        dev_ptr_A,
        dev_ptr_B,
        dev_ptr_C,
        dev_ptr_D,
        dev_ptr_Aux,
        dev_ptr_Vector,
        dev_scaleA,
        dev_scaleB,
        dev_lda,
        dev_ldb,
        dev_ldc,
        dev_ldd,
        dev_ldr);
  }
};

}  // namespace flux
}  // namespace bytedance
