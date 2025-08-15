//===- gemm_grouped_v3_impl.hpp ----------------------------------- C++ ---===//
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

// This file should be included before any other cutlass device headers.
// The order of cutlass headers is carefully adjusted
#pragma once
#include <type_traits>
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/args/comm_none.h"
#include "flux/cuda/cutlass_v3_builder.hpp"
#include "flux/cuda/dispatch_policy_ext.hpp"

namespace bytedance {
namespace flux {

template <class SizeT, class AlignT = int>
SizeT
make_align(SizeT size, AlignT align = 128) {
  return cutlass::round_nearest(size, align);
}

template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV3BaseKernel {
  using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static_assert(meta.arch() == _Sm90{}, "requires _Sm90{}");
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr bool has_bias = not cute::is_void_v<decltype(to_cutlass_element(dt_conf.c()))>;

  using ArchTag = decltype(to_cutlass_archtag(meta.arch()));
  using OpClass = cutlass::arch::OpClassTensorOp;
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementC = decltype(to_cutlass_element(dt_conf.c()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));
  using ElementAccumulator = decltype(to_cutlass_element(dt_conf.acc()));
  using ElementCNonVoid = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;
  using GmemLayoutA = decltype(to_cutlass_layout_a(meta.gemm_layout()));
  using GmemLayoutB = decltype(to_cutlass_layout_b(meta.gemm_layout()));
  using GmemLayoutC = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using GmemLayoutD = decltype(to_cutlass_layout_c(meta.gemm_layout()));
  using StrideC = cutlass::gemm::TagToStrideC_t<GmemLayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<GmemLayoutD>;
  using TileShape = decltype(hparams.tile_shape());
  static constexpr auto v3_params = to_gemm_v3_hparams(hparams.impl_spec());
  using ClusterShape = decltype(v3_params.cluster_shape());

  /// mma
  template <int EpiSmemSize = 0>
  auto
  default_collective_mma(cute::Int<EpiSmemSize> carveout_smem_size = cute::_0{}) const {
    using CollectiveEpilogue = identity_t<decltype(this->default_collective_epilogue())>;
    constexpr int epi_smem_size = carveout_smem_size == 0
                                      ? sizeof(typename CollectiveEpilogue::SharedStorage)
                                      : carveout_smem_size;

    auto old_params = cutlass_v3_builder::default_mainloop_params(
        meta, hparams, TypeWrapper<void>{}, cute::Int<epi_smem_size>{});

    constexpr bool is_input_fp8 = dt_conf.is_input_fp8();
    if constexpr (
        (is_input_fp8) and (not to_gemm_v3_meta(meta.impl_spec()).fast_accum()) and
        (not to_gemm_v3_meta(meta.impl_spec()).block_scale())) {
      constexpr int Stages = old_params.stages();
      using ClusterShape = decltype(old_params.cluster_shape());
      using NewDispatchPolicy = cutlass::gemm::FluxMainloopSm90ArrayTmaGmmaWarpSpecializedFP8<
          Stages,
          ClusterShape,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
      auto params = old_params.dispatch_policy(TypeWrapper<NewDispatchPolicy>{});
      return cutlass_v3_builder::build_collective_mainloop(params);
    } else {
      return cutlass_v3_builder::build_collective_mainloop(old_params);
    }
  }

  // epilogue
  template <class EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>
  auto
  default_collective_epilogue() const {
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits_v<ElementD>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        GmemLayoutC *,
        AlignmentC,
        ElementC,
        GmemLayoutC *,
        AlignmentC,
        EpilogueSchedule>::CollectiveOp;
    return make_declval<CollectiveEpilogue>();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT, class DerivedImpl>
struct GemmGroupedV3BaseDevice
    : public GemmOperatorBaseDefaultImplMixin<
          GemmGroupedV3BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, DerivedImpl>> {
 public:
  using Base = GemmOperatorBaseDefaultImplMixin<GemmGroupedV3BaseDevice>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedV3BaseDevice)
  using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  static constexpr int ProblemShapeAlignment = 128;
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  using InternalStrideA = cute::remove_pointer_t<typename GemmKernelT::StrideA>;
  using InternalStrideB = cute::remove_pointer_t<typename GemmKernelT::StrideB>;
  using InternalStrideC = cute::remove_pointer_t<typename GemmKernelT::StrideC>;
  using InternalStrideD = cute::remove_pointer_t<typename GemmKernelT::StrideD>;
  using ElementA = typename GemmKernelT::ElementA;
  using ElementB = typename GemmKernelT::ElementB;
  using ElementC = typename GemmKernelT::ElementC;
  using ElementD = typename GemmKernelT::ElementD;
  using ElementAccumulator = typename GemmKernelT::ElementAccumulator;

  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    return make_declval<cutlass::gemm::device::GemmUniversalAdapter<GemmKernelT>>();
  }

  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    return static_cast<DerivedImpl const *>(this)->to_gemm_args(args, args_workspace);
  }

  std::size_t
  get_args_workspace_size_impl(GemmGroupedV3Arguments const &args) const {
    std::size_t workspace_size = 0;
    // problem_sizes
    using ProblemSizeType = decay_and_strip_t<decltype(args.problem_sizes[0])>;
    workspace_size = make_align(
        workspace_size + sizeof(ProblemSizeType) * args.problem_count, ProblemShapeAlignment);
    // ptr & stride
    // A
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideA) * args.problem_count);
    // B
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideB) * args.problem_count);
    // C
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideC) * args.problem_count);
    // D
    workspace_size = make_align(workspace_size + sizeof(void *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideD) * args.problem_count);
    // ptr_alpha
    if (args.ptr_alpha != nullptr) {
      workspace_size = make_align(workspace_size + sizeof(float *) * args.problem_count);
    }
    return workspace_size;
  }

  std::size_t
  get_args_workspace_size_impl(BlockScaleGroupedGemmV3Arguments const &args) const {
    std::size_t workspace_size = 0;
    // problem_sizes
    using ProblemSizeType = decay_and_strip_t<decltype(args.problem_sizes[0])>;
    workspace_size = make_align(
        workspace_size + sizeof(ProblemSizeType) * args.problem_count, ProblemShapeAlignment);
    // ptr & stride
    // A
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideA) * args.problem_count);
    // B
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideB) * args.problem_count);
    // C
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideC) * args.problem_count);
    // D
    workspace_size = make_align(workspace_size + sizeof(void *) * args.problem_count);
    workspace_size = make_align(workspace_size + sizeof(InternalStrideD) * args.problem_count);
    if (args.ptr_alpha != nullptr) {
      workspace_size = make_align(workspace_size + sizeof(float *) * args.problem_count);
    }
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    // ptr_blockscale_B
    workspace_size = make_align(workspace_size + sizeof(void const *) * args.problem_count);
    return workspace_size;
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
      GemmGroupedV3Arguments const &args, void *args_workspace, void *stream) const {
    std::vector<uint8_t> host_workspace = this->get_host_workspace_buffer(args);
    this->initialize_args_workspace_with_buffer(host_workspace, args_workspace, stream);
  }

  void
  initialize_args_workspace_impl(
      BlockScaleGroupedGemmV3Arguments const &args, void *args_workspace, void *stream) const {
    std::vector<uint8_t> host_workspace = this->get_host_workspace_buffer(args);
    this->initialize_args_workspace_with_buffer(host_workspace, args_workspace, stream);
  }

  std::vector<uint8_t>
  get_host_workspace_buffer(GemmGroupedV3Arguments const &args) const {
    std::size_t workspace_size = this->get_args_workspace_size_impl(args);
    std::vector<uint8_t> workspace_host(workspace_size);
    uint8_t *workspace_ptr = workspace_host.data();
    std::size_t workspace_offset = 0;

    // problem_sizes
    using ProblemSizeType = decay_and_strip_t<decltype(args.problem_sizes[0])>;
    std::size_t sizeof_problem_sizes = sizeof(ProblemSizeType) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, args.problem_sizes, sizeof_problem_sizes);
    workspace_offset = make_align(workspace_offset + sizeof_problem_sizes, ProblemShapeAlignment);
    // ptr & stride

    std::vector<InternalStrideA> stride_A(args.problem_count);
    std::vector<InternalStrideB> stride_B(args.problem_count);
    std::vector<InternalStrideC> stride_C(args.problem_count);
    std::vector<InternalStrideD> stride_D(args.problem_count);
    for (int i = 0; i < args.problem_count; ++i) {
      auto [M, N, K] = args.problem_sizes[i];
      stride_A[i] = cutlass::make_cute_packed_stride(InternalStrideA{}, cute::make_shape(M, K, 1));
      stride_B[i] = cutlass::make_cute_packed_stride(InternalStrideB{}, cute::make_shape(N, K, 1));
      stride_C[i] = cutlass::make_cute_packed_stride(InternalStrideC{}, cute::make_shape(M, N, 1));
      stride_D[i] = cutlass::make_cute_packed_stride(InternalStrideD{}, cute::make_shape(M, N, 1));
    }

    // A
    std::size_t sizeof_ptr = sizeof(void *) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, args.ptr_A, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_A = sizeof(InternalStrideA) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_A.data(), sizeof_stride_A);
    workspace_offset = make_align(workspace_offset + sizeof_stride_A);

    // B
    memcpy(workspace_ptr + workspace_offset, args.ptr_B, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_B = sizeof(InternalStrideB) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_B.data(), sizeof_stride_B);
    workspace_offset = make_align(workspace_offset + sizeof_stride_B);

    // C
    memcpy(workspace_ptr + workspace_offset, args.ptr_C, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_C = sizeof(InternalStrideC) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_C.data(), sizeof_stride_C);
    workspace_offset = make_align(workspace_offset + sizeof_stride_C);

    // D
    memcpy(workspace_ptr + workspace_offset, args.ptr_D, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_D = sizeof(InternalStrideD) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_D.data(), sizeof_stride_D);
    workspace_offset = make_align(workspace_offset + sizeof_stride_D);

    // ptr_alpha
    if (args.ptr_alpha != nullptr) {
      memcpy(
          workspace_ptr + workspace_offset, args.ptr_alpha, sizeof(float *) * args.problem_count);
      workspace_offset = make_align(workspace_offset + sizeof_ptr);
    }

    FLUX_CHECK(workspace_offset == workspace_size) << workspace_offset << " != " << workspace_size;
    return workspace_host;
  }

  std::vector<uint8_t>
  get_host_workspace_buffer(BlockScaleGroupedGemmV3Arguments const &args) const {
    std::size_t workspace_size = this->get_args_workspace_size_impl(args);
    std::vector<uint8_t> workspace_host(workspace_size);
    uint8_t *workspace_ptr = workspace_host.data();
    std::size_t workspace_offset = 0;

    // problem_sizes
    using ProblemSizeType = decay_and_strip_t<decltype(args.problem_sizes[0])>;
    std::size_t sizeof_problem_sizes = sizeof(ProblemSizeType) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, args.problem_sizes, sizeof_problem_sizes);
    workspace_offset = make_align(workspace_offset + sizeof_problem_sizes, ProblemShapeAlignment);
    // ptr & stride

    std::vector<InternalStrideA> stride_A(args.problem_count);
    std::vector<InternalStrideB> stride_B(args.problem_count);
    std::vector<InternalStrideC> stride_C(args.problem_count);
    std::vector<InternalStrideD> stride_D(args.problem_count);
    for (int i = 0; i < args.problem_count; ++i) {
      auto [M, N, K] = args.problem_sizes[i];
      stride_A[i] = cutlass::make_cute_packed_stride(InternalStrideA{}, cute::make_shape(M, K, 1));
      stride_B[i] = cutlass::make_cute_packed_stride(InternalStrideB{}, cute::make_shape(N, K, 1));
      stride_C[i] = cutlass::make_cute_packed_stride(InternalStrideC{}, cute::make_shape(M, N, 1));
      stride_D[i] = cutlass::make_cute_packed_stride(InternalStrideD{}, cute::make_shape(M, N, 1));
    }

    // A
    std::size_t sizeof_ptr = sizeof(void *) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, args.ptr_A, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_A = sizeof(InternalStrideA) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_A.data(), sizeof_stride_A);
    workspace_offset = make_align(workspace_offset + sizeof_stride_A);

    // B
    memcpy(workspace_ptr + workspace_offset, args.ptr_B, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_B = sizeof(InternalStrideB) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_B.data(), sizeof_stride_B);
    workspace_offset = make_align(workspace_offset + sizeof_stride_B);

    // C
    memcpy(workspace_ptr + workspace_offset, args.ptr_C, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_C = sizeof(InternalStrideC) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_C.data(), sizeof_stride_C);
    workspace_offset = make_align(workspace_offset + sizeof_stride_C);

    // D
    memcpy(workspace_ptr + workspace_offset, args.ptr_D, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    std::size_t sizeof_stride_D = sizeof(InternalStrideD) * args.problem_count;
    memcpy(workspace_ptr + workspace_offset, stride_D.data(), sizeof_stride_D);
    workspace_offset = make_align(workspace_offset + sizeof_stride_D);

    // ptr_alpha
    if (args.ptr_alpha != nullptr) {
      memcpy(
          workspace_ptr + workspace_offset, args.ptr_alpha, sizeof(float *) * args.problem_count);
      workspace_offset = make_align(workspace_offset + sizeof_ptr);
    }

    // ptr_blockscale_A
    memcpy(workspace_ptr + workspace_offset, args.ptr_blockscale_A, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);
    // ptr_blockscale_B
    memcpy(workspace_ptr + workspace_offset, args.ptr_blockscale_B, sizeof_ptr);
    workspace_offset = make_align(workspace_offset + sizeof_ptr);

    FLUX_CHECK(workspace_offset == workspace_size) << workspace_offset << " != " << workspace_size;
    return workspace_host;
  }

  auto
  parse_common_gemm_args_from_workspace(
      GemmGroupedV3Arguments const &args, void *args_workspace) const {
    std::size_t workspace_size = this->get_args_workspace_size_impl(args);
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(args_workspace);

    std::size_t workspace_offset = 0;
    // problem_sizes
    using ProblemSizeType = decay_and_strip_t<decltype(args.problem_sizes[0])>;
    static_assert(
        cute::is_same_v<ProblemSizeType, typename ProblemShape::UnderlyingProblemShape>,
        "ProblemSize type mismatch");
    auto dev_problem_sizes = reinterpret_cast<ProblemSizeType *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(
        workspace_offset + sizeof(ProblemSizeType) * args.problem_count, ProblemShapeAlignment);
    // ptr & stride
    using Gemm = identity_t<decltype(this->gemm_device())>;
    // A
    auto dev_ptr_A = reinterpret_cast<ElementA const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);
    auto dev_stride_A = reinterpret_cast<InternalStrideA *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideA) * args.problem_count);
    // B
    auto dev_ptr_B = reinterpret_cast<ElementB const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);
    auto dev_stride_B = reinterpret_cast<InternalStrideB *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideB) * args.problem_count);
    // C
    auto dev_ptr_C = reinterpret_cast<ElementC const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);
    auto dev_stride_C = reinterpret_cast<InternalStrideC *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideC) * args.problem_count);
    // D
    auto dev_ptr_D = reinterpret_cast<ElementD **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void *) * args.problem_count);
    auto dev_stride_D = reinterpret_cast<InternalStrideD *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideD) * args.problem_count);
    // ptr_alpha
    float **dev_ptr_alpha = nullptr;
    if (args.ptr_alpha != nullptr) {
      dev_ptr_alpha = reinterpret_cast<float **>(workspace_ptr + workspace_offset);
      workspace_offset = make_align(workspace_offset + sizeof(float *) * args.problem_count);
    }

    FLUX_CHECK(workspace_offset == workspace_size) << workspace_offset << " != " << workspace_size;
    return cute::make_tuple(
        dev_problem_sizes,
        dev_ptr_A,
        dev_stride_A,
        dev_ptr_B,
        dev_stride_B,
        dev_ptr_C,
        dev_stride_C,
        dev_ptr_D,
        dev_stride_D,
        dev_ptr_alpha);
  }

  auto
  parse_common_gemm_args_from_workspace(
      BlockScaleGroupedGemmV3Arguments const &args, void *args_workspace) const {
    std::size_t workspace_size = this->get_args_workspace_size_impl(args);
    uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(args_workspace);

    std::size_t workspace_offset = 0;
    // problem_sizes
    using ProblemSizeType = decay_and_strip_t<decltype(args.problem_sizes[0])>;
    static_assert(
        cute::is_same_v<ProblemSizeType, typename ProblemShape::UnderlyingProblemShape>,
        "ProblemSize type mismatch");
    auto dev_problem_sizes = reinterpret_cast<ProblemSizeType *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(
        workspace_offset + sizeof(ProblemSizeType) * args.problem_count, ProblemShapeAlignment);
    // ptr & stride
    using Gemm = identity_t<decltype(this->gemm_device())>;
    // A
    auto dev_ptr_A = reinterpret_cast<ElementA const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);
    auto dev_stride_A = reinterpret_cast<InternalStrideA *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideA) * args.problem_count);
    // B
    auto dev_ptr_B = reinterpret_cast<ElementB const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);
    auto dev_stride_B = reinterpret_cast<InternalStrideB *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideB) * args.problem_count);
    // C
    auto dev_ptr_C = reinterpret_cast<ElementC const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);
    auto dev_stride_C = reinterpret_cast<InternalStrideC *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideC) * args.problem_count);
    // D
    auto dev_ptr_D = reinterpret_cast<ElementD **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void *) * args.problem_count);
    auto dev_stride_D = reinterpret_cast<InternalStrideD *>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(InternalStrideD) * args.problem_count);
    // ptr_alpha
    float **dev_ptr_alpha = nullptr;
    if (args.ptr_alpha != nullptr) {
      dev_ptr_alpha = reinterpret_cast<float **>(workspace_ptr + workspace_offset);
      workspace_offset = make_align(workspace_offset + sizeof(float *) * args.problem_count);
    }

    using ElementBlockScale = float;
    auto dev_ptr_blockscale_A =
        reinterpret_cast<ElementBlockScale const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);
    auto dev_ptr_blockscale_B =
        reinterpret_cast<ElementBlockScale const **>(workspace_ptr + workspace_offset);
    workspace_offset = make_align(workspace_offset + sizeof(void const *) * args.problem_count);

    FLUX_CHECK(workspace_offset == workspace_size) << workspace_offset << " != " << workspace_size;
    return cute::make_tuple(
        dev_problem_sizes,
        dev_ptr_A,
        dev_stride_A,
        dev_ptr_B,
        dev_stride_B,
        dev_ptr_C,
        dev_stride_C,
        dev_ptr_D,
        dev_stride_D,
        dev_ptr_blockscale_A,
        dev_ptr_blockscale_B,
        dev_ptr_alpha);
  }
};
}  // namespace flux
}  // namespace bytedance
