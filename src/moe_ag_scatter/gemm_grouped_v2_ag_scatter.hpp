//===- gemm_grouped_v2_ag_scatter.hpp -------------------------- C++ ------===//
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
#include <cutlass/layout/matrix.h>
#include "cutlass_impls/default_ag_scatter_gemm_grouped_with_absmax.h"
#include "flux/args/moe_ag_scatter.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_grouped_v2_impl.hpp"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "moe_ag_scatter/workspace_util.h"

namespace bytedance {
namespace flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV2AGScatter_Kernel : public GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT>;
  using typename Base::ArchTag;
  using typename Base::OpClass;

  using Base::kAlignmentA, Base::kAlignmentB, Base::NumStages, Base::is_fp8_gemm;
  using typename Base::ElementA;
  using typename Base::ElementAccumulator;
  using typename Base::ElementB;
  using typename Base::ElementC;
  using typename Base::ElementCNonVoid;
  using typename Base::ElementD;
  using typename Base::GmemLayoutA;
  using typename Base::GmemLayoutB;
  using typename Base::GmemLayoutC;
  using typename Base::GmemLayoutD;
  using typename Base::InstructionShape;
  using typename Base::ThreadblockShape;
  using typename Base::WarpShape;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static_assert(
      cute::
          is_same_v<cutlass::layout::RowMajor, decltype(to_cutlass_layout_c(meta.gemm_layout()))>,
      "output must be row-major");
  static constexpr auto gemm_v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static_assert(meta.comm_op() == _AGScatter{}, "requires _AGScatter{}");

  auto
  epilogue() const {
    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMaxAlwaysScaleD<
            cutlass::epilogue::thread::Identity,  // maybe not need this, so use Identity
            ElementCNonVoid,
            ElementCNonVoid,
            cute::Int<128 / cutlass::sizeof_bits<ElementCNonVoid>::value>{},
            ElementAccumulator,
            ElementAccumulator>;
    return make_declval<EpilogueOp>();
  }

  auto
  gemm_kernel() const {
    using Epilogue = decltype(this->epilogue());

    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

    using Operator = cute::conditional_t<
        to_gemm_v2_meta(meta.impl_spec()).fast_accum(),
        cutlass::arch::OpMultiplyAddFastAccum,
        cutlass::arch::OpMultiplyAdd>;
    using SM89Impl = typename cutlass::gemm::kernel::DefaultAGScatterGemmGroupedWithAbsMax<
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
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmGroupedV2AGScatter_Device
    : public GemmGroupedV2BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmGroupedV2AGScatter_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using KernelBuilder = GemmGroupedV2AGScatter_Kernel<GemmMetaT, GemmHParamsT>;
  using Base =
      GemmGroupedV2BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmGroupedV2AGScatter_Device>;
  using typename Base::ElementA;
  using typename Base::ElementAccumulator;
  using typename Base::ElementB;
  using typename Base::ElementC;
  using typename Base::ElementCNonVoid;
  using typename Base::ElementD;
  using typename Base::GmemLayoutA;
  using typename Base::GmemLayoutB;
  using typename Base::GmemLayoutC;
  using typename Base::GmemLayoutD;
  using typename Base::ThreadblockShape;

  static_assert(
      std::is_same_v<typename Base::GmemLayoutA, cutlass::layout::RowMajor>,
      "only support GemmLayout RCR");
  static_assert(
      std::is_same_v<typename Base::GmemLayoutB, cutlass::layout::ColumnMajor>,
      "only support GemmLayout RCR");
  static_assert(
      std::is_same_v<typename Base::GmemLayoutC, cutlass::layout::RowMajor>,
      "only support GemmLayout RCR");
  static_assert(
      std::is_same_v<typename Base::GmemLayoutD, cutlass::layout::RowMajor>,
      "only support GemmLayout RCR");

  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedV2AGScatter_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto gemm_v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr int kAlignment = 128;

  void
  initialize_args_workspace(
      std::any const &unify_args, void *args_workspace, void *stream) const override {
    const auto &args = std::any_cast<GemmGroupedV2AGScatterArguments>(unify_args);
    this->initialize_args_workspace_impl(args, args_workspace, stream);
  }

  void
  initialize_args_workspace_impl(
      std::any const &unify_args, void *args_workspace, void *stream) const {
    const auto &args = std::any_cast<GemmGroupedV2AGScatterArguments>(unify_args);
    int threadblock_count = get_threadblock_count(args.sm_margin);
    make_workspace_async(
        args,
        GemmLayoutEnum::RCR,
        sizeof(ElementA),
        sizeof(ElementD),
        threadblock_count,
        args_workspace,
        (cudaStream_t)stream);
#if 0
    int workspace_size = get_args_workspace_size(unify_args);
    std::cerr << "workspace size: " << workspace_size << "\n";
    std::cerr << "problem_count: " << args.ep_nexperts * args.num_groups << "\n";
    std::cerr << "num_tiles: " << get_tile_count_approx(args) << "\n";
    std::vector<int8_t> workspace(workspace_size);
    cudaMemcpy(workspace.data(), args_workspace, workspace_size, cudaMemcpyDeviceToHost);
    std::fstream f(
        "workspace_" + std::to_string(args.rank) + ".bin", std::ios::out | std::ios::binary);
    f.write((char *)workspace.data(), workspace_size);
    f.close();
#endif
  }

  auto
  to_gemm_args_impl(GemmGroupedV2AGScatterArguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;

    // the workspace structure
    void *workspace = args_workspace;
    MoeAgScatterWorkspaceArgumements ws_args;

    int problem_count = args.ep_nexperts * args.num_groups;

    // the offsets
    int offset_problem_sizes = 0;
    int offset_ptr_A = pad_to(
        offset_problem_sizes + problem_count * sizeof(cutlass::gemm::GemmCoord), kAlignment);
    int offset_ptr_B = pad_to(offset_ptr_A + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_C = pad_to(offset_ptr_B + problem_count * sizeof(void *), kAlignment);
    int offset_ptr_D = pad_to(offset_ptr_C + problem_count * sizeof(void *), kAlignment);
    int offset_scale_D = pad_to(offset_ptr_D + problem_count * sizeof(void *), kAlignment);
    int offset_lda = pad_to(offset_scale_D + problem_count * sizeof(float *), kAlignment);
    int offset_ldb = pad_to(offset_lda + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldc = pad_to(offset_ldb + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldd = pad_to(offset_ldc + problem_count * sizeof(int64_t), kAlignment);
    int offset_ldr = pad_to(offset_ldd + problem_count * sizeof(int64_t), kAlignment);
    int offset_gather_A = pad_to(offset_ldr + problem_count * sizeof(int64_t), kAlignment);
    int offset_scatter_D = pad_to(offset_gather_A + problem_count * sizeof(int *), kAlignment);
    int offset_tile_count = pad_to(offset_scatter_D + problem_count * sizeof(int *), kAlignment);
    int offset_problem_info = pad_to(offset_tile_count + 1 * sizeof(int), kAlignment);

    // the workspace structure
    ws_args.problem_sizes = (cutlass::gemm::GemmCoord *)((char *)workspace + offset_problem_sizes);
    ws_args.ptr_A = (void **)((char *)workspace + offset_ptr_A);
    ws_args.ptr_B = (void **)((char *)workspace + offset_ptr_B);
    ws_args.ptr_C = (void **)((char *)workspace + offset_ptr_C);
    ws_args.ptr_D = (void **)((char *)workspace + offset_ptr_D);
    ws_args.scale_D = (float **)((char *)workspace + offset_scale_D);
    ws_args.lda = (int64_t *)((char *)workspace + offset_lda);
    ws_args.ldb = (int64_t *)((char *)workspace + offset_ldb);
    ws_args.ldc = (int64_t *)((char *)workspace + offset_ldc);
    ws_args.ldd = (int64_t *)((char *)workspace + offset_ldd);
    ws_args.ldr = (int64_t *)((char *)workspace + offset_ldr);
    ws_args.gather_A = (int **)((char *)workspace + offset_gather_A);
    ws_args.scatter_D = (int **)((char *)workspace + offset_scatter_D);
    ws_args.tile_count = (int *)((char *)workspace + offset_tile_count);
    ws_args.problem_info = (ProblemInfo *)((char *)workspace + offset_problem_info);

    int threadblock_count = get_threadblock_count(args.sm_margin);

    using EpilogueOutputOpParams = typename Gemm::EpilogueOutputOp::Params;
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));
    EpilogueOutputOpParams epilogue_params{
        {ElementD(args.alpha), ElementD(args.beta)},
        nullptr,                   // scaleA: always nullptr
        nullptr,                   // scaleB: always nullptr
        nullptr,                   // scaleC: always nullptr
        (float *)ws_args.scale_D,  // here is a magic: pass a `float **` as a `float *`
        nullptr,                   // scaleAux: always nullptr
        nullptr,                   // abs_max_Aux: always nullptr
        nullptr                    // abs_max_D: always nullptr
    };

    typename Gemm::Arguments gemm_args{
        (cutlass::gemm::GemmCoord *)ws_args.problem_sizes,  // GemmCoord*
        problem_count,                                      // int
        threadblock_count,                                  // int
        epilogue_params,                                    // EpilogueOutputOp::Params
        reinterpret_cast<typename Base::ElementA **>(ws_args.ptr_A),
        reinterpret_cast<typename Base::ElementB **>(ws_args.ptr_B),
        reinterpret_cast<typename Base::ElementCNonVoid **>(ws_args.ptr_C),
        reinterpret_cast<typename Base::ElementD **>(ws_args.ptr_D),
        nullptr,
        nullptr,
        ws_args.lda,
        ws_args.ldb,
        ws_args.ldc,
        ws_args.ldd,
        ws_args.ldr,
        ws_args.ldd,
        args.rank,
        args.world_size,
        args.barrier_ptr,
        ws_args.gather_A,
        ws_args.scatter_D,
        ws_args.problem_info,
        args.accum_per_rank_ptr,
        ws_args.tile_count,
        args.ep_nexperts};
    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &unified_args, void *args_workspace) const {
    auto const &args = std::any_cast<GemmGroupedV2AGScatterArguments>(unified_args);
    return this->to_gemm_args_impl(args, args_workspace);
  }

  // Preparing args from host to device workspace
  std::size_t
  get_args_workspace_size(std::any const &unify_args) const override {
    const auto &args = std::any_cast<GemmGroupedV2AGScatterArguments>(unify_args);
    int problem_count = args.num_groups * args.ep_nexperts;
    // not the accurate num_tiles but num_tiles won't exceed this
    int num_tiles = get_tile_count_approx(args);
    int threadblock_count = get_threadblock_count(args.sm_margin);
    num_tiles = pad_to(num_tiles, threadblock_count);
    // the workspace size
    int bytes =
        pad_to(sizeof(cutlass::gemm::GemmCoord) * problem_count, kAlignment) * 1  // problem_sizes
        + pad_to(sizeof(void *) * problem_count, kAlignment) * 4    // ptr_A/ptr_B/ptr_C/ptr_D
        + pad_to(sizeof(float *) * problem_count, kAlignment) * 1   // scale_D
        + pad_to(sizeof(int64_t) * problem_count, kAlignment) * 5   // lda/ldb/ldc/ldd/ldr
        + pad_to(sizeof(int *) * problem_count, kAlignment) * 2     // gather_A/scatter_D
        + pad_to(sizeof(int) * 1, kAlignment) * 1                   // tile_count
        + pad_to(sizeof(ProblemInfo) * num_tiles, kAlignment) * 1;  // problem_info

    return bytes;
  }

 private:
  int
  get_threadblock_count(int sm_margin) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    int num_multiprocessor = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    num_multiprocessor = (sm_margin < 0 || sm_margin >= num_multiprocessor)
                             ? num_multiprocessor
                             : (num_multiprocessor - sm_margin);
    int threadblock_count = Gemm::maximum_active_blocks() * num_multiprocessor;

    FLUX_CHECK(
        threadblock_count &&
        "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");
    return threadblock_count;
  }

  int
  get_tile_count_approx(const GemmGroupedV2AGScatterArguments &args) const {
    int tiled_n = (args.N + args.tile_size_n - 1) / args.tile_size_n;
    int tiled_m = (args.M_this_ep + args.tile_size_m - 1) / args.tile_size_m + args.ep_nexperts;
    return tiled_m * tiled_n * args.num_groups;
  }
};
}  // namespace flux
}  // namespace bytedance
