//===- gemm_v2_reduce_scatter.hpp --------------------------------- C++ ---===//
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

#include <type_traits>
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/container/tuple.hpp"
#include "cute/util/type_traits.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/args/gemm_rs.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"
#include "flux/cuda/cuda_common.h"
#include "cutlass/gemm/device/gemm_universal_base.h"

#include "gemm_rs/visitor_2x_bsr.hpp"
#include "gemm_rs/epilogue_evt.hpp"
#include "gemm_rs/epilogue_evt_nvshmem.hpp"
#include "gemm_rs/tile_scheduler/threadblock_swizzle.hpp"
#include "gemm_rs/tile_scheduler/threadblock_swizzle_internode.hpp"
#include "gemm_rs/tile_scheduler/threadblock_swizzle_pcie.hpp"
#include "gemm_rs/reduce_scatter_kernel.hpp"

#include "cutlass_impls/sm80/gemmk_visitor_load.hpp"
#include "cutlass_impls/sm80/gemmk_universal_with_visitor.hpp"
#include "cutlass_impls/sm89/default_gemm_with_absmax.h"
#include "tile_scheduler/threadblock_swizzle_segment_util.hpp"

constexpr bool kFlattenTile = false;

namespace cutlass::gemm::device {
template <typename GemmKernel_>
class GemmReduceScatter : public GemmUniversalBase<GemmKernel_> {
 public:
  using Base = GemmUniversalBase<GemmKernel_>;
  using GemmKernel = GemmKernel_;
  using ThreadblockShape = typename GemmKernel::Mma::Shape;
  using Arguments = typename GemmKernel::Arguments;

 public:
  //---------------------------------------------------------------------------------------------
  // Stateful API
  //---------------------------------------------------------------------------------------------

  /// Initializes GEMM state from arguments and workspace memory
  Status
  initialize(
      Arguments const &args,
      void *workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr) {
    args_ = args;
    CUTLASS_CHECK_RTN(Base::initialize(args, workspace, stream, cuda_adapter));
    return rs_op.initialize(args.rs_args);
  }

  /// Lightweight update given a subset of arguments.
  Status
  update(Arguments const &args) {
    auto status = Base::update(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return rs_op.update(args.rs_args);
  }

  /// Runs the kernel using initialized state.
  Status
  run(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    auto rs_args = args_.rs_args;
    auto event = (cudaEvent_t)rs_args.event;
    auto rs_stream = (cudaStream_t)rs_args.rs_stream;
    // wait for current done
    CUDA_CHECK(cudaEventRecord(event, stream));
    CUDA_CHECK(cudaStreamWaitEvent(rs_stream, event));

    static int gemm_only = bytedance::flux::get_int_from_env("FLUX_GEMM_RS_GEMM_ONLY", 0);
    CUTLASS_CHECK(Base::run(stream, cuda_adapter));
    if (!gemm_only)
      CUTLASS_CHECK_RTN(rs_op.run(rs_stream));

    // wait for reduce_scatter done
    CUDA_CHECK(cudaEventRecord(event, rs_stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, event));

    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status
  operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return run(stream, cuda_adapter);
  }

  /// Runs the kernel using initialized state.
  Status
  operator()(
      Arguments const &args,
      void *workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr) {
    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter);
    }

    return status;
  }

 private:
  using T = typename GemmKernel::ElementC;
  bytedance::flux::ReduceScatterOp<T, ThreadblockShape::kM, ThreadblockShape::kN, kFlattenTile>
      rs_op;
  Arguments args_;
};
}  // namespace cutlass::gemm::device

namespace bytedance::flux {

namespace {
template <typename T>
struct tree_node_types {
  using types = cute::tuple<>;
};
template <typename... Ops>
struct tree_node_types<cutlass::epilogue::threadblock::TreeVisitor2x<Ops...>> {
  using types = cute::tuple<Ops...>;
};

template <typename T>
using tree_node_types_t = typename tree_node_types<T>::types;
template <int idx, typename tree_visitor_type>
using tree_node_type =
    typename cute::tuple_element<idx, tree_node_types_t<tree_visitor_type>>::type;
template <typename tree_visitor_type>
static constexpr int tree_visitor_size = cute::tuple_size<tree_node_types_t<tree_visitor_type>>();
}  // namespace

template <class GemmMetaT, class GemmHParamsT>
struct GemmV2ReduceScatter_Kernel : public GemmV2BaseKernel<
                                        GemmMetaT,
                                        GemmHParamsT,
                                        GemmV2ReduceScatter_Kernel<GemmMetaT, GemmHParamsT>> {
  using Base = GemmV2BaseKernel<GemmMetaT, GemmHParamsT, GemmV2ReduceScatter_Kernel>;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr bool has_bias = Base::has_bias;
  static_assert(meta.comm_op() == _ReduceScatter{}, "requires _ReduceScatter{}");
  static constexpr auto rs_meta = to_reduce_scatter_meta(meta.comm_spec());
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr bool is_fp8_gemm = is_fp8_dtype(dt_conf.a()) && is_fp8_dtype(dt_conf.b());
  static constexpr bool is_sm89 = (meta.arch() == _Sm89{});

  auto
  tb_swizzle() const {
    using namespace cutlass::gemm::threadblock;
    if constexpr (is_fp8_gemm && is_sm89) {
      return make_declval<GemmIdentityThreadblockSwizzlePcie>();
    } else if constexpr (rs_meta.comm_kind() == _InterNode{}) {
      return make_declval<ThreadblockSwizzleStreamKRankOffsetInterNode>();
    } else if constexpr (rs_meta.comm_kind() == _IntraNodePcie{}) {
      return make_declval<ThreadblockSwizzleStreamKPcie>();
    } else {
      return make_declval<ThreadblockSwizzleStreamKRankOffset>();
    }
  }

  template <class... Ts>
  auto
  evt_store_d(gemm_v2_impl::KernelParams<Ts...> params) const {
    using namespace cutlass::epilogue::threadblock;
    using T = typename Base::ElementD;
    constexpr bool fuse_reduction = rs_meta.fuse_reduction();
    constexpr bool no_nvlink = rs_meta.comm_kind() == _IntraNodePcie{};
    if constexpr (rs_meta.comm_kind() == _IntraNode{} || no_nvlink) {
      constexpr bool support_fuse_reduction =
          ((int)meta.arch()() < (int)(ArchEnum::Sm90) && std::is_same_v<T, cutlass::half_t>) ||
          meta.arch() == _Sm90{};
      using OutputTileThreadMap = decltype(this->output_tile_thread_map(params));
      using StoreD = VisitorAuxStoreScatter<
          OutputTileThreadMap,
          typename Base::ElementD,
          cutlass::FloatRoundStyle::round_to_nearest,
          cute::Stride<int64_t, cute::_1, int64_t>,
          typename Base::ThreadblockShape,
          fuse_reduction && support_fuse_reduction && !no_nvlink,
          no_nvlink,      // PcieMode
          kFlattenTile>;  // FlattenOutput
      return make_declval<StoreD>();
    } else if constexpr (rs_meta.comm_kind() == _InterNode{}) {
      using OutputTileThreadMapStore = OutputTileThreadLayoutBSR<
          typename Base::ThreadblockShape,
          typename Base::WarpShape,
          typename Base::ElementD,
          params.alignment_c(),
          Base::EVTEpilogueStages,
          fuse_reduction>;
      using StoreD = VisitorAuxStoreScatterInterNode<
          OutputTileThreadMapStore,
          typename Base::ElementD,
          cutlass::FloatRoundStyle::round_to_nearest,
          cute::Stride<int64_t, cute::_1, int64_t>,
          typename Base::ThreadblockShape,
          fuse_reduction>;  // FIXME pass ThreadBlockShape as a parameter
      return make_declval<StoreD>();
    } else {
      static_assert(
          cutlass::detail::dependent_false<decltype(rs_meta.comm_kind())>,
          "unsupported comm_kind()");
    }
  }

  auto
  kernel_params() const {
    using TBSwizzle = decltype(this->tb_swizzle());
    // using AlignmentC_Type = cute::Int<128 / cute::sizeof_bits_v<typename
    // Base::ElementCNonVoid>>;
    if constexpr (is_fp8_gemm && is_sm89) {
      using AlignmentC_Type = cute::Int<8>;
      using ElementCNonVoid = typename Base::ElementCNonVoid;
      using ElementAccumulator = typename Base::ElementAccumulator;
      using SM89Epilogue = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
          cutlass::epilogue::thread::Identity,  // maybe not need this, so use Identity
          ElementCNonVoid,
          ElementCNonVoid,
          AlignmentC_Type{},
          ElementAccumulator,
          ElementAccumulator>;
      return gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, SM89Epilogue>();
    } else {
      constexpr bool fuse_reduction = meta.comm_spec().fuse_reduction();
      constexpr int fused_align_size =
          std::is_same_v<typename Base::ElementCNonVoid, cutlass::half_t> ? 32 : 16;
      constexpr int align_c = (fuse_reduction ? fused_align_size : 128) /
                              cute::sizeof_bits_v<typename Base::ElementCNonVoid>;
      using AlignmentC_Type = cute::Int<align_c>;
      auto kparams = gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, void>();
      using EVT_D = decltype(this->evt_d(kparams));
      using StoreD = decltype(this->evt_store_d(kparams));
      using EVT = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EVT_D>;
      return gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, EVT>();
    }
  }

  auto
  gemm_kernel() const {
    using ElementCompute = typename Base::ElementD;
    auto params = this->kernel_params();
    if constexpr (rs_meta.comm_kind() == _IntraNodePcie{}) {
      if constexpr (is_fp8_gemm && is_sm89) {
        using ElementA = typename Base::ElementA;
        using ElementB = typename Base::ElementB;
        using ElementC = typename Base::ElementC;
        using ElementCNonVoid = typename Base::ElementCNonVoid;
        using ElementD = typename Base::ElementD;
        using ElementAccumulator = typename Base::ElementAccumulator;
        constexpr int AlignmentA = Base::AlignmentA;
        constexpr int AlignmentB = Base::AlignmentB;
        using GmemLayoutA = typename Base::GmemLayoutA;
        using GmemLayoutB = typename Base::GmemLayoutB;
        using GmemLayoutC = typename Base::GmemLayoutC;
        using OpClass = typename Base::OpClass;
        using ArchTag = typename Base::ArchTag;
        using ThreadblockShape = typename Base::ThreadblockShape;
        using WarpShape = typename Base::WarpShape;
        using InstructionShape = typename Base::InstructionShape;
        using Operation = cute::conditional_t<
            to_gemm_v2_meta(meta.impl_spec()).fast_accum(),
            cutlass::arch::OpMultiplyAddFastAccum,
            cutlass::arch::OpMultiplyAdd>;
        using SM89Impl = cutlass::gemm::kernel::DefaultGemmRSWithAbsMax<
            ElementA,
            GmemLayoutA,
            cutlass::ComplexTransform::kNone,
            AlignmentA,
            ElementB,
            GmemLayoutB,
            cutlass::ComplexTransform::kNone,
            AlignmentB,
            ElementCNonVoid,
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
            Operation>;
        return make_declval<typename SM89Impl::GemmKernel>();
      } else {
        using ElementScale = typename Base::ElementScale;
        using ElementD = typename Base::ElementD;
        using ElementCompute = std::conditional_t<this->is_s8_gemm, ElementScale, ElementD>;
        using MulAddOp = std::conditional_t<
            this->is_s8_gemm,
            cutlass::arch::OpMultiplyAddSaturate,
            cutlass::arch::OpMultiplyAdd>;
        using Impl = cutlass::gemm::kernel::GemmkWithVisitor<
            typename Base::ElementA,
            typename Base::GmemLayoutA,
            cutlass::ComplexTransform::kNone,
            Base::AlignmentA,
            typename Base::ElementB,
            typename Base::GmemLayoutB,
            cutlass::ComplexTransform::kNone,
            Base::AlignmentB,
            typename Base::ElementCNonVoid,
            typename Base::GmemLayoutC,
            params.alignment_c(),
            typename Base::ElementAccumulator,
            ElementCompute,
            typename Base::OpClass,
            typename Base::ArchTag,
            typename Base::ThreadblockShape,
            typename Base::WarpShape,
            typename Base::InstructionShape,
            decltype(params.evt()),
            decltype(params.tb_swizzle()),
            hparams.mainloop_stage(),
            MulAddOp,
            Base::EVTEpilogueStages>;
        return make_declval<typename Impl::GemmKernel>();
      }
    } else {
      return this->default_gemm_kernel(params);
    }
  }

  template <class... Ts>
  auto
  custom_evt_d(gemm_v2_impl::KernelParams<Ts...> params) const {
    using namespace cutlass::epilogue::threadblock;
    constexpr bool no_nvlink = rs_meta.comm_kind() == _IntraNodePcie{};
    if constexpr (this->is_s8_gemm) {
      return this->s8gemm_dequant_evt_d(params);
    } else if constexpr (no_nvlink) {
      using ElementAccumulator = typename Base::ElementAccumulator;
      using ElementC = typename Base::ElementC;
      using ElementCNonVoid = typename Base::ElementCNonVoid;
      using ElementD = typename Base::ElementD;
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
        using C = VisitorAuxLoadGemmk<  // using VisitorAuxLoadGemmk instead
            OutputTileThreadMap,
            ElementCNonVoid,
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
    } else {
      return this->default_evt_d(params);
    }
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV2ReduceScatter_Device
    : public GemmV2BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmV2ReduceScatter_Device<GemmMetaT, GemmHParamsT, GemmKernelT>,
          GemmV2ReduceScatter_Kernel<GemmMetaT, GemmHParamsT>> {
 public:
  using KernelBuilder = GemmV2ReduceScatter_Kernel<GemmMetaT, GemmHParamsT>;
  using Base = GemmV2BaseDevice<
      GemmMetaT,
      GemmHParamsT,
      GemmKernelT,
      GemmV2ReduceScatter_Device,
      KernelBuilder>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2ReduceScatter_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto rs_meta = to_reduce_scatter_meta(meta.comm_spec());
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  using Base::has_bias;
  using Base::is_fp8_gemm;
  using Base::is_sm89;

  auto
  custom_gemm_device() const {
    if constexpr (rs_meta.comm_kind() == _IntraNodePcie{}) {
      return make_declval<cutlass::gemm::device::GemmReduceScatter<GemmKernelT>>();
    } else {
      return make_declval<cutlass::gemm::device::GemmUniversalBase<GemmKernelT>>();
    }
  }

  auto
  to_gemm_args_impl(GemmReduceScatterArguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    auto ptr_A = static_cast<typename Base::ElementA const *>(args.input);
    auto ptr_B = static_cast<typename Base::ElementB const *>(args.weight);
    auto ptr_C = static_cast<typename Base::ElementC *>(const_cast<void *>(args.bias));
    auto ptr_scatter_D = reinterpret_cast<typename Base::ElementD **>(args.output_scatter_ptrs);
    auto ptr_barrier [[maybe_unused]] =
        reinterpret_cast<typename SystemBarrier::T **>(args.barrier_ptrs);

    auto stride_C = cutlass::make_cute_packed_stride(
        typename Base::StrideC{}, cute::make_shape(args.m, args.n, 1));
    auto stride_D = stride_C;

    using EVT = identity_t<decltype(KernelBuilder().kernel_params().evt())>;
    static_assert(tree_visitor_size<EVT> == 2);
    using EvtArgumentsType = typename EVT::Arguments;
    using EvtDArgumentType = typename tree_node_type<1, EVT>::Arguments;
    using EvtStoreDArgumentType = typename tree_node_type<0, EVT>::Arguments;
    constexpr bool no_nvlink = rs_meta.comm_kind() == _IntraNodePcie{};
    auto evt_d_args = [&]() {
      if constexpr (Base::has_bias) {
        if constexpr (no_nvlink) {
          return EvtDArgumentType{
              {args.beta},  // Beta
              {ptr_C,
               typename Base::ElementC(0),
               stride_C,
               args.world_size,
               args.reduce_scatter_args.use_gemmk},  // C
              {{{args.alpha}}, {}, {}},              // compute0 args
              {}                                     // compute 1
          };
        } else {
          return EvtDArgumentType{
              {args.beta},                                    // Beta
              {ptr_C, typename Base::ElementC(0), stride_C},  // C
              {{{args.alpha}}, {}, {}},                       // compute0 args
              {}                                              // compute 1
          };
        }
      } else {
        return EvtDArgumentType{{{args.alpha}}, {}, {}};
      }
    }();
    auto evt_d_store_args = [&]() {
      if constexpr (rs_meta.comm_kind() == _InterNode{}) {
        return EvtStoreDArgumentType{
            ptr_scatter_D,
            stride_D,
            args.rank,
            args.world_size,
            ptr_barrier,
            (typename Base::ElementD *)args.reduce_buffer_ptrs[args.rank],
            args.nnodes};
      } else {
        return EvtStoreDArgumentType{
            ptr_scatter_D,
            stride_D,
            args.rank,
            args.world_size,
            ptr_barrier,
            args.reduce_scatter_args.use_barrier_queue,  // use_barrier_queue
            args.reduce_scatter_args.use_gemmk,          // use_gemmk
            args.reduce_scatter_args.per_tile_flags,     // per_tile_flags
            args.reduce_scatter_args.n_split};
      }
    }();

    int stride_b = this->get_stride_b(args.n, args.k);
    auto callback_args = EvtArgumentsType{evt_d_args, evt_d_store_args};

    auto gemm_args = GemmArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {args.m, args.n, args.k},                 // problem_size
        1,                                        // split_k factors
        callback_args,
        ptr_A,            // ptr_A
        ptr_B,            // ptr_B
        nullptr,          // ptr_C (unused)
        nullptr,          // ptr_D (unused)
        args.m * args.k,  // batch_stride_A
        args.n * args.k,  // batch_stride_B
        args.m * args.n,  // batch_stride_C (unused)
        args.m * args.n,  // batch_stride_D
        args.k,           // stride_a
        stride_b,         // stride_b
        args.n,           // stride_c (unused)
        args.n,           // stride_d
        args.avail_sms);  // avail_sms
    if constexpr (rs_meta.comm_kind() == _IntraNodePcie{}) {
      gemm_args.rank = args.rank;
      gemm_args.world_size = args.world_size;
      gemm_args.nnodes = args.nnodes;
      gemm_args.rs_args = {
          .rank = args.rank,
          .world_size = args.world_size,
          .nnodes = args.nnodes,
          .m = args.m,
          .n = args.n,
          .num_blocks = args.reduce_scatter_args.reduce_scatter_num_blocks,
          .output_scatter_ptrs = ptr_scatter_D,
          .barrier_ptrs = ptr_barrier,
          .reduce_buffer_ptrs = (typename Base::ElementD **)args.reduce_buffer_ptrs,
          .rs_stream = args.reduce_scatter_args.rs_stream,
          .event = args.reduce_scatter_args.event,
          .use_gemmk = args.reduce_scatter_args.use_gemmk,
          .use_barrier_queue = args.reduce_scatter_args.use_barrier_queue,
          .per_tile_flags = args.reduce_scatter_args.per_tile_flags,
          .use_cudaMemcpyAsync = args.reduce_scatter_args.use_cudaMemcpyAsync,
          .n_split = args.reduce_scatter_args.n_split,
          .sub_world_size = args.reduce_scatter_args.sub_world_size,
          .opaque = args.reduce_scatter_args.opaque,
          .use_1d_ring = args.reduce_scatter_args.use_1d_ring,
          .use_p2p_read = args.reduce_scatter_args.use_p2p_read,
          .args_workspace = args_workspace};
    }
    return gemm_args;
  }

  auto
  to_fp8_gemm_args_impl(GemmReduceScatterArguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));

    auto ptr_C = static_cast<typename Base::ElementC *>(const_cast<void *>(args.bias));
    auto ptr_scatter_D = reinterpret_cast<typename Base::ElementD **>(args.output_scatter_ptrs);
    auto ptr_barrier [[maybe_unused]] =
        reinterpret_cast<typename SystemBarrier::T **>(args.barrier_ptrs);

    auto ptr_Aux = static_cast<ElementD *>(args.Aux);
    auto ptr_Vector = static_cast<ElementC *>(args.Vector);

    int stride_b = this->get_stride_b(args.n, args.k);
    int stride_c = this->get_stride_c(args.m, args.n);
    int stride_d = stride_c;

    using EVT = identity_t<decltype(KernelBuilder().kernel_params().evt())>;
    constexpr bool no_nvlink = rs_meta.comm_kind() == _IntraNodePcie{};

    typename EVT::Params epilogue_params{
        {ElementD(args.alpha), ElementD(args.beta)},
        args.scaleA,
        args.scaleB,
        args.scaleC,
        args.scaleD,
        args.scaleAux,
        args.abs_max_Aux,
        args.abs_max_D};

    auto gemm_args = GemmArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {args.m, args.n, args.k},
        /* batch_count = */ 1,
        epilogue_params,
        args.input,
        args.weight,
        args.bias,  // TODO(houqi.1993) always nullptr for now
        args.output_scatter_ptrs[args.rank],
        args.Aux,
        args.Vector,
        args.m * args.k,
        args.n * args.k,
        args.m * args.n,
        args.m * args.n,
        (int)args.m,  // Batch stride vector
        args.k,
        stride_b,
        stride_c,
        stride_d,
        (int64_t)0  // Leading dimension of vector. This must be 0
    );

    // FP8 only arguments.
    if constexpr (rs_meta.comm_kind() == _IntraNodePcie{}) {
      gemm_args.rs_args = {
          .rank = args.rank,
          .world_size = args.world_size,
          .nnodes = args.nnodes,
          .m = args.m,
          .n = args.n,
          .num_blocks = args.reduce_scatter_args.reduce_scatter_num_blocks,
          .output_scatter_ptrs = ptr_scatter_D,
          .barrier_ptrs = ptr_barrier,
          .reduce_buffer_ptrs = (typename Base::ElementD **)args.reduce_buffer_ptrs,
          .rs_stream = args.reduce_scatter_args.rs_stream,
          .event = args.reduce_scatter_args.event,
          .use_gemmk = args.reduce_scatter_args.use_gemmk,
          .use_barrier_queue = args.reduce_scatter_args.use_barrier_queue,
          .per_tile_flags = args.reduce_scatter_args.per_tile_flags,
          .use_cudaMemcpyAsync = args.reduce_scatter_args.use_cudaMemcpyAsync,
          .n_split = args.reduce_scatter_args.n_split,
          .sub_world_size = args.reduce_scatter_args.sub_world_size,
          .opaque = args.reduce_scatter_args.opaque,
          .use_1d_ring = args.reduce_scatter_args.use_1d_ring,
          .use_p2p_read = args.reduce_scatter_args.use_p2p_read,
          .args_workspace = args_workspace};
    }
    return gemm_args;
  }

  auto
  to_s8_gemm_args_impl(GemmReduceScatterArguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    auto kernel_builder = GemmV2ReduceScatter_Kernel<GemmMetaT, GemmHParamsT>();
    auto params = kernel_builder.kernel_params();
    using EVT = identity_t<decltype(params.evt())>;

    using EvtArgumentsType = typename EVT::Arguments;
    using EVT_D = identity_t<decltype(kernel_builder.s8gemm_dequant_evt_d(params))>;
    using StoreD = identity_t<decltype(kernel_builder.evt_store_d(params))>;
    using EvtDArgumentType = typename EVT_D::Arguments;
    using EvtStoreDArgumentType = typename StoreD::Arguments;

    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementBias = typename Base::ElementCNonVoid;
    using ElementD = typename Base::ElementD;
    using ElementScale = typename Base::ElementScale;

    auto ptr_A = static_cast<ElementA const *>(args.input);
    auto ptr_B = static_cast<ElementB const *>(args.weight);
    auto ptr_bias = static_cast<ElementBias const *>(args.bias);
    auto ptr_scatter_D = reinterpret_cast<ElementD **>(args.output_scatter_ptrs);
    auto ptr_barrier [[maybe_unused]] =
        reinterpret_cast<typename SystemBarrier::T **>(args.barrier_ptrs);
    auto ptr_scale_A = static_cast<ElementScale const *>(args.scaleA);
    auto ptr_scale_B = static_cast<ElementScale const *>(args.scaleB);
    auto beta = static_cast<ElementBias>(args.beta);

    int stride_a = args.k;
    int stride_b = this->get_stride_b(args.n, args.k);
    int stride_c = this->get_stride_c(args.m, args.n);
    int stride_d = stride_c;

    auto packed_stride_D = cutlass::make_cute_packed_stride(
        typename Base::StrideD{}, cute::make_shape(args.m, args.n, 1));

    auto evt_d_store_args = [&]() {
      if constexpr (rs_meta.comm_kind() == _InterNode{}) {
        return EvtStoreDArgumentType{
            ptr_scatter_D,
            packed_stride_D,
            args.rank,
            args.world_size,
            ptr_barrier,
            (ElementD *)args.reduce_buffer_ptrs[args.rank],
            args.nnodes};
      } else {
        return EvtStoreDArgumentType{
            ptr_scatter_D,
            packed_stride_D,
            args.rank,
            args.world_size,
            ptr_barrier,
            args.reduce_scatter_args.use_barrier_queue,  // use_barrier_queue
            args.reduce_scatter_args.use_gemmk,          // use_gemmk
            args.reduce_scatter_args.per_tile_flags,     // per_tile_flags
            args.reduce_scatter_args.n_split};
      }
    }();

    auto evt_d_args = [&]() {
      if constexpr (has_bias) {
        return EvtDArgumentType{
            {beta},                                                        // beta
            {ptr_bias, ElementBias(0), {cute::_0{}, cute::_1{}, args.n}},  // bias
            {
                {
                    {ptr_scale_A, ElementScale(0), {cute::_1{}, cute::_0{}, args.m}},  // ScaleA
                    {ptr_scale_B, ElementScale(0), {cute::_0{}, cute::_1{}, args.n}},  // ScaleB
                    {}                                                                 // Compute0
                },   // EVTCompute0
                {},  // Accum
                {}   // Compute1
            },       // EVTCompute1
            {}       // EVTCompute2
        };
      } else {
        return EvtDArgumentType{
            {
                {ptr_scale_A, ElementScale(0), {cute::_1{}, cute::_0{}, args.m}},  // ScaleA
                {ptr_scale_B, ElementScale(0), {cute::_0{}, cute::_1{}, args.n}},  // ScaleB
                {}                                                                 // Compute0
            },                                                                     // EVTCompute0
            {},                                                                    // Accum
            {}                                                                     // Compute1
        };
      }
    }();
    auto callback_args = EvtArgumentsType{evt_d_args, evt_d_store_args};

    auto gemm_args = GemmArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {args.m, args.n, args.k},                 // problem_size
        1,                                        // split_k factors
        callback_args,
        ptr_A,            // ptr_A
        ptr_B,            // ptr_B
        nullptr,          // ptr_C (unused)
        nullptr,          // ptr_D (unused)
        args.m * args.k,  // batch_stride_A
        args.n * args.k,  // batch_stride_B
        args.m * args.n,  // batch_stride_C (unused)
        args.m * args.n,  // batch_stride_D
        stride_a,         // stride_a
        stride_b,         // stride_b
        stride_c,         // stride_c (unused)
        stride_d,         // stride_d
        args.avail_sms);  // avail_sms

    if constexpr (rs_meta.comm_kind() == _IntraNodePcie{}) {
      gemm_args.rank = args.rank;
      gemm_args.world_size = args.world_size;
      gemm_args.nnodes = args.nnodes;
      gemm_args.rs_args = {
          .rank = args.rank,
          .world_size = args.world_size,
          .nnodes = args.nnodes,
          .m = args.m,
          .n = args.n,
          .num_blocks = args.reduce_scatter_args.reduce_scatter_num_blocks,
          .output_scatter_ptrs = ptr_scatter_D,
          .barrier_ptrs = ptr_barrier,
          .reduce_buffer_ptrs = (typename Base::ElementD **)args.reduce_buffer_ptrs,
          .rs_stream = args.reduce_scatter_args.rs_stream,
          .event = args.reduce_scatter_args.event,
          .use_gemmk = args.reduce_scatter_args.use_gemmk,
          .use_barrier_queue = args.reduce_scatter_args.use_barrier_queue,
          .per_tile_flags = args.reduce_scatter_args.per_tile_flags,
          .use_cudaMemcpyAsync = args.reduce_scatter_args.use_cudaMemcpyAsync,
          .n_split = args.reduce_scatter_args.n_split,
          .sub_world_size = args.reduce_scatter_args.sub_world_size,
          .opaque = args.reduce_scatter_args.opaque,
          .use_1d_ring = args.reduce_scatter_args.use_1d_ring,
          .use_p2p_read = args.reduce_scatter_args.use_p2p_read,
          .args_workspace = args_workspace};
    }

    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    if constexpr (is_sm89 && is_fp8_gemm) {
      return to_fp8_gemm_args_impl(to_argument_type(args), args_workspace);
    } else if constexpr (this->is_s8_gemm) {
      return to_s8_gemm_args_impl(to_argument_type(args), args_workspace);
    } else {
      return to_gemm_args_impl(to_argument_type(args), args_workspace);
    }
  }

  std::size_t
  get_barrier_workspace_size(std::any const &var_args) const override {
    const auto &args = to_argument_type(var_args);

    auto align_buffer = [](size_t size) { return (size + 127) / 128 * 128; };
    using ThreadblockShape = decltype(to_gemm_shape(hparams.tile_shape()));
    int epi_tile_m = ThreadblockShape::kM;
    int epi_tile_n = ThreadblockShape::kN;

    size_t nflags =
        ((args.m + epi_tile_m - 1) / epi_tile_m) * ((args.n + epi_tile_n - 1) / epi_tile_n);
    return align_buffer(nflags * sizeof(SystemBarrier::T));
  }

  [[nodiscard]] size_t
  get_args_workspace_size(std::any const &args) const override {
    if constexpr (rs_meta.comm_kind() == _IntraNodePcie{}) {
      const auto &op_args = to_argument_type(args);
      return op_args.world_size * sizeof(bytedance::flux::SegmentInfo);
    }
    return 0;
  }

 private:
  auto
  to_argument_type(std::any const &args) const {
    return std::any_cast<GemmReduceScatterArguments>(args);
  }
};
}  // namespace bytedance::flux
