//===- gemm_v2_reduce_scatter.hpp --------------------------------- C++ ---===//
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
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cute/container/tuple.hpp"
#include "cute/util/type_traits.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/args/reduce_scatter.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"
#include "flux/cuda/cuda_common.h"
#include "cutlass/gemm/device/gemm_universal_base.h"

#include "reduce_scatter/gemmk_visitor_load.hpp"
#include "flux/utils.h"
// #include "reduce_scatter/gemmk_universal_with_visitor.hpp"
#include "reduce_scatter/visitor_2x_bsr.hpp"
#include "reduce_scatter/epilogue_evt.hpp"
#include "reduce_scatter/epilogue_evt_nvshmem.hpp"
#include "reduce_scatter/tile_scheduler/threadblock_swizzle.hpp"
#include "reduce_scatter/tile_scheduler/threadblock_swizzle_acrossnode.hpp"

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
    // return rs_op.initialize(args.rs_args);
  }

  /// Lightweight update given a subset of arguments.
  Status
  update(Arguments const &args) {
    auto status = Base::update(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    // return rs_op.update(args.rs_args);
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
    static int cuda_launch_blocking = bytedance::flux::get_int_from_env("CUDA_LAUNCH_BLOCKING", 0);
    if (cuda_launch_blocking) {
      static int counter = 0;
      if (counter++ == 0) {
        std::cerr << "[warning]: running with CUDA_LAUNCH_BLOCKING=1 will cause performance "
                     "issues. only for debug\n";
      }
      CUTLASS_CHECK(Base::run(stream, cuda_adapter));
      // if (!gemm_only)
      // CUTLASS_CHECK_RTN(rs_op.run(rs_stream));
    } else {
      // if (!gemm_only)
      // CUTLASS_CHECK_RTN(rs_op.run(rs_stream));
      CUTLASS_CHECK(Base::run(stream, cuda_adapter));
    }

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
  // bytedance::flux::ReduceScatterOp<T, ThreadblockShape::kM, ThreadblockShape::kN, kFlattenTile>
  //     rs_op;
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
class GemmV2ReduceScatter
    : public GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2ReduceScatter<GemmMetaT, GemmHParamsT>> {
 public:
  using Base = GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2ReduceScatter>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2ReduceScatter)

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
    if constexpr (rs_meta.comm_kind() == _AcrossNode{}) {
      return make_declval<ThreadblockSwizzleStreamKRankOffsetAcrossNode>();
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
    } else if constexpr (rs_meta.comm_kind() == _AcrossNode{}) {
      using OutputTileThreadMapStore = OutputTileThreadLayoutBSR<
          typename Base::ThreadblockShape,
          typename Base::WarpShape,
          typename Base::ElementD,
          params.alignment_c(),
          Base::EVTEpilogueStages,
          fuse_reduction>;
      using StoreD = VisitorAuxStoreScatterAccrossNode<
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
    // using AlignmentC_Type = cute::Int<128 / cute::sizeof_bits_v<typename Base::ElementCNoVoid>>;
    if constexpr (is_fp8_gemm && is_sm89) {
      using AlignmentC_Type = cute::Int<8>;
      using ElementCNoVoid = typename Base::ElementCNoVoid;
      using ElementAccumulator = typename Base::ElementAccumulator;
      using SM89Epilogue = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
          cutlass::epilogue::thread::Identity,  // maybe not need this, so use Identity
          ElementCNoVoid,
          ElementCNoVoid,
          AlignmentC_Type{},
          ElementAccumulator,
          ElementAccumulator>;
      return gemm_v2_impl::KernelParams<TBSwizzle, AlignmentC_Type, SM89Epilogue>();
    } else {
      constexpr bool fuse_reduction = meta.comm_spec().fuse_reduction();
      constexpr int fused_align_size =
          std::is_same_v<typename Base::ElementCNoVoid, cutlass::half_t> ? 32 : 16;
      constexpr int align_c = (fuse_reduction ? fused_align_size : 128) /
                              cute::sizeof_bits_v<typename Base::ElementCNoVoid>;
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

    return this->default_gemm_kernel(params);
  }

  auto
  custom_gemm_device() const {
    using GemmKernel = decltype(gemm_kernel());

    return make_declval<cutlass::gemm::device::GemmUniversalBase<
        FluxGemmKernel<GemmMetaT, GemmHParamsT, GemmKernel>>>();
  }

  template <class... Ts>
  auto
  custom_evt_d(gemm_v2_impl::KernelParams<Ts...> params) const {
    using namespace cutlass::epilogue::threadblock;
    constexpr bool no_nvlink = rs_meta.comm_kind() == _IntraNodePcie{};

    return this->default_evt_d(params);
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
        typename Base::StrideC{}, cute::make_shape(args.m, args.n, cute::Int<1>{}));
    auto stride_D = stride_C;

    using EVT = identity_t<decltype(this->kernel_params().evt())>;
    static_assert(tree_visitor_size<EVT> == 2);
    using EvtArgumentsType = typename EVT::Arguments;
    using EvtDArgumentType = typename tree_node_type<1, EVT>::Arguments;
    using EvtStoreDArgumentType = typename tree_node_type<0, EVT>::Arguments;
    constexpr bool no_nvlink = rs_meta.comm_kind() == _IntraNodePcie{};
    auto evt_d_args = [&]() {
      if constexpr (has_bias) {
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
      if constexpr (rs_meta.comm_kind() == _AcrossNode{}) {
        return EvtStoreDArgumentType{
            ptr_scatter_D,
            stride_D,
            args.rank,
            args.world_size,
            ptr_barrier,
            (typename Base::ElementD *)args.local_reduce_buffer,
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

    return gemm_args;
  }

  auto
  to_fp8_gemm_args_impl(GemmReduceScatterFp8Arguments const &args, void *args_workspace) const {
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

    using EVT = identity_t<decltype(this->kernel_params().evt())>;
    constexpr bool no_nvlink = rs_meta.comm_kind() == _IntraNodePcie{};

    typename EVT::Params epilogue_params{
        {ElementD(args.alpha), ElementD(args.beta)},
        args.scaleA,
        args.scaleB,
        args.scaleC,
        args.scaleD,
        args.scaleAux,
        args.abs_max_Aux,
        args.abs_max_D};  // TODO(houqi.1993)

    auto gemm_args = GemmArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {args.m, args.n, args.k},
        /* batch_count = */ 1,
        epilogue_params,
        args.input,
        args.weight,
        args.bias,                            // ptr_C not used
        args.output_scatter_ptrs[args.rank],  // ptr_D not used
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

    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    if constexpr (is_sm89 && is_fp8_gemm) {
      return to_fp8_gemm_args_impl(
          std::any_cast<GemmReduceScatterFp8Arguments>(args), args_workspace);
    } else {
      return to_gemm_args_impl(std::any_cast<GemmReduceScatterArguments>(args), args_workspace);
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
    return 0;
  }

  void
  initialize_args_workspace(
      std::any const &args,
      void *args_workspace = nullptr,
      void *stream = nullptr) const override {}

 private:
  auto
  to_argument_type(std::any const &args) const {
    if constexpr (is_fp8_gemm && is_sm89) {
      return std::any_cast<GemmReduceScatterFp8Arguments>(args);
    } else {
      return std::any_cast<GemmReduceScatterArguments>(args);
    }
  }
};

using namespace cute;
struct GemmV2ReduceScatter_Space : OpSpaceBase<GemmV2ReduceScatter_Space> {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(
          _FP16{},
          _BF16{},
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}),
      cute::make_tuple(_ReduceScatter{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV2{}),
      cute::make_tuple(None{}),
      cute::make_tuple(
          make_reduce_scatter_meta(_False{}, _IntraNode{}),
          make_reduce_scatter_meta(_True{}, _IntraNode{}),
          make_reduce_scatter_meta(_False{}, _AcrossNode{}),
          make_reduce_scatter_meta(_True{}, _AcrossNode{})));

  static constexpr auto AllGemmHParams = make_space_gemm_hparams(
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(
          Shape<_128, _128, _32>{}, Shape<_128, _128, _64>{}, Shape<_128, _256, _32>{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(cute::_3{}, cute::_4{}));
};
}  // namespace bytedance::flux
