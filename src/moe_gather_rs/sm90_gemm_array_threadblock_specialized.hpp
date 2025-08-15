//===- sm90_gemm_array_threadblock_specialized.hpp C++ ---===//
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
// clang-format off
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/numeric_conversion.h"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "flux/cuda/memory_utils.hpp"
#include "flux/flux.h"
#include "moe_gather_rs/sm90_group_tile_scheduler_threadblock_specialized.hpp"
///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_,
    const int TOPK_,
    const int GATHER_RS_BM_,
    const int GATHER_RS_BN_,
    const int GAHER_RS_N_CTAS_>
class GroupGemmUniversalRSTS {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static_assert(cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, typename CollectiveMainloop_::DispatchPolicy::Schedule>);

  static constexpr bool IsGdcEnabled = false;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using Schedule = typename DispatchPolicy::Schedule;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  template <const int BLOCK_N, typename Element>
  constexpr auto static select_vec_size() {
    constexpr int ELEMENT_SIZE = sizeof(Element);
    if constexpr (BLOCK_N % (32 * 16 / ELEMENT_SIZE) == 0) {
      // 128bit vec size
      return bytedance::flux::make_declval<cute::_16>();
    } else if constexpr (BLOCK_N % (32 * 8 / ELEMENT_SIZE) == 0) {
      // 64bit vec
      return bytedance::flux::make_declval<cute::_8>();
    } else if constexpr (BLOCK_N % (32 * 4 / ELEMENT_SIZE) == 0) {
      return bytedance::flux::make_declval<cute::_4>();
    } else {
      static_assert(cutlass::detail::dependent_false<Element>, "BLOCK_N is to small!");
    }
  }

  static constexpr int TOPK = TOPK_;
  static constexpr int GATHER_RS_BM = GATHER_RS_BM_;
  static constexpr int GATHER_RS_BN = GATHER_RS_BN_;
  static constexpr int GAHER_RS_N_CTAS = GAHER_RS_N_CTAS_;
  static constexpr int N_SMS_90 = 132;
  using ELEMENTD_VEC_SIZE = decltype(select_vec_size<GATHER_RS_BN, ElementD>());
  static constexpr int ElementD_Vec_Size = ELEMENTD_VEC_SIZE{};
  static constexpr int ElementD_Size = sizeof(ElementD);
  static constexpr int ElementD_Vec_N = ElementD_Vec_Size / ElementD_Size;
  static constexpr int GATHER_RS_BN_PER_STAGE =
      32 * ElementD_Vec_N;  // only used for double buffer implementation

  static_assert(ArchTag::kMinComputeCapability >= 90);
  static_assert(
      cute::is_void_v<TileScheduler_>,
      "Ptr-Array Cooperative and Grouped Gemm Cooperative kernel only supports the default "
      "scheduler.");

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  using TileScheduler = cutlass::gemm::kernel::detail::
      PersistentTileSchedulerSm90GroupThreadblockSpecialized<ProblemShape, GAHER_RS_N_CTAS>;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaThreads = CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t NumMmaWarpGroups = NumMmaThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = NumMmaThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;

    struct TensorMapStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorMapStorage = typename CollectiveMainloop::TensorMapStorage;
      using EpilogueTensorMapStorage = typename CollectiveEpilogue::TensorMapStorage;

      alignas(128) MainloopTensorMapStorage mainloop;
      alignas(128) EpilogueTensorMapStorage epilogue;
    } tensormaps;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
    int32_t rank;
    int32_t world_size;
    int32_t topk;
    ElementD **output_scatter_ptrs;
    ElementD **inter_Ds;
    int32_t *routing_idx;
    int32_t *barrier;
    int32_t SPLITS;
    int32_t totalM;
    int32_t n_dim;
    // following args are for epert parallel
    int32_t tp_world_size;
    int32_t ep_world_size;
    int32_t globalM;
    int32_t max_token_per_rank;
    int32_t ep_m_start;
    int32_t ep_m_end;
    float **input_scale_ptr;
    float **output_vec_scale_ptr;
    int32_t input_groups;
    int32_t *ep_pos_filtered;
    int32_t *ep_token_idx_filtered;
    int32_t *ep_total_token_acc;
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void *workspace{nullptr};
    int32_t rank;
    int32_t world_size;
    int32_t topk;
    ElementD **output_scatter_ptrs;
    ElementD **inter_Ds;
    int32_t *routing_idx;
    int32_t *barrier;
    int32_t SPLITS;
    int32_t totalM;
    int32_t n_dim;
    // following args are for epert parallel
    int32_t tp_world_size;
    int32_t ep_world_size;
    int32_t globalM;
    int32_t max_token_per_rank;
    int32_t ep_m_start;
    int32_t ep_m_end;
    float **input_scale_ptr;
    float **output_vec_scale_ptr;
    int32_t input_groups;
    int32_t *ep_pos_filtered;
    int32_t *ep_token_idx_filtered;
    int32_t *ep_total_token_acc;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    ProblemShape problem_shapes = args.problem_shape;

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue, sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveMainloop::get_workspace_size(problem_shapes, args.mainloop, sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);


    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace);
    }
    else {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes.get_host_problem_shape(), TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace);
    }

    return {
        args.mode,
        problem_shapes,
        CollectiveMainloop::to_underlying_arguments(
            problem_shapes, args.mainloop, mainloop_workspace),
        CollectiveEpilogue::to_underlying_arguments(
            problem_shapes, args.epilogue, epilogue_workspace),
        hw_info,
        scheduler,
        workspace,
        args.rank,
        args.world_size,
        args.topk,
        args.output_scatter_ptrs,
        args.inter_Ds,
        args.routing_idx,
        args.barrier,
        args.SPLITS,
        args.totalM,
        args.n_dim,
        args.tp_world_size,
        args.ep_world_size,
        args.globalM,
        args.max_token_per_rank,
        args.ep_m_start,
        args.ep_m_end,
        args.input_scale_ptr,
        args.output_vec_scale_ptr,
        args.input_groups,
        args.ep_pos_filtered,
        args.ep_token_idx_filtered,
        args.ep_total_token_acc};
  }

  static bool
  can_implement(Arguments const &args) {
    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      // Group GEMM currently only supports rank-3 problem shapes
      implementable &= (args.mode == GemmUniversalMode::kGrouped && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3);
    } else {
      implementable &= (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, sm_count);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, sm_count);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveMainloop::initialize_workspace(args.problem_shape, args.mainloop, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    else {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape.get_host_problem_shape(), TileShape{}, ClusterShape{}, params.hw_info, args);
    }

    grid_shape.x += GAHER_RS_N_CTAS;
    return grid_shape;
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  // impl for the double buffered async gather-rs
  template <const int BM, const int BN, const int TOPK, typename Element>
  struct GatherRsSharedStorageDoubleBuffer {
    cute::array_aligned<Element, BM * TOPK * BN> data;
    cute::array_aligned<int32_t, BM * TOPK> routing_idx;
    // int32_t routing_idx[BM * TOPK];
    // Element data[BM * TOPK][BN];
  };

  template <
      const int BLOCK_M,
      const int BLOCK_N,
      const int N_DIM,
      const int TOPK,
      typename Element,
      const int N_THREADS,
      const int VECSIZE>
  CUTLASS_DEVICE void
  gather_rs_impl_async_double_buffer(
      Params const &params, char *smem_buf, int blk_m, int sid, float input_scale) {
    constexpr int vec_bits = VECSIZE * 8;
    using VecType = uint_bit_t<vec_bits>;
    NumericConverter<float, Element, FloatRoundStyle::round_to_nearest> converter1;
    NumericConverter<Element, float, FloatRoundStyle::round_to_nearest> converter2;
    using SmemStorage = GatherRsSharedStorageDoubleBuffer<BLOCK_M, BLOCK_N, TOPK, Element>;
    SmemStorage *smem = reinterpret_cast<SmemStorage *>(smem_buf);
    int smem_data_base[2];
    smem_data_base[0] = __cvta_generic_to_shared(&smem[0].data[0]);
    smem_data_base[1] = __cvta_generic_to_shared(&smem[1].data[0]);
    constexpr int ELEMENT_SIZE = sizeof(Element);
    constexpr int N_REG = VECSIZE / ELEMENT_SIZE;
    // if(threadIdx.x==0){
    //   printf("VEC:%d N_REG:%d \n", VECSIZE, N_REG);
    // }
    float acc[N_REG];
    Element reg128[N_REG];
    constexpr int THREADS_PER_ROW = BLOCK_N / N_REG;
    constexpr int ROW_STRIDE = N_THREADS / THREADS_PER_ROW;
    static_assert(N_THREADS % THREADS_PER_ROW == 0);
    constexpr int ROW_UNROLL = (BLOCK_M + ROW_STRIDE - 1) / ROW_STRIDE;
    constexpr int COL_UNROLL = N_DIM / BLOCK_N;  // also the stages numbers
    constexpr int TOKEN_IDX_N = BLOCK_M * TOPK;
    constexpr int IDX_LOAD_UNROLL = (TOKEN_IDX_N + N_THREADS - 1) / N_THREADS;
    static_assert(COL_UNROLL >= 1);

    int totalM = params.totalM;
    int totalToken = totalM / params.topk;
    int token_per_rank = totalToken / params.world_size;
    int totalN = params.n_dim;
    int new_n_dim = totalN / params.SPLITS;
    int token_offset_start = blk_m * BLOCK_M;
    int token_offset_end = min(token_offset_start + BLOCK_M, totalToken);
    int routing_idx_start = blk_m * BLOCK_M * TOPK;
    int routing_idx_end = min(routing_idx_start + BLOCK_M * TOPK, totalToken * TOPK);
    int remote_token_offset = params.rank * token_per_rank;
    Element *inter_D = params.inter_Ds[0];

    // threadwise offset
    int t_row_offset = threadIdx.x / THREADS_PER_ROW;
    int t_col_offset = threadIdx.x % THREADS_PER_ROW * N_REG;
    int splits_col_offset = sid * new_n_dim;
    int stage_select = 0;
    int stage_next;
    int32_t *smem_routing_idx = &smem[0].routing_idx[0];
    // load the routing index first
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < IDX_LOAD_UNROLL; iter++) {
      int offset = iter * N_THREADS + threadIdx.x;
      if (routing_idx_start + offset < routing_idx_end)
        smem_routing_idx[offset] = params.routing_idx[routing_idx_start + offset];
    }
    __syncthreads();

    CUTLASS_PRAGMA_UNROLL
    for (int row_iter = 0; row_iter < ROW_UNROLL; row_iter++) {
      int row_offset = row_iter * ROW_STRIDE + t_row_offset;
      bool guard = row_offset < (token_offset_end - token_offset_start);
      stage_select = 0;

      // perform the prologue here
      int block_col_offset_select = 0 * BLOCK_N;
      CUTLASS_PRAGMA_UNROLL
      for (int topk = 0; topk < TOPK; topk++) {
        int64_t gmem_row_offset = smem_routing_idx[topk + row_offset * TOPK];
        int64_t load_offset =
            gmem_row_offset * totalN + t_col_offset + block_col_offset_select + splits_col_offset;
        int load_s_addr =
            smem_data_base[stage_select] +
            sizeof(Element) * OFFSET(row_offset * TOPK + topk, t_col_offset, BLOCK_N);
        cutlass::arch::async_load<VECSIZE>(load_s_addr, inter_D + load_offset, guard);
      }

      asm("cp.async.commit_group;\n" ::);
      asm("cp.async.wait_group 0;\n" ::);

      CUTLASS_PRAGMA_UNROLL
      for (int col_iter = 1; col_iter < COL_UNROLL; col_iter++) {
        // clear the acc array
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N_REG; i++) {
          acc[i] = 0;
        }
        stage_select = (col_iter & 1) ^ 1;
        stage_next = stage_select ^ 1;
        // issue the async load instruction first to hide the writing latency
        int block_col_offset_next = col_iter * BLOCK_N;
        int block_col_offset_select = (col_iter - 1) * BLOCK_N;
        CUTLASS_PRAGMA_UNROLL
        for (int topk = 0; topk < TOPK; topk++) {
          int64_t gmem_row_offset = smem_routing_idx[topk + row_offset * TOPK];
          int64_t load_offset =
              gmem_row_offset * totalN + t_col_offset + block_col_offset_next + splits_col_offset;
          int load_s_addr =
              smem_data_base[stage_next] +
              sizeof(Element) * OFFSET(row_offset * TOPK + topk, t_col_offset, BLOCK_N);
          cutlass::arch::async_load<VECSIZE>(load_s_addr, inter_D + load_offset, guard);
        }
        // perform the top-k reduction and writing here
        CUTLASS_PRAGMA_UNROLL
        for (int topk = 0; topk < TOPK; topk++) {
          *reinterpret_cast<VecType *>(&reg128[0]) = *reinterpret_cast<VecType *>(
              &smem[stage_select].data[(row_offset * TOPK + topk) * BLOCK_N + t_col_offset]);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < N_REG; i++) {
            acc[i] += converter1(reg128[i]);
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N_REG; i++) {
          reg128[i] = converter2(acc[i] * input_scale);
        }
        // write back to the taget rank
        int token_idx = token_offset_start + row_offset;
        int dst_rank = token_idx / token_per_rank;
        int64_t remote_row = token_idx % token_per_rank + remote_token_offset;
        int64_t write_offset =
            remote_row * totalN + t_col_offset + block_col_offset_select + splits_col_offset;
        cutlass::arch::global_store<VecType, sizeof(VecType)>(
            *reinterpret_cast<VecType *>(&reg128[0]),
            (void *)(params.output_scatter_ptrs[dst_rank] + write_offset),
            guard);

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);

        if constexpr (THREADS_PER_ROW != 32)
          __syncthreads();
      }
      // perform the gather-rs computation and writing
      // for the last col iteration
      // clear the acc array
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N_REG; i++) {
        acc[i] = 0;
      }
      stage_select = (COL_UNROLL & 1) ^ 1;
      block_col_offset_select = (COL_UNROLL - 1) * BLOCK_N;
      CUTLASS_PRAGMA_UNROLL
      for (int topk = 0; topk < TOPK; topk++) {
        *reinterpret_cast<VecType *>(&reg128[0]) = *reinterpret_cast<VecType *>(
            &smem[stage_select].data[(row_offset * TOPK + topk) * BLOCK_N + t_col_offset]);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N_REG; i++) {
          acc[i] += converter1(reg128[i]);
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N_REG; i++) {
        reg128[i] = converter2(acc[i] * input_scale);
      }
      // write back to the taget rank
      int token_idx = token_offset_start + row_offset;
      int dst_rank = token_idx / token_per_rank;
      int64_t remote_row = token_idx % token_per_rank + remote_token_offset;
      int64_t write_offset =
          remote_row * totalN + t_col_offset + block_col_offset_select + splits_col_offset;
      cutlass::arch::global_store<VecType, sizeof(VecType)>(
          *reinterpret_cast<VecType *>(&reg128[0]),
          (void *)(params.output_scatter_ptrs[dst_rank] + write_offset),
          guard);
    }
    __syncthreads();
  }

  CUTLASS_DEVICE
  void
  tp_topk_gather_rs_dma_double_buffer(
      Params const &params,
      char *smem_buf,
      int split_start,
      int split_end,
      int dma_blk_idx,
      int dma_blks) {
    /*
    Note: Double buffer only supports one input group of gemmgroup,
    and does not support the row-wise output scale(can be supported in the
    future if necessary).
    */
    using Barrier = cutlass::detail::SystemBarrier;
    // perform the reduction with float
    // constexpr int SHM_SIZE_PER_STAGE = sizeof(
    //     GatherRsSharedStorageDoubleBuffer<GATHER_RS_BM, GATHER_RS_BN_PER_STAGE, TOPK, ElementD>);
    // constexpr int Gather_RS_Stages = sizeof(SharedStorage) / SHM_SIZE_PER_STAGE;
    // static_assert(Gather_RS_Stages > 1); //already checked @ the dispatch function
    float input_scale = 1.0;
    if (params.input_scale_ptr != nullptr) {
      input_scale = *params.input_scale_ptr[0];
    }
    int32_t *barrier = params.barrier;
    int rank = params.rank;
    int world_size = params.world_size;
    int wait_flag = N_SMS_90 - GAHER_RS_N_CTAS;

    int32_t dma_block_idx = dma_blk_idx;
    int totalM = params.totalM;
    int totalToken = totalM / params.topk;
    int token_per_rank = totalToken / params.world_size;
    int n_dim = params.n_dim;
    const int new_n_dim = n_dim / params.SPLITS;
    int token_M_blk_count = (totalToken + GATHER_RS_BM - 1) / GATHER_RS_BM;
    int token_M_blks_per_rank = token_M_blk_count / world_size;
    int token_M_blk_offset = rank * token_M_blks_per_rank;
    constexpr int N_THREADS = MaxThreadsPerBlock;
    CUTLASS_PRAGMA_NO_UNROLL
    for (int sid = split_start; sid < split_end; sid++) {
      Barrier::wait_eq(barrier, threadIdx.x, sid, wait_flag);
      for (int blk_m = dma_block_idx; blk_m < token_M_blk_count; blk_m += dma_blks) {
        int swizzled_m = (blk_m + token_M_blk_offset) % token_M_blk_count;
        // __syncthreads();
        gather_rs_impl_async_double_buffer<
            GATHER_RS_BM,
            GATHER_RS_BN_PER_STAGE,
            GATHER_RS_BN,
            TOPK,
            ElementD,
            N_THREADS,
            ElementD_Vec_Size>(params, smem_buf, swizzled_m, sid, input_scale);
      }
    }
  }
  template <const int BLOCK_M, const int BLOCK_N, const int TOPK>
  struct EP_Shared_Storage {
    float smem_data[BLOCK_M][BLOCK_N];
    int smem_idx[BLOCK_M * TOPK];
    int smem_token_idx[BLOCK_M];
  };
  /*
  ///////////////////////////////////////////////////////////////////
  template <
      const int BLOCK_M,
      const int BLOCK_N,
      const int TOPK,
      typename Element,
      const int N_THREADS,
      const bool HAS_VEC_SCALE,
      const int VecSize,
      const int INPUT_GROUPS>
  __device__ void
  ep_gather_rs_impl_v2(
      Params const &params,
      void *smem_buf,
      int blk_m,
      int blk_n,
      int sid,
      int ep_m_start,
      int ep_m_end,
      int token_cur_ep_rank) {
    NumericConverter<float, Element, FloatRoundStyle::round_to_nearest> converter1;
    NumericConverter<Element, float, FloatRoundStyle::round_to_nearest> converter2;
    float input_scales[INPUT_GROUPS];
    float *output_vec_scales[INPUT_GROUPS];
    Element *reg_inter_D[INPUT_GROUPS];
    CUTLASS_PRAGMA_UNROLL
    for (int gid = 0; gid < INPUT_GROUPS; gid++) {
      reg_inter_D[gid] = params.inter_Ds[gid];
      input_scales[gid] = 1.0;
    }
    if (params.input_scale_ptr != nullptr) {
      for (int gid = 0; gid < INPUT_GROUPS; gid++) {
        input_scales[gid] = *params.input_scale_ptr[gid];
      }
    }
    if (params.output_vec_scale_ptr != nullptr) {
      for (int gid = 0; gid < INPUT_GROUPS; gid++) {
        output_vec_scales[gid] = params.output_vec_scale_ptr[gid];
      }
    }

    auto storage = reinterpret_cast<EP_Shared_Storage<BLOCK_M, BLOCK_N, TOPK> *>(smem_buf);
    int32_t *smem_idx = &storage->smem_idx[0];
    float *smem_data = &storage->smem_data[0][0];
    int32_t *smem_token_idx = &storage->smem_token_idx[0];
    constexpr int ELEMENT_SIZE = sizeof(Element);

    constexpr int h_vec_bits = VecSize * 8;
    constexpr int f_vec_bits = VecSize * 8 * 2;
    static_assert(f_vec_bits <= 128);
    using H_VecType = uint_bit_t<h_vec_bits>;
    using F_VecType = uint_bit_t<f_vec_bits>;
    constexpr int N_REG = VecSize / ELEMENT_SIZE;
    static_assert(BLOCK_N % (32 * N_REG) == 0);

    float acc[N_REG];
    Element reg128[N_REG];
    int wid = threadIdx.x / 32;
    int wtid = threadIdx.x % 32;
    constexpr int N_WARPS = N_THREADS / 32;  // each warp is responsible for a row
    constexpr int UNROLL_M = (BLOCK_M + N_WARPS - 1) / N_WARPS;
    constexpr int UNROLL_N = BLOCK_N / 32 / N_REG;
    // load the routing_idx first to the shared memory
    // constexpr int TOKEN_IDX_N = BLOCK_M * TOPK;
    // constexpr int IDX_LOAD_UNROLL = (TOKEN_IDX_N + N_THREADS - 1) / N_THREADS;
    int globalM = params.globalM;
    int totalToken = globalM / params.topk;
    int token_per_rank = totalToken / params.world_size;
    int64_t NDim = params.n_dim;
    int new_n_dim = NDim / params.SPLITS;

    int token_offset_start = blk_m * BLOCK_M;
    int token_offset_end = min(token_offset_start + BLOCK_M, token_cur_ep_rank);
    Element **output_scatter_ptrs = reinterpret_cast<Element **>(params.output_scatter_ptrs);
    static_assert(N_THREADS >= BLOCK_M);
    if (threadIdx.x + token_offset_start < token_offset_end) {
      smem_token_idx[threadIdx.x] = params.ep_token_idx_filtered[token_offset_start + threadIdx.x];
    }
    __syncthreads();

#pragma unroll
    for (int row_iter = 0; row_iter < UNROLL_M; row_iter++) {
      int row_offset = row_iter * N_WARPS + wid;
      if (row_offset < token_offset_end - token_offset_start) {
        int global_token_idx = smem_token_idx[row_offset];
        int topk_count = params.ep_pos_filtered[global_token_idx * (TOPK + 1) + TOPK];
        // manually unroll the topk-count
        int gmem_row_offset_0 = params.ep_pos_filtered[global_token_idx * (TOPK + 1)];

        float vec_scale_0 = 1.0;
        if constexpr (HAS_VEC_SCALE) {
          vec_scale_0 = output_vec_scales[0][gmem_row_offset_0 - ep_m_start];
        }
#pragma unroll
        for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
          int smem_col_offset = col_iter * 32 * N_REG + wtid * N_REG;
          int smem_row_offset = row_offset;
          int gmem_col_offset = smem_col_offset + sid * new_n_dim + blk_n * BLOCK_N;
          int64_t load_offset = (gmem_row_offset_0 - ep_m_start) * NDim + gmem_col_offset;
          *reinterpret_cast<H_VecType *>(&reg128[0]) = *reinterpret_cast<H_VecType *>(
              reg_inter_D[0] + load_offset);
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            if constexpr (HAS_VEC_SCALE) {
              acc[i] = converter1(reg128[i]) * vec_scale_0 * input_scales[0];
            } else {
              acc[i] = converter1(reg128[i]) * input_scales[0];
            }
          }
          *reinterpret_cast<F_VecType *>(smem_data + smem_row_offset * BLOCK_N + smem_col_offset) =
              *reinterpret_cast<F_VecType *>(&acc[0]);
        }

#pragma unroll
        for (int input_gid = 1; input_gid < INPUT_GROUPS; input_gid++) {
          float vec_scale = 1.0;
          if constexpr (HAS_VEC_SCALE) {
            vec_scale = output_vec_scales[input_gid][gmem_row_offset_0 - ep_m_start];
          }
#pragma unroll
          for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
            int smem_col_offset = col_iter * 32 * N_REG + wtid * N_REG;
            int smem_row_offset = row_offset;
            int gmem_col_offset = smem_col_offset + sid * new_n_dim + blk_n * BLOCK_N;
            int64_t load_offset = (gmem_row_offset_0 - ep_m_start) * NDim + gmem_col_offset;
            *reinterpret_cast<H_VecType *>(&reg128[0]) = *reinterpret_cast<H_VecType *>(
                reg_inter_D[input_gid] + load_offset);
            *reinterpret_cast<F_VecType *>(&acc[0]) = *reinterpret_cast<F_VecType *>(
                smem_data + smem_row_offset * BLOCK_N + smem_col_offset);
#pragma unroll
            for (int i = 0; i < N_REG; i++) {
              if constexpr (HAS_VEC_SCALE) {
                acc[i] += converter1(reg128[i]) * vec_scale * input_scales[input_gid];
              } else {
                acc[i] += converter1(reg128[i]) * input_scales[input_gid];
              }
            }
            *reinterpret_cast<F_VecType *>(
                smem_data + smem_row_offset * BLOCK_N + smem_col_offset) =
                *reinterpret_cast<F_VecType *>(&acc[0]);
          }
        }

#pragma unroll
        for (int topk = 1; topk < topk_count; topk++) {
          int gmem_row_offset = params.ep_pos_filtered[global_token_idx * (TOPK + 1) + topk];

#pragma unroll
          for (int input_gid = 0; input_gid < INPUT_GROUPS; input_gid++) {
            float vec_scale = 1.0;
            if constexpr (HAS_VEC_SCALE) {
              vec_scale = output_vec_scales[input_gid][gmem_row_offset - ep_m_start];
            }
#pragma unroll
            for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
              int smem_col_offset = col_iter * 32 * N_REG + wtid * N_REG;
              int smem_row_offset = row_offset;
              int gmem_col_offset = smem_col_offset + sid * new_n_dim + blk_n * BLOCK_N;
              int64_t load_offset = (gmem_row_offset - ep_m_start) * NDim + gmem_col_offset;
              *reinterpret_cast<H_VecType *>(&reg128[0]) = *reinterpret_cast<H_VecType *>(
                  reg_inter_D[input_gid] + load_offset);
              *reinterpret_cast<F_VecType *>(&acc[0]) = *reinterpret_cast<F_VecType *>(
                  smem_data + smem_row_offset * BLOCK_N + smem_col_offset);
#pragma unroll
              for (int i = 0; i < N_REG; i++) {
                if constexpr (HAS_VEC_SCALE) {
                  acc[i] += converter1(reg128[i]) * vec_scale * input_scales[input_gid];
                } else {
                  acc[i] += converter1(reg128[i]) *
                            input_scales[input_gid];  // TODO merge input_scale into weight scale
                }
              }
              *reinterpret_cast<F_VecType *>(
                  smem_data + smem_row_offset * BLOCK_N + smem_col_offset) =
                  *reinterpret_cast<F_VecType *>(&acc[0]);
            }
          }
        }

        // write the result back to the global memory
        int token_idx = global_token_idx;
        int dst_rank = token_idx / token_per_rank;
        int remote_row = token_idx % token_per_rank;

        constexpr int N_REG_WRITE = 2;
        static_assert(N_REG >= N_REG_WRITE);
        constexpr int UNROLL_WRITE = BLOCK_N / 32 / N_REG_WRITE;
#pragma unroll
        for (int col_iter = 0; col_iter < UNROLL_WRITE; col_iter++) {
          int smem_col_offset = col_iter * 32 * N_REG_WRITE + wtid * N_REG_WRITE;
          int gmem_col_offset = smem_col_offset + sid * new_n_dim + blk_n * BLOCK_N;
          FETCH_64bit(&acc[0]) =
              *reinterpret_cast<float2 *>(smem_data + row_offset * BLOCK_N + smem_col_offset);
#pragma unroll
          for (int i = 0; i < N_REG_WRITE; i++) {
            reg128[i] = converter2(acc[i]);
          }
          auto val = *reinterpret_cast<uint32_t *>(&reg128[0]);
          using W_VecType = uint_bit_t<32>;
          int64_t write_offset = remote_row * NDim + gmem_col_offset;
          cutlass::arch::global_red<W_VecType, sizeof(W_VecType), Element>(
              val, output_scatter_ptrs[dst_rank] + write_offset, 1);
        }
      }
    }
  }

  CUTLASS_DEVICE void
  ep_topk_gather_rs_dma_v2(
      Params const &params,
      char *smem_buf,
      int split_start,
      int split_end,
      int dma_blk_idx,
      int dma_blks) {
    // using BarrierSync = cutlass::detail::SyncthreadsSync;
    using Barrier = cutlass::detail::SystemBarrier;
    // perform the reduction with float
    // extern __shared__ char smem_buf[];
    int32_t ep_m_start = params.ep_m_start;
    int32_t ep_m_end = params.ep_m_end;
    int32_t *barrier = params.barrier;
    int rank = params.rank;
    int world_size = params.world_size;
    static_assert(
        sizeof(SharedStorage) > sizeof(EP_Shared_Storage<GATHER_RS_BM, GATHER_RS_BN, TOPK>));

    bool has_output_vec_scale = params.output_vec_scale_ptr != nullptr;
    constexpr int N_THREADS = MaxThreadsPerBlock;
    int wait_flag = N_SMS_90 - GAHER_RS_N_CTAS;
    int32_t dma_block_idx = dma_blk_idx;
    int globalM = params.globalM;  // M for all experts
    // int totalToken = globalM / TOPK;
    int n_dim = params.n_dim;
    const int new_n_dim = n_dim / params.SPLITS;  // BUG FIX ME
    int token_cur_ep_rank = *params.ep_total_token_acc;
    int token_M_blk_count = (token_cur_ep_rank + GATHER_RS_BM - 1) / GATHER_RS_BM;
    int token_M_blks_per_rank = token_M_blk_count / world_size;
    int token_M_blk_offset = rank * token_M_blks_per_rank;
    constexpr int EP_VEC_SIZE = ElementD_Vec_Size > 8 ? ElementD_Vec_Size / 2 : ElementD_Vec_Size;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int sid = split_start; sid < split_end; sid++) {
      Barrier::wait_eq(barrier, threadIdx.x, sid, wait_flag);
      for (int blk_m = dma_block_idx; blk_m < token_M_blk_count; blk_m += dma_blks) {
        int swizzled_m = (blk_m + token_M_blk_offset) % token_M_blk_count;
        for (int blk_n = 0; blk_n < new_n_dim / GATHER_RS_BN; blk_n++) {
          __syncthreads();
          if (!has_output_vec_scale) {
            if (params.input_groups == 1) {
              ep_gather_rs_impl_v2<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  false,
                  EP_VEC_SIZE,
                  1>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            } else if (params.input_groups == 2) {
              ep_gather_rs_impl_v2<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  false,
                  EP_VEC_SIZE,
                  2>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            }
          } else {
            if (params.input_groups == 1) {
              ep_gather_rs_impl_v2<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  true,
                  EP_VEC_SIZE,
                  1>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            } else if (params.input_groups == 2) {
              ep_gather_rs_impl_v2<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  true,
                  EP_VEC_SIZE,
                  2>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            }
          }
        }
      }
    }
  }
  */
  template <const int BLOCK_M, const int BLOCK_N, const int TOPK>
  struct EP_Shared_StorageV3 {
    int smem_idx[BLOCK_M * TOPK];
    int smem_token_idx[BLOCK_M];
    int smem_local_token_idx[BLOCK_M];
    int smem_tgt_rank[BLOCK_M];
    int topks[BLOCK_M];
  };

  template <
      const int BLOCK_M,
      const int BLOCK_N,
      const int TOPK,
      typename Element,
      const int N_THREADS,
      const bool HAS_VEC_SCALE,
      const int VecSize,
      const int INPUT_GROUPS>
  __device__ void
  ep_gather_rs_impl_v3(
      Params const &params,
      void *smem_buf,
      int blk_m,
      int blk_n,
      int sid,
      int ep_m_start,
      int ep_m_end,
      int token_cur_ep_rank) {
    NumericConverter<float, Element, FloatRoundStyle::round_to_nearest> converter1;
    NumericConverter<Element, float, FloatRoundStyle::round_to_nearest> converter2;
    float input_scales[INPUT_GROUPS];
    float *output_vec_scales[INPUT_GROUPS];
    Element *reg_inter_D[INPUT_GROUPS];
    CUTLASS_PRAGMA_UNROLL
    for (int gid = 0; gid < INPUT_GROUPS; gid++) {
      reg_inter_D[gid] = params.inter_Ds[gid];
      input_scales[gid] = 1.0;
    }
    if (params.input_scale_ptr != nullptr) {
      for (int gid = 0; gid < INPUT_GROUPS; gid++) {
        input_scales[gid] = *params.input_scale_ptr[gid];
      }
    }
    if (params.output_vec_scale_ptr != nullptr) {
      for (int gid = 0; gid < INPUT_GROUPS; gid++) {
        output_vec_scales[gid] = params.output_vec_scale_ptr[gid];
      }
    }

    auto storage = reinterpret_cast<EP_Shared_StorageV3<BLOCK_M, BLOCK_N, TOPK> *>(smem_buf);
    int32_t *smem_idx = &storage->smem_idx[0];
    int32_t *smem_topk_count = &storage->topks[0];
    int32_t *smem_token_idx = &storage->smem_token_idx[0];
    int32_t *smem_local_token_idx = &storage->smem_local_token_idx[0];
    int32_t *smem_tgt_rank = &storage->smem_tgt_rank[0];
    constexpr int ELEMENT_SIZE = sizeof(Element);
    constexpr int N_REG = VecSize / ELEMENT_SIZE;
    static_assert(BLOCK_N % (32 * N_REG) == 0);
    constexpr int vec_bits = VecSize * 8;
    using VecType = uint_bit_t<cute::min(128, vec_bits)>;

    float acc[N_REG];
    Element reg128[N_REG];
    int wid = threadIdx.x / 32;
    int wtid = threadIdx.x % 32;
    constexpr int N_WARPS = N_THREADS / 32;  // each warp is responsible for a row
    constexpr int UNROLL_M = (BLOCK_M + N_WARPS - 1) / N_WARPS;
    constexpr int UNROLL_N = BLOCK_N / 32 / N_REG;
    // constexpr int TOKEN_IDX_N = BLOCK_M * TOPK;
    // constexpr int IDX_LOAD_UNROLL = (TOKEN_IDX_N + N_THREADS - 1) / N_THREADS;
    int globalM = params.globalM;
    int totalToken = globalM / params.topk;
    int64_t NDim = params.n_dim;
    int64_t new_n_dim = NDim / params.SPLITS;
    int remote_token_offset = params.rank * params.max_token_per_rank;
    int token_offset_start = blk_m * BLOCK_M;
    int token_offset_end = min(token_offset_start + BLOCK_M, token_cur_ep_rank);
    Element **output_scatter_ptrs = reinterpret_cast<Element **>(params.output_scatter_ptrs);
    static_assert(N_THREADS >= BLOCK_M);
    if (threadIdx.x + token_offset_start < token_offset_end) {
      int global_token_idx = params.ep_token_idx_filtered[token_offset_start + threadIdx.x];
      smem_token_idx[threadIdx.x] = global_token_idx;
      int topk_count = params.ep_pos_filtered[global_token_idx * (TOPK + 3) + TOPK];
      smem_topk_count[threadIdx.x] = topk_count;
      int local_idx_in_tgt_rank = params.ep_pos_filtered[global_token_idx * (TOPK + 3) + TOPK + 1];
      smem_local_token_idx[threadIdx.x] = local_idx_in_tgt_rank;
      int tgt_rank =params.ep_pos_filtered[global_token_idx * (TOPK + 3) + TOPK + 2];
      smem_tgt_rank[threadIdx.x] = tgt_rank;
      for (int i = 0; i < topk_count; i++) {
        int routed_pos = params.ep_pos_filtered[global_token_idx * (TOPK + 3) + i];
        smem_idx[threadIdx.x * TOPK + i] = routed_pos;
      }
    }
    __syncthreads();

#pragma unroll
    for (int row_iter = 0; row_iter < UNROLL_M; row_iter++) {
      int row_offset = row_iter * N_WARPS + wid;
      if (row_offset < token_offset_end - token_offset_start) {
        int cur_topk = smem_topk_count[row_offset];
        int local_token_idx_in_tgt = smem_local_token_idx[row_offset];
        int dst_rank = smem_tgt_rank[row_offset];
#pragma unroll
        for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
          int col_offset = col_iter * 32 * N_REG + wtid * N_REG;
          int gmem_col_offset = col_offset + sid * new_n_dim + blk_n * BLOCK_N;
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            acc[i] = 0;
          }
#pragma unroll
          for (int input_gid = 0; input_gid < INPUT_GROUPS; input_gid++) {
#pragma unroll
            for (int topk = 0; topk < cur_topk; topk++) {
              int64_t gmem_row_offset = smem_idx[topk + row_offset * TOPK] - ep_m_start;
              float vec_scale = 1.0;
              if constexpr (HAS_VEC_SCALE) {
                vec_scale = output_vec_scales[input_gid][gmem_row_offset];
              }
              *reinterpret_cast<VecType *>(&reg128[0]) = *reinterpret_cast<VecType *>(
                  reg_inter_D[input_gid] + gmem_row_offset * NDim + gmem_col_offset);
#pragma unroll
              for (int i = 0; i < N_REG; i++) {
                if constexpr (HAS_VEC_SCALE) {
                  acc[i] += converter1(reg128[i]) * input_scales[input_gid] * vec_scale;
                } else {
                  acc[i] += converter1(reg128[i]) * input_scales[input_gid];
                }
              }
              // if (token_idx == 0 && gmem_col_offset == 0) {
              //   printf("rank:%d acc[0]:%f cur_topk:%d topk:%d gmem_row_offset:%d\n",
              //   params.rank, acc[0], cur_topk, topk, gmem_row_offset);
              // }
            }
          }
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            reg128[i] = converter2(acc[i]);
          }
          // write back to the taget rank

          int64_t remote_row = local_token_idx_in_tgt + remote_token_offset;
          *reinterpret_cast<VecType *>(
              output_scatter_ptrs[dst_rank] + remote_row * NDim + gmem_col_offset) =
              *reinterpret_cast<VecType *>(&reg128[0]);
        }
      }
    }
  }
  CUTLASS_DEVICE void
  ep_topk_gather_rs_dma_v3(
      Params const &params,
      char *smem_buf,
      int split_start,
      int split_end,
      int dma_blk_idx,
      int dma_blks) {
    // using BarrierSync = cutlass::detail::SyncthreadsSync;
    using Barrier = cutlass::detail::SystemBarrier;
    // perform the reduction with float
    // extern __shared__ char smem_buf[];
    int32_t ep_m_start = params.ep_m_start;
    int32_t ep_m_end = params.ep_m_end;
    int32_t *barrier = params.barrier;
    int rank = params.rank;
    int world_size = params.world_size;
    static_assert(
        sizeof(SharedStorage) > sizeof(EP_Shared_StorageV3<GATHER_RS_BM, GATHER_RS_BN, TOPK>));

    bool has_output_vec_scale = params.output_vec_scale_ptr != nullptr;
    constexpr int N_THREADS = MaxThreadsPerBlock;
    int wait_flag = N_SMS_90 - GAHER_RS_N_CTAS;
    int32_t dma_block_idx = dma_blk_idx;
    int globalM = params.globalM;  // M for all experts
    // int totalToken = globalM / TOPK;
    int n_dim = params.n_dim;
    const int new_n_dim = n_dim / params.SPLITS;  // BUG FIX ME
    int token_cur_ep_rank = *params.ep_total_token_acc;
    int token_M_blk_count = (token_cur_ep_rank + GATHER_RS_BM - 1) / GATHER_RS_BM;
    int token_M_blks_per_rank = token_M_blk_count / world_size;
    int token_M_blk_offset = rank * token_M_blks_per_rank;
    constexpr int EP_VEC_SIZE = ElementD_Vec_Size;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int sid = split_start; sid < split_end; sid++) {
      Barrier::wait_eq(barrier, threadIdx.x, sid, wait_flag);
      for (int blk_m = dma_block_idx; blk_m < token_M_blk_count; blk_m += dma_blks) {
        int swizzled_m = (blk_m + token_M_blk_offset) % token_M_blk_count;
        for (int blk_n = 0; blk_n < new_n_dim / GATHER_RS_BN; blk_n++) {
          __syncthreads();
          if (!has_output_vec_scale) {
            if (params.input_groups == 1) {
              ep_gather_rs_impl_v3<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  false,
                  EP_VEC_SIZE,
                  1>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            } else if (params.input_groups == 2) {
              ep_gather_rs_impl_v3<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  false,
                  EP_VEC_SIZE,
                  2>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            }
          } else {
            if (params.input_groups == 1) {
              ep_gather_rs_impl_v3<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  true,
                  EP_VEC_SIZE,
                  1>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            } else if (params.input_groups == 2) {
              ep_gather_rs_impl_v3<
                  GATHER_RS_BM,
                  GATHER_RS_BN,
                  TOPK,
                  ElementD,
                  N_THREADS,
                  true,
                  EP_VEC_SIZE,
                  2>(
                  params,
                  smem_buf,
                  swizzled_m,
                  blk_n,
                  sid,
                  ep_m_start,
                  ep_m_end,
                  token_cur_ep_rank);
            }
          }
        }
      }
    }
  }

  ///////////////////////////////////////////////////////////////////
  template <const int BLOCK_M, const int BLOCK_N, const int TOPK>
  struct TP_Shared_Storage {
    int smem_idx[BLOCK_M * TOPK];
  };

  template <
      const int BLOCK_M,
      const int BLOCK_N,
      const int TOPK,
      typename Element,
      const int N_THREADS,
      const int VECSIZE,
      const int INPUT_GROUPS>
  __device__ void
  tp_gather_rs_impl(Params const &params, void *smem_buf, int blk_m, int blk_n, int sid) {
    float input_scales[INPUT_GROUPS];
    Element *reg_inter_D[INPUT_GROUPS];
    CUTLASS_PRAGMA_UNROLL
    for (int gid = 0; gid < INPUT_GROUPS; gid++) {
      reg_inter_D[gid] = params.inter_Ds[gid];
      input_scales[gid] = 1.0;
    }
    if (params.input_scale_ptr != nullptr) {
      for (int gid = 0; gid < INPUT_GROUPS; gid++) {
        input_scales[gid] = *params.input_scale_ptr[gid];
      }
    }

    NumericConverter<float, Element, FloatRoundStyle::round_to_nearest> converter1;
    NumericConverter<Element, float, FloatRoundStyle::round_to_nearest> converter2;
    int32_t *smem_idx = reinterpret_cast<int32_t *>(smem_buf);
    constexpr int ELEMENT_SIZE = sizeof(Element);
    constexpr int N_REG = VECSIZE / ELEMENT_SIZE;
    float acc[N_REG];
    Element reg128[N_REG];
    constexpr int vec_bits = VECSIZE * 8;
    using VecType = uint_bit_t<cute::min(128, vec_bits)>;
    static_assert(BLOCK_N % (32 * N_REG) == 0);
    int wid = threadIdx.x / 32;
    int wtid = threadIdx.x % 32;
    constexpr int N_WARPS = N_THREADS / 32;  // each warp is responsible for a row
    constexpr int UNROLL_M = (BLOCK_M + N_WARPS - 1) / N_WARPS;
    constexpr int UNROLL_N = BLOCK_N / 32 / N_REG;
    // load the routing_idx first to the shared memory
    constexpr int TOKEN_IDX_N = BLOCK_M * TOPK;
    constexpr int IDX_LOAD_UNROLL = (TOKEN_IDX_N + N_THREADS - 1) / N_THREADS;
    // printf("before params\n");
    int totalM = params.totalM;
    // printf("total M :%d \n", totalM);
    int totalToken = totalM / params.topk;
    int token_per_rank = totalToken / params.world_size;
    int NDim = params.n_dim;
    int new_n_dim = NDim / params.SPLITS;
    int routing_idx_start = blk_m * BLOCK_M * TOPK;
    int routing_idx_end = min(routing_idx_start + BLOCK_M * TOPK, totalToken * TOPK);
    int token_offset_start = blk_m * BLOCK_M;
    int token_offset_end = min(token_offset_start + BLOCK_M, totalToken);

    int remote_token_offset = params.rank * token_per_rank;
    Element **output_scatter_ptrs = reinterpret_cast<Element **>(params.output_scatter_ptrs);
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < IDX_LOAD_UNROLL; iter++) {
      int offset = iter * N_THREADS + threadIdx.x;
      if (routing_idx_start + offset < routing_idx_end) {
        smem_idx[offset] = params.routing_idx[routing_idx_start + offset];
      }
    }
    __syncthreads();
#pragma unroll
    for (int row_iter = 0; row_iter < UNROLL_M; row_iter++) {
      int row_offset = row_iter * N_WARPS + wid;
      if (row_offset < token_offset_end - token_offset_start) {
#pragma unroll
        for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
          int col_offset = col_iter * 32 * N_REG + wtid * N_REG;
          int gmem_col_offset = col_offset + sid * new_n_dim + blk_n * BLOCK_N;
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            acc[i] = 0;
          }

#pragma unroll
          for (int input_gid = 0; input_gid < INPUT_GROUPS; input_gid++) {
#pragma unroll
            for (int topk = 0; topk < TOPK; topk++) {
              int gmem_row_offset = smem_idx[topk + row_offset * TOPK];
              *reinterpret_cast<VecType *>(&reg128[0]) = *reinterpret_cast<VecType *>(
                  reg_inter_D[input_gid] + gmem_row_offset * NDim + gmem_col_offset);
#pragma unroll
              for (int i = 0; i < N_REG; i++) {
                acc[i] += converter1(reg128[i]) * input_scales[input_gid];
              }
            }
          }
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            reg128[i] = converter2(acc[i]);
          }
          // write back to the taget rank
          int token_idx = token_offset_start + row_offset;
          int dst_rank = token_idx / token_per_rank;
          int remote_row = token_idx % token_per_rank + remote_token_offset;
          *reinterpret_cast<VecType *>(
              output_scatter_ptrs[dst_rank] + remote_row * NDim + gmem_col_offset) =
              *reinterpret_cast<VecType *>(&reg128[0]);
        }
      }
    }
  }

  CUTLASS_DEVICE void
  tp_topk_gather_rs_dma(
      Params const &params,
      char *smem_buf,
      int split_start,
      int split_end,
      int dma_blk_idx,
      int dma_blks) {
    // using BarrierSync = cutlass::detail::SyncthreadsSync;
    using Barrier = cutlass::detail::SystemBarrier;
    int32_t *barrier = params.barrier;
    int rank = params.rank;
    int world_size = params.world_size;
    static_assert(
        sizeof(SharedStorage) > sizeof(TP_Shared_Storage<GATHER_RS_BM, GATHER_RS_BN, TOPK>));

    constexpr int N_THREADS = MaxThreadsPerBlock;
    int wait_flag = N_SMS_90 - GAHER_RS_N_CTAS;

    int32_t dma_block_idx = dma_blk_idx;
    int totalM = params.totalM;
    int totalToken = totalM / TOPK;
    int n_dim = params.n_dim;
    const int new_n_dim = n_dim / params.SPLITS;
    int token_M_blk_count = (totalToken + GATHER_RS_BM - 1) / GATHER_RS_BM;
    int token_M_blks_per_rank = token_M_blk_count / world_size;
    int token_M_blk_offset = rank * token_M_blks_per_rank;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int sid = split_start; sid < split_end; sid++) {
      Barrier::wait_eq(barrier, threadIdx.x, sid, wait_flag);
      for (int blk_m = dma_block_idx; blk_m < token_M_blk_count; blk_m += dma_blks) {
        int swizzled_m = (blk_m + token_M_blk_offset) % token_M_blk_count;
        for (int blk_n = 0; blk_n < new_n_dim / GATHER_RS_BN; blk_n++) {
          __syncthreads();
          if (params.input_groups == 2) {
            tp_gather_rs_impl<
                GATHER_RS_BM,
                GATHER_RS_BN,
                TOPK,
                ElementD,
                N_THREADS,
                ElementD_Vec_Size,
                2>(params, smem_buf, swizzled_m, blk_n, sid);
          } else if (params.input_groups == 1) {
            tp_gather_rs_impl<
                GATHER_RS_BM,
                GATHER_RS_BN,
                TOPK,
                ElementD,
                N_THREADS,
                ElementD_Vec_Size,
                1>(params, smem_buf, swizzled_m, blk_n, sid);
          }
        }
      }
    }
  }

  CUTLASS_DEVICE
  void
  operator()(Params const &params, char *smem_buf) {
    using namespace cute;
    using X = Underscore;
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
      // set the scheduled ready flag for the outside gather-rs kernel
      int32_t *barrier = params.barrier;
      barrier[params.SPLITS] = 1;
    }
    constexpr int SHM_SIZE_PER_STAGE = sizeof(
        GatherRsSharedStorageDoubleBuffer<GATHER_RS_BM, GATHER_RS_BN_PER_STAGE, TOPK, ElementD>);
    constexpr int CAN_USE_DOUBLE_BUFFER = (sizeof(SharedStorage) / SHM_SIZE_PER_STAGE) >= 2;
    if (blockIdx.x + GAHER_RS_N_CTAS >= gridDim.x) {
      // perform the topk-gather-reduce-scatter outside of the gemm-kernel
      // gather_rs_dma_async_double_buffer(params, smem_buf, 0, params.SPLITS-1, blockIdx.x
      // GAHER_RS_N_CTAS, GAHER_RS_N_CTAS);
#ifndef MOE_GATHER_RS_SEPARATE_IMPL
      if (params.ep_world_size > 1) {
        ep_topk_gather_rs_dma_v3(
            params, smem_buf, 0, params.SPLITS - 1, blockIdx.x % GAHER_RS_N_CTAS, GAHER_RS_N_CTAS);
      } else {
        if (params.input_groups == 1 && CAN_USE_DOUBLE_BUFFER) {
          // double buffer implementation has better performance and only
          // supports one input group
          tp_topk_gather_rs_dma_double_buffer(
              params,
              smem_buf,
              0,
              params.SPLITS - 1,
              blockIdx.x % GAHER_RS_N_CTAS,
              GAHER_RS_N_CTAS);
        } else {
          tp_topk_gather_rs_dma(
              params,
              smem_buf,
              0,
              params.SPLITS - 1,
              blockIdx.x % GAHER_RS_N_CTAS,
              GAHER_RS_N_CTAS);
        }
      }
#endif
      return;
    }
    int add_val = 1;
    if (blockIdx.x == 0) {
      // in some cases, the number of launched threadblocks is smaller than existing
      // SMs.
      add_val += N_SMS_90 - gridDim.x;
    }
// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
    printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(size(TiledMma{}) == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");
    static_assert(NumMmaWarpGroups == 2, "Cooperative kernels currently only support NumMmaWarpGroups == 2");

    if constexpr (cutlass::epilogue::collective::detail::sm90_is_ptr_array_tma_dispatch_policy_v<typename CollectiveEpilogue::DispatchPolicy>) {
      static_assert(NumMmaWarpGroups == CollectiveEpilogue::NumEpilogueWarpGroups,
                    "Tiled MmA does not match expected warp groups performing the epilogue");
    }

    static_assert(cute::rank(InternalStrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      Warp3 = 3
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_idx = canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    int sid_processed = -1;
    // Note: Tma Descriptor Prefetch (from either const or param) is not applicable here

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = size(TiledMma{});
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = size(TiledMma{});
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = [] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    const auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    const auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    const auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
    if (not work_tile_info.is_valid()) {
      // When problem shapes are only on device, the grid launched may be larger than the total number of blocks across groups
      return;
    }

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx;
        int32_t const mock_l_coord = 0;
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        // Fetch a copy of tensormaps for the CTA
        auto input_tensormaps = collective_mainloop.tensormaps_init(params.mainloop, shared_storage.tensormaps.mainloop, sm_count, sm_idx);

        // Update tensormap for the initial batch for the CTA
        if (work_tile_info.is_valid()) {
          collective_mainloop.tensormaps_perform_update(
            shared_storage.tensormaps.mainloop,
            params.mainloop,
            input_tensormaps,
            problem_shape_MNKL,
            curr_batch
          );
          // Ensure warp is converged before issuing tensormap fence release
          __syncwarp();
          // Entire warp must do this (i.e. it's aligned)
          collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
        }

        bool do_load_order_arrive = true;
        bool did_batch_change = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;
            continue;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, mock_l_coord);

          // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          if (did_batch_change) {
            collective_mainloop.tensormaps_fence_acquire(input_tensormaps);
          }


          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            input_tensormaps,
            blk_coord,
            k_tile_iter, work_k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
          // Update starting pipeline state for the next tile
          // Wait for the last TMA stage to complete loading, before issuing tensormap updates
          mainloop_pipe_producer_state.advance(work_k_tile_count - 1);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
          work_tile_info = next_work_tile_info;
          auto next_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx
          did_batch_change = next_batch != curr_batch;

          if (work_tile_info.is_valid() && did_batch_change) {
            curr_batch = next_batch;
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(curr_batch), 1);
            }
            // Purpose of this pipeline state is to make sure TMA loads have finished before doing descriptor updates
            // Since this state is waiting for loads to finish, it must start in the inverted phase.
            typename CollectiveMainloop::PipelineState mainloop_pipe_tma_consumer_state =
              {mainloop_pipe_producer_state.index(), !mainloop_pipe_producer_state.phase(), mainloop_pipe_producer_state.count()};
            mainloop_pipeline.consumer_wait(mainloop_pipe_tma_consumer_state);
            collective_mainloop.tensormaps_perform_update(
              shared_storage.tensormaps.mainloop,
              params.mainloop,
              input_tensormaps,
              problem_shape_MNKL,
              curr_batch
            );
            // Ensure warp is converged before issuing tensor replace
            __syncwarp();
            // Entire warp must do this (i.e. it's aligned)
            collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
          }
          // Advance the producer state for the last remaining stage that was being waited for above
          mainloop_pipe_producer_state.advance(1);
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      } // Mainloop Producer Warp End

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        auto epi_load_tensormap = get<0>(collective_epilogue.load_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx));

        bool did_batch_change = true;
        constexpr bool IsEpiLoad = true;

        if (work_tile_info.is_valid()) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_load_tensormap,
            problem_shape_MNKL,
            work_tile_info.L_idx,
            0
          );

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
        }

        load_order_barrier.wait();

        while (work_tile_info.is_valid()) {
          int32_t curr_batch = work_tile_info.L_idx;

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }
            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            if (did_batch_change) {
              collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_load_tensormap);
            }

            bool wait = work_tile_info.is_valid() && curr_batch != next_work_tile_info.L_idx;

            epi_load_pipe_producer_state = collective_epilogue.load(
              epi_load_pipeline,
              epi_load_pipe_producer_state,
              problem_shape_MNKL,
              blk_shape,
              blk_coord,
              tiled_mma,
              lane_idx,
              shared_storage.tensors.epilogue,
              epi_load_tensormap,
              work_tile_info.reduction_subtile_idx(),
              wait
            );
          }

          work_tile_info = next_work_tile_info;
          did_batch_change = curr_batch != work_tile_info.L_idx;

          if (work_tile_info.is_valid() && did_batch_change) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // tensormap update
            {
              collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
                shared_storage.tensormaps.epilogue,
                params.epilogue,
                epi_load_tensormap,
                problem_shape_MNKL,
                work_tile_info.L_idx,
                0
              );

              // Converge before issuing tensormap fence release since fence is aligned
              __syncwarp();
              collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
            }
          }

        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Index of warp group within consumer warp groups
      int consumer_warp_group_idx = warp_group_role == WarpGroupRole::Consumer0 ? 0 : 1;

      int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
      int32_t const sm_count = params.hw_info.sm_count;
      // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
      bool do_store_tail = false;
      // Get a copy of tensormaps
      auto epi_store_tensormap = get<0>(collective_epilogue.store_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx, consumer_warp_group_idx));

      bool did_batch_change = true;
      constexpr bool IsEpiLoad = false;

      if (work_tile_info.is_valid()) {

        if (warp_idx_in_warp_group == 0) {
          collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
            shared_storage.tensormaps.epilogue,
            params.epilogue,
            epi_store_tensormap,
            problem_shape_MNKL,
            work_tile_info.L_idx,
            consumer_warp_group_idx
          );

          // Converge before issuing tensormap fence release since fence is aligned
          __syncwarp();
          collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue,
                                                                     epi_store_tensormap,
                                                                     consumer_warp_group_idx);
        }
      }

      using MMABarrierSync = cutlass::detail::NamedBarrierSync<
          size(TiledMma{}),
          (int)bytedance::flux::FluxNamedBarriers::GatherRSProducer>;
      using MMABarrier = cutlass::detail::GenericSystemBarrier<MMABarrierSync>;
      int L_coords_per_splits = params.problem_shape.groups() / params.SPLITS;

      while (work_tile_info.is_valid()) {
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
        }

        int32_t curr_batch = work_tile_info.L_idx;

        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);

        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)

        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {

          collective_mainloop.mma(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            work_k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            params.mainloop
          );

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            work_k_tile_count
          );

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }
        // Perform reduction across splits, if needed
        TileScheduler::fixup(
          params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

        if (did_batch_change) {
          collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_store_tensormap);
        }

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {

          // Epilogue and write to gD
          auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
          collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            accumulators,
            tiled_mma,
            mma_thread_idx,
            shared_storage.tensors.epilogue,
            epi_store_tensormap,
            work_tile_info.reduction_subtile_idx()
          );

          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
          epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
          do_store_tail = true;
        }

        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
        work_tile_info = next_work_tile_info;

        did_batch_change = curr_batch != work_tile_info.L_idx;
        if (work_tile_info.is_valid()) {
          if (work_tile_info.L_idx / L_coords_per_splits != l_coord / L_coords_per_splits) {
            CUTLASS_PRAGMA_NO_UNROLL
            for (int sid = sid_processed + 1; sid < work_tile_info.L_idx / L_coords_per_splits;
                 sid++) {
              MMABarrier::arrive_inc(params.barrier, mma_thread_idx, sid, add_val);
            }
            sid_processed = work_tile_info.L_idx / L_coords_per_splits - 1;
          }
        }
        if (work_tile_info.is_valid() && did_batch_change) {
          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
          }
          if (warp_idx_in_warp_group == 0) {
            collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
              shared_storage.tensormaps.epilogue,
              params.epilogue,
              epi_store_tensormap,
              problem_shape_MNKL,
              work_tile_info.L_idx,
              consumer_warp_group_idx
            );

            // Converge before issuing tensormap fence release since fence is aligned
            __syncwarp();
            collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue,
                                                                       epi_store_tensormap,
                                                                       consumer_warp_group_idx);
          }
        }

      } // Scheduler work fetch loop

      // arrive inc for the last sid, in case the task is too small, some threadblockes does not
      // get the valid task at all
      CUTLASS_PRAGMA_NO_UNROLL
      for (int sid = sid_processed + 1; sid < params.SPLITS; sid++) {
        MMABarrier::arrive_inc(params.barrier, mma_thread_idx, sid, add_val);
      }
      // Cooperative only needs TMA to complete at the very end of the kernel
      if (do_store_tail) {
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state
        );
      }

    } // Consumer Warp Groups End

    //  finish the computation, can help with the gather-rs here
    {
      int n_compute_ctas = gridDim.x - GAHER_RS_N_CTAS;
      // gather_rs_dma_async_double_buffer(params, smem_buf, params.SPLITS-1, params.SPLITS,
      // blockIdx.x % n_compute_ctas, n_compute_ctas);
      if (params.ep_world_size > 1) {
        ep_topk_gather_rs_dma_v3(
            params,
            smem_buf,
            params.SPLITS - 1,
            params.SPLITS,
            blockIdx.x % n_compute_ctas,
            n_compute_ctas);
      } else {
        if (params.input_groups == 1 && CAN_USE_DOUBLE_BUFFER) {
          tp_topk_gather_rs_dma_double_buffer(
              params,
              smem_buf,
              params.SPLITS - 1,
              params.SPLITS,
              blockIdx.x % n_compute_ctas,
              n_compute_ctas);
        } else {
          tp_topk_gather_rs_dma(
              params,
              smem_buf,
              params.SPLITS - 1,
              params.SPLITS,
              blockIdx.x % n_compute_ctas,
              n_compute_ctas);
        }
      }
    }

#endif
  }

 private:
  // Kernel helper function to get next work unit
  CUTLASS_DEVICE
  typename TileScheduler::WorkTileInfo
  fetch_next_work(
      typename TileScheduler::WorkTileInfo &work_tile_info, TileScheduler &scheduler) const {
    // Check whether we should continue on with the current work unit. If this is the case,
    // the work unit will have been updated in continue_current_work to reflect the new
    // tile to be computed.
    if (scheduler.continue_current_work(work_tile_info)) {
      return work_tile_info;
    }

    // Get next work tile
    scheduler.advance_to_next_work();
    return scheduler.get_current_work();
  }
};

///////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::kernel
// clang-format on
