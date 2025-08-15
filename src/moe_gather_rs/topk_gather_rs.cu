//===- topk_gather_rs.cu ---------------------------------------------- C++ ---===//
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

#include "cute/container/array_aligned.hpp"
#include "cute/numeric/int.hpp"
#include "cutlass/arch/memory.h"
#include "cutlass/barrier.h"
#include "cutlass/numeric_conversion.h"
#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/system_barrier.hpp"
#include "flux/flux.h"
#include "moe_gather_rs/topk_gather_rs.hpp"

namespace bytedance {
namespace flux {

CUTLASS_DEVICE
void
global_red_bf16_2(uint32_t const &D, void *ptr, bool pred_guard) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p red.global.sys.add.noftz.bf16x2 [%0], %1;\n"
      "}\n"
      :
      : "l"(ptr), "r"(data), "r"((int)pred_guard));
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

CUTLASS_DEVICE
void
global_red_f16_2(uint32_t const &D, void *ptr, bool pred_guard) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p red.global.sys.add.noftz.f16x2 [%0], %1;\n"
      "}\n"
      :
      : "l"(ptr), "r"(data), "r"((int)pred_guard));
#else
  CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

template <typename Element>
CUTLASS_DEVICE void
global_red_v2(uint32_t val, void *ptr, bool guard) {
  if constexpr (std::is_same_v<Element, __half>) {
    global_red_f16_2(val, ptr, guard);
  } else if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
    global_red_bf16_2(val, ptr, guard);
  } else {
    static_assert(cutlass::detail::dependent_false<Element>, "unsupported Element");
  }
}

namespace {
template <
    const int BLOCK_M,
    const int BLOCK_N,
    const int TOPK,
    typename Element,
    const int N_THREADS>
__device__ void
gather_rs_impl(
    TopKReduceGatherRSArguments const &params,
    void *smem_buf,
    int blk_m,
    int blk_n,
    int sid,
    float input_scale) {
  int32_t *smem_idx = reinterpret_cast<int32_t *>(smem_buf);
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int N_REG = 16 / ELEMENT_SIZE;
  float acc[N_REG];
  Element reg128[N_REG];
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
  Element *inter_D = reinterpret_cast<Element *>(params.inter_D);
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
        for (int topk = 0; topk < TOPK; topk++) {
          int gmem_row_offset = smem_idx[topk + row_offset * TOPK];
          FETCH_128bit(&reg128[0]) =
              FETCH_128bit(inter_D + gmem_row_offset * NDim + gmem_col_offset);
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            acc[i] += element_to_float(reg128[i]);
          }
        }
#pragma unroll
        for (int i = 0; i < N_REG; i++) {
          reg128[i] = float_to_element<Element>(acc[i] * input_scale);
        }
        // write back to the taget rank
        int token_idx = token_offset_start + row_offset;
        int dst_rank = token_idx / token_per_rank;
        int remote_row = token_idx % token_per_rank + remote_token_offset;
        FETCH_128bit(output_scatter_ptrs[dst_rank] + remote_row * NDim + gmem_col_offset) =
            FETCH_128bit(&reg128[0]);
      }
    }
  }
}
}  // namespace

template <
    typename Element,
    const int GATHER_RS_BM,
    const int GATHER_RS_BN,
    const int TOPK,
    const int N_THREADS>
__global__ void
topk_gather_rs_kernel(TopKReduceGatherRSArguments const params) {
  using BarrierSync = cutlass::detail::SyncthreadsSync;
  using Barrier = cutlass::Barrier;
  // perform the reduction with float
  __shared__ int smem_buf[GATHER_RS_BM * TOPK];
  int32_t *barrier = params.barrier;
  int rank = params.rank;
  int world_size = params.world_size;

  constexpr int N_SMS_90 = 132;  // TODO: fix me here,
  float input_scale = 1.0;
  if (params.input_scale_ptr != nullptr) {
    input_scale = *params.input_scale_ptr;
  }
  int n_compute_ctas = N_SMS_90 - gridDim.x;
  int32_t dma_block_idx = blockIdx.x;
  int totalM = params.totalM;
  int totalToken = totalM / TOPK;
  int n_dim = params.n_dim;
  const int new_n_dim = n_dim / params.SPLITS;  // BUG FIX ME

  int token_M_blk_count = (totalToken + GATHER_RS_BM - 1) / GATHER_RS_BM;
  int token_M_blks_per_rank = token_M_blk_count / world_size;
  int token_M_blk_offset = rank * token_M_blks_per_rank;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < params.SPLITS; sid++) {
    Barrier::wait_eq(barrier, threadIdx.x, sid, n_compute_ctas);
    for (int blk_m = dma_block_idx; blk_m < token_M_blk_count; blk_m += gridDim.x) {
      int swizzled_m = (blk_m + token_M_blk_offset) % token_M_blk_count;
      for (int blk_n = 0; blk_n < new_n_dim / GATHER_RS_BN; blk_n++) {
        __syncthreads();
        gather_rs_impl<GATHER_RS_BM, GATHER_RS_BN, TOPK, Element, N_THREADS>(
            params, &smem_buf[0], swizzled_m, blk_n, sid, input_scale);
      }
    }
  }
}

void
topk_gather_rs(TopKReduceGatherRSArguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  const int NBLOCKS = args.n_tb_blocks;
  constexpr int NTHREADS = 1024;
  dim3 grid_dim(NBLOCKS, 1, 1);
  dim3 block_dim(NTHREADS);

  constexpr int GATHER_RS_BM = 128;
  constexpr int GATHER_RS_BN = 1024;
  const int n_dim_per_splits = args.n_dim / args.SPLITS;
  FLUX_CHECK(n_dim_per_splits % GATHER_RS_BN == 0);

  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(cute::_4{}, cute::_5{}, cute::_6{})),  // topk
      [&](auto tup) {
        auto [cdtype, ctopk] = tup;
        return cdtype == dtype and ctopk == args.topk;
      },
      [&](auto tup) {
        auto [cdtype, ctopk] = tup;
        constexpr int TOPK = decltype(ctopk){};
        using Element = decltype(to_cuda_dtype(cdtype));
        topk_gather_rs_kernel<Element, GATHER_RS_BM, GATHER_RS_BN, TOPK, NTHREADS>
            <<<grid_dim, block_dim, 0, stream>>>(args);
      },
      [&]() { FLUX_CHECK(false) << "unsupported for topk=" << args.topk << " dtype:" << dtype; });
}

template <const int BLOCK_M, const int BLOCK_N, const int TOPK>
struct EP_Shared_Storage {
  float smem_data[BLOCK_M][BLOCK_N];
  int smem_idx[BLOCK_M * TOPK];
};

template <
    const int BLOCK_M,
    const int BLOCK_N,
    const int TOPK,
    typename Element,
    const int N_THREADS,
    const bool HAS_VEC_SCALE,
    const int VecSize = 4>
__device__ void
ep_gather_rs_impl(
    TopKReduceGatherRSArguments const &params,
    void *smem_buf,
    int blk_m,
    int blk_n,
    int sid,
    int ep_m_start,
    int ep_m_end,
    float input_scale,
    float *output_vec_scale_ptr) {
  auto storage = reinterpret_cast<EP_Shared_Storage<BLOCK_M, BLOCK_N, TOPK> *>(smem_buf);
  int32_t *smem_idx = &storage->smem_idx[0];
  float *smem_data = &storage->smem_data[0][0];
  constexpr int ELEMENT_SIZE = sizeof(Element);
  static_assert(VecSize == 4);
  constexpr int N_REG = VecSize / ELEMENT_SIZE;
  static_assert(N_REG == 2);
  float acc[N_REG];
  Element reg128[N_REG];
  int wid = threadIdx.x / 32;
  int wtid = threadIdx.x % 32;
  constexpr int N_WARPS = N_THREADS / 32;  // each warp is responsible for a row
  constexpr int UNROLL_M = (BLOCK_M + N_WARPS - 1) / N_WARPS;
  constexpr int UNROLL_N = BLOCK_N / 32 / N_REG;
  // load the routing_idx first to the shared memory
  constexpr int TOKEN_IDX_N = BLOCK_M * TOPK;
  constexpr int IDX_LOAD_UNROLL = (TOKEN_IDX_N + N_THREADS - 1) / N_THREADS;
  // printf("before params\n");
  int globalM = params.globalM;
  // printf("total M :%d \n", totalM);
  int totalToken = globalM / params.topk;
  int token_per_rank = totalToken / params.world_size;
  int NDim = params.n_dim;
  int new_n_dim = NDim / params.SPLITS;
  int routing_idx_start = blk_m * BLOCK_M * TOPK;
  int routing_idx_end = min(routing_idx_start + BLOCK_M * TOPK, totalToken * TOPK);
  int token_offset_start = blk_m * BLOCK_M;
  int token_offset_end = min(token_offset_start + BLOCK_M, totalToken);
  float2 zero = {0, 0};
  // int remote_token_offset = params.rank * token_per_rank;
  Element *inter_D = reinterpret_cast<Element *>(params.inter_D);
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
      int count = 0;
#pragma unroll
      for (int topk = 0; topk < TOPK; topk++) {
        int gmem_row_offset = smem_idx[topk + row_offset * TOPK];
        count += (gmem_row_offset >= ep_m_start && gmem_row_offset < ep_m_end);
      }
      if (count > 0) {
        //         // set the whole row to be zero

#pragma unroll
        for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
          int smem_col_offset = col_iter * 32 * N_REG + wtid * N_REG;
          int smem_row_offset = row_offset;
          FETCH_64bit(smem_data + smem_row_offset * BLOCK_N + smem_col_offset) = zero;
        }
#pragma unroll
        for (int topk = 0; topk < TOPK; topk++) {
          int gmem_row_offset = smem_idx[topk + row_offset * TOPK];
          if (gmem_row_offset >= ep_m_start && gmem_row_offset < ep_m_end) {
            float vec_scale = 1.0;
            if constexpr (HAS_VEC_SCALE) {
              vec_scale = output_vec_scale_ptr[gmem_row_offset - ep_m_start];
            }
#pragma unroll
            for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
              int smem_col_offset = col_iter * 32 * N_REG + wtid * N_REG;
              int smem_row_offset = row_offset;
              int gmem_col_offset = smem_col_offset + sid * new_n_dim + blk_n * BLOCK_N;
              FETCH_32bit(&reg128[0]) =
                  FETCH_32bit(inter_D + (gmem_row_offset - ep_m_start) * NDim + gmem_col_offset);
              FETCH_64bit(&acc[0]) =
                  FETCH_64bit(smem_data + smem_row_offset * BLOCK_N + smem_col_offset);
#pragma unroll
              for (int i = 0; i < N_REG; i++) {
                if constexpr (HAS_VEC_SCALE) {
                  acc[i] += element_to_float(reg128[i]) * vec_scale * input_scale;
                } else {
                  acc[i] += element_to_float(reg128[i]) * input_scale;
                }
              }
              FETCH_64bit(smem_data + smem_row_offset * BLOCK_N + smem_col_offset) =
                  FETCH_64bit(&acc[0]);
            }
          }
        }
        // write the result back to the global memory
        int token_idx = token_offset_start + row_offset;
        int dst_rank = token_idx / token_per_rank;
        int remote_row = token_idx % token_per_rank;
#pragma unroll
        for (int col_iter = 0; col_iter < UNROLL_N; col_iter++) {
          int smem_col_offset = col_iter * 32 * N_REG + wtid * N_REG;
          int gmem_col_offset = smem_col_offset + sid * new_n_dim + blk_n * BLOCK_N;
          FETCH_64bit(&acc[0]) =
              *reinterpret_cast<float2 *>(smem_data + row_offset * BLOCK_N + smem_col_offset);
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            reg128[i] = float_to_element<Element>(acc[i]);
          }
          // if(threadIdx.x % 32 ==0 && token_idx == 0 && col_iter==0){
          //   printf("rank:%d count:%d %f %f\n", params.rank, count, acc[0], acc[1]);
          // }
          auto val = *reinterpret_cast<uint32_t *>(&reg128[0]);
          global_red_v2<Element>(
              val, output_scatter_ptrs[dst_rank] + remote_row * NDim + gmem_col_offset, 1);
          // global_red_bf16_2(
          //     val, output_scatter_ptrs[dst_rank] + remote_row * NDim + gmem_col_offset, 1);
        }
      }
    }
  }
}

template <
    typename Element,
    const int GATHER_RS_BM,
    const int GATHER_RS_BN,
    const int TOPK,
    const int N_THREADS>
__global__ void
ep_topk_gather_rs_kernel(
    TopKReduceGatherRSArguments const params, int32_t ep_m_start, int32_t ep_m_end) {
  // using BarrierSync = cutlass::detail::SyncthreadsSync;
  // using Barrier = cutlass::Barrier;
  using Barrier = cutlass::detail::SystemBarrier;
  // perform the reduction with float
  extern __shared__ char smem_buf[];
  int32_t *barrier = params.barrier;
  int rank = params.rank;
  int world_size = params.world_size;

  constexpr int N_SMS_90 = 132;
  float input_scale = 1.0;
  if (params.input_scale_ptr != nullptr) {
    input_scale = *params.input_scale_ptr;
  }
  bool has_output_vec_scale = params.output_vec_scale_ptr != nullptr;
  int n_compute_ctas = N_SMS_90 - gridDim.x;
  int32_t dma_block_idx = blockIdx.x;
  int globalM = params.globalM;  // M for all experts
  int totalToken = globalM / TOPK;
  int n_dim = params.n_dim;
  const int new_n_dim = n_dim / params.SPLITS;  // BUG FIX ME

  int token_M_blk_count = (totalToken + GATHER_RS_BM - 1) / GATHER_RS_BM;
  int token_M_blks_per_rank = token_M_blk_count / world_size;
  int token_M_blk_offset = rank * token_M_blks_per_rank;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < params.SPLITS - 1; sid++) {
    Barrier::wait_eq(barrier, threadIdx.x, sid, n_compute_ctas);
    for (int blk_m = dma_block_idx; blk_m < token_M_blk_count; blk_m += gridDim.x) {
      int swizzled_m = (blk_m + token_M_blk_offset) % token_M_blk_count;
      for (int blk_n = 0; blk_n < new_n_dim / GATHER_RS_BN; blk_n++) {
        __syncthreads();
        if (!has_output_vec_scale) {
          ep_gather_rs_impl<GATHER_RS_BM, GATHER_RS_BN, TOPK, Element, N_THREADS, false>(
              params,
              smem_buf,
              swizzled_m,
              blk_n,
              sid,
              ep_m_start,
              ep_m_end,
              input_scale,
              params.output_vec_scale_ptr);

        } else {
          ep_gather_rs_impl<GATHER_RS_BM, GATHER_RS_BN, TOPK, Element, N_THREADS, true>(
              params,
              smem_buf,
              swizzled_m,
              blk_n,
              sid,
              ep_m_start,
              ep_m_end,
              input_scale,
              params.output_vec_scale_ptr);
        }
      }
    }
  }
}

void
ep_topk_gather_rs(
    TopKReduceGatherRSArguments const &args,
    DataTypeEnum dtype,
    int32_t ep_m_start,
    int32_t ep_m_end,
    cudaStream_t stream) {
  const int NBLOCKS = args.n_tb_blocks;
  constexpr int NTHREADS = 1024;
  dim3 grid_dim(NBLOCKS, 1, 1);
  dim3 block_dim(NTHREADS);

  constexpr int GATHER_RS_BM = 64;
  constexpr int GATHER_RS_BN = 640;
  // printf("rank:%d start:%d end:%d\n", args.rank, ep_m_start, ep_m_end);
  // TODO check the parameter at the host side
  const int n_dim_per_splits = args.n_dim / args.SPLITS;
  FLUX_CHECK(n_dim_per_splits % GATHER_RS_BN == 0);

  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(cute::_4{}, cute::_5{}, cute::_6{})),  // topk
      [&](auto tup) {
        auto [cdtype, ctopk] = tup;
        return cdtype == dtype and ctopk == args.topk;
      },
      [&](auto tup) {
        auto [cdtype, ctopk] = tup;
        constexpr int TOPK = decltype(ctopk){};
        using Element = decltype(to_cuda_dtype(cdtype));
        int smem_size = sizeof(EP_Shared_Storage<GATHER_RS_BM, GATHER_RS_BN, TOPK>);
        CUDA_CHECK(cudaFuncSetAttribute(
            ep_topk_gather_rs_kernel<Element, GATHER_RS_BM, GATHER_RS_BN, TOPK, NTHREADS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
        ep_topk_gather_rs_kernel<Element, GATHER_RS_BM, GATHER_RS_BN, TOPK, NTHREADS>
            <<<grid_dim, block_dim, smem_size, stream>>>(args, ep_m_start, ep_m_end);
      },
      [&]() { FLUX_CHECK(false) << "unsupported for topk=" << args.topk << " dtype:" << dtype; });
}

}  // namespace flux
}  // namespace bytedance
