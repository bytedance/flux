//===- dis_scatter_backward_impl.cu --------------------------------------- C++ ---===//
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
#include <cooperative_groups.h>
#include <nvshmem.h>

#include "cute/numeric/integral_constant.hpp"
#include "dis_scatter_backward_impl.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/memory_utils.hpp"
#include "flux/flux.h"
#define ALL2ALL_COMBINE_BLOCK 4
#define ALL2ALL_MAX_TASK_BLOCKS 40
namespace bytedance {
namespace flux {

__device__ void
load_128bit_sys(void *src_ptr, void *dst_ptr) {
  float4 *src = reinterpret_cast<float4 *>(src_ptr);
  float4 *dst = reinterpret_cast<float4 *>(dst_ptr);
  float4 value;
  uintptr_t addr = (uintptr_t)(src);

  asm volatile("ld.global.relaxed.sys.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(addr));

  *dst = value;
}

template <const int NUM_THREADS, const int BLOCK_SIZE>
__global__ void
dis_scatter_backward_build_index_kernel(const DisScatterBackwardBuildIndexParams params) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // each thread process a token block which consists of `BLOCK_SIZE` tokens
  int token_block_offset = bid * NUM_THREADS + tid;
  int token_idx_start = token_block_offset * BLOCK_SIZE;
  int token_idx_end = token_idx_start + BLOCK_SIZE;
  int experts_per_rank = params.total_num_experts / params.world_size;
  int experts_per_node = params.total_num_experts / params.n_nodes;
  int token_idx_to_send[BLOCK_SIZE];
  int topk_act_count[BLOCK_SIZE];
  int topk = params.topk;
  const int *ep_cum_sum_ptr =
      params.ep_cum_sum_gpu_ptr != nullptr ? params.ep_cum_sum_gpu_ptr : params.ep_cum_sum;
  // registers to save the token idx to be sent in the block
  const int max_token = params.max_token_per_rank;
  for (int node_offset = 0; node_offset < params.n_nodes; node_offset++) {
    int target_node = (params.cur_node_id + node_offset) % params.n_nodes;
    int node_start = params.get_global_token_start(target_node);
    int node_end = params.get_global_token_end(target_node);
    int n_token_tgt_node = node_end - node_start;
    if (token_idx_start < n_token_tgt_node) {
      int global_idx_start = token_idx_start + node_start;
      int global_idx_end = min(token_idx_end, n_token_tgt_node) + node_start;
      int hit_token_count = 0;
      for (int global_idx = global_idx_start; global_idx < global_idx_end; global_idx++) {
        int local_idx = global_idx - node_start;
        int hit_topk_count = 0;
        for (int top_id = 0; top_id < topk; top_id++) {
          int exp_id = params.ag_exp_indices[global_idx * topk + top_id];
          int exp_tgt_node = exp_id / experts_per_node;
          int exp_tgt_rank = exp_id / experts_per_rank;
          int exp_tgt_local_rank = exp_tgt_rank % params.local_world_size;
          if (exp_tgt_node == params.cur_node_id) {
            // perform the scatter idx calculation here
            int scatter_idx = params.ag_scatter_idx[global_idx * topk + top_id];
            int offset_within_tgt_rank = scatter_idx - ep_cum_sum_ptr[exp_tgt_local_rank];
            int pos = target_node * max_token * topk + local_idx * topk + hit_topk_count;
            params.token_scatterd_pos[pos] = offset_within_tgt_rank;
            params.token_scatterd_local_rank[pos] = exp_tgt_local_rank;
            hit_topk_count += 1;
          }
        }
        if (hit_topk_count > 0) {
          // TODO check whether to use the local idx or the global idx
          token_idx_to_send[hit_token_count] = local_idx;
          topk_act_count[hit_token_count] = hit_topk_count;
          hit_token_count += 1;
        }
      }
      // record the block idx accordingly here
      if (hit_token_count > 0) {
        // current block has `hit_token_count` tokenes to perform the topk-reduction
        int block_offset = atomicAdd(params.block_count_send + target_node, 1);
        // params.block_idx_send[];
        params.block_idx_send[target_node * max_token + block_offset] = token_block_offset;
        params.block_n_tokens[target_node * max_token + block_offset] = hit_token_count;
        for (int i = 0; i < hit_token_count; i++) {
          params.token_idx_send[target_node * max_token + BLOCK_SIZE * token_block_offset + i] =
              token_idx_to_send[i];
          params.token_topk_count[target_node * max_token + BLOCK_SIZE * token_block_offset + i] =
              topk_act_count[i];
        }
      }
    }
  }
  // perform the reduction index calculation
  int cur_node_id = params.cur_node_id;
  int node_start = params.get_global_token_start(cur_node_id);
  int node_end = params.get_global_token_end(cur_node_id);
  int n_token_tgt_node = node_end - node_start;
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE; i++) {
    token_idx_to_send[i] = 0;
  }
  if (token_idx_start < n_token_tgt_node) {
    int global_idx_start = token_idx_start + node_start;
    int global_idx_end = min(token_idx_end, n_token_tgt_node) + node_start;
    for (int nid = 0; nid < params.n_nodes; nid++) {
      int offset_within_block = 0;
      for (int global_idx = global_idx_start; global_idx < global_idx_end; global_idx++) {
        int local_idx = global_idx - node_start;
        int idx_within_block = global_idx - global_idx_start;
        int topk_count = 0;
        for (int top_id = 0; top_id < topk; top_id++) {
          int exp_id = params.ag_exp_indices[global_idx * topk + top_id];
          int exp_tgt_node = exp_id / experts_per_node;
          if (exp_tgt_node == nid) {
            topk_count += 1;
          }
        }
        if (topk_count > 0) {
          int pos = nid * max_token + BLOCK_SIZE * token_block_offset + offset_within_block;
          int write_pos = local_idx * (topk + 1) + token_idx_to_send[idx_within_block];
          params.reduce_token_idx[write_pos] = pos;
          token_idx_to_send[idx_within_block] += 1;
          offset_within_block += 1;
        }
      }
    }
    for (int global_idx = global_idx_start; global_idx < global_idx_end; global_idx++) {
      int local_idx = global_idx - node_start;
      int idx_within_block = global_idx - global_idx_start;
      // write the number of token to be reduced
      params.reduce_token_idx[local_idx * (topk + 1) + topk] = token_idx_to_send[idx_within_block];
    }
  }
}

template <const int NUM_THREADS>
__global__ void
dis_scatter_backward_build_index_single_node_kernel(
    const DisScatterBackwardBuildIndexSingleNodeParams params) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gtid = tid + bid * blockDim.x;
  int rank = params.rank;
  int topk = params.topk;
  const int *ep_cum_sum_ptr =
      params.ep_cum_sum_gpu_ptr != nullptr ? params.ep_cum_sum_gpu_ptr : params.ep_cum_sum;
  int scatter_pos_offset = ep_cum_sum_ptr[rank];
  int num_experts_per_rank = params.total_num_experts / params.world_size;
  for (int cur_rank = 0; cur_rank < params.world_size; cur_rank++) {
    int g_token_start = params.get_global_token_start(cur_rank);
    int g_token_end = params.get_global_token_end(cur_rank);
    if (gtid < g_token_end - g_token_start) {
      int g_token_pos = g_token_start + gtid;
      int topk_hit = 0;
      for (int top_id = 0; top_id < topk; top_id++) {
        int pos = g_token_pos * topk + top_id;
        int exp_id = params.ag_exp_indices[pos];
        if (exp_id / num_experts_per_rank == rank) {
          int scatter_pos = params.ag_scatter_idx[pos] - scatter_pos_offset;
          // write the scatter pos
          int write_pos = g_token_pos * (topk + 1) + topk_hit;
          params.token_local_reduce_pos[write_pos] = scatter_pos;
          topk_hit += 1;
        }
      }
      if (topk_hit > 0) {
        params.token_local_reduce_pos[g_token_pos * (topk + 1) + topk] = topk_hit;
        int tmp_pos = atomicAdd(params.token_count_send + cur_rank, 1);
        params.token_idx_send[tmp_pos + cur_rank * params.max_token_per_rank] = g_token_pos;
      }
    }
  }
  // perform the index computation for the topk_reduce kernel
  int g_token_start = params.get_global_token_start(rank);
  int g_token_end = params.get_global_token_end(rank);
  if (gtid < g_token_end - g_token_start) {
    int token_local_idx = gtid;
    int hit_rank = 0;
    int g_token_pos = g_token_start + gtid;
    for (int cur_rank = 0; cur_rank < params.world_size; cur_rank++) {
      int hit_topk = 0;
      for (int top_id = 0; top_id < topk; top_id++) {
        int pos = g_token_pos * topk + top_id;
        int exp_id = params.ag_exp_indices[pos];
        if (exp_id / num_experts_per_rank == cur_rank) {
          hit_topk += 1;
        }
      }
      if (hit_topk > 0) {
        int pos = cur_rank * params.max_token_per_rank + token_local_idx;
        int write_pos = token_local_idx * (topk + 1) + hit_rank;
        params.reduce_token_idx[write_pos] = pos;
        hit_rank += 1;
      }
    }
    params.reduce_token_idx[token_local_idx * (topk + 1) + topk] = hit_rank;
  }
}

template <
    typename Element,
    const int NUM_THREADS,
    const int64_t N_DIM,
    const int BLOCK_M,
    const int TOPK>
__global__ void
dis_scatter_backward_kernel(const DisScatterBackwardParams params) {
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int VEC_SIZE = 16;
  constexpr int WARP_SIZE = 32;

  constexpr int NUM_WARP = NUM_THREADS / WARP_SIZE;
  static_assert(VEC_SIZE % ELEMENT_SIZE == 0);
  constexpr int N_ELEMENT_PER_VEC = VEC_SIZE / ELEMENT_SIZE;
  constexpr int N_ELEMENT_PER_ITER = WARP_SIZE * N_ELEMENT_PER_VEC;
  static_assert(N_DIM % N_ELEMENT_PER_ITER == 0, "N_DIM must be divisible by N_ELEMENT_PER_ITER");
  constexpr int N_UNROLL = N_DIM / N_ELEMENT_PER_ITER;
  static_assert(TOPK < WARP_SIZE);
  float acc[N_ELEMENT_PER_VEC];
  Element reg128[N_ELEMENT_PER_VEC];
  int wid = threadIdx.x / 32;
  int wtid = threadIdx.x % 32;
  __shared__ __align__(16) Element *shm_buffer[TOPK * NUM_WARP];
  Element **internal_src_ptr = &shm_buffer[TOPK * wid];
  const int max_token = params.index_args.max_token_per_rank;
  int local_rank = params.index_args.rank % params.index_args.local_world_size;
  int n_nodes = params.index_args.n_nodes;
  Element *reduction_ptr;
  if (n_nodes > 1)
    reduction_ptr = reinterpret_cast<Element *>(params.reduction_buffer);
  else
    reduction_ptr = reinterpret_cast<Element *>(params.result_buffer);
  Element *output_ptr = reinterpret_cast<Element *>(params.output_ptrs[local_rank]);

  for (int node_offset = 1; node_offset < params.index_args.n_nodes; node_offset++) {
    int send_node = (params.index_args.cur_node_id + node_offset) % params.index_args.n_nodes;
    int wid_send = wid + blockIdx.x * NUM_WARP;
    int block_to_send = params.index_args.block_count_send[send_node];
    // each warp is responsible for a block
    for (int block_offset = wid_send; block_offset < block_to_send;
         block_offset += NUM_WARP * gridDim.x) {
      // keep performing the reduction and sending until there are not any data left
      int token_block_idx = params.index_args.block_idx_send[send_node * max_token + block_offset];
      int n_token_in_block =
          params.index_args.block_n_tokens[send_node * max_token + block_offset];
      for (int token_count = 0; token_count < n_token_in_block; token_count++) {
        int token_pos = send_node * max_token + token_block_idx * BLOCK_M + token_count;
        int local_token_idx = params.index_args.token_idx_send[token_pos];
        int token_act_topk = params.index_args.token_topk_count[token_pos];
        // preload the ptrs into the shared memory
        if (wtid < token_act_topk) {
          int pos = send_node * max_token * TOPK + local_token_idx * TOPK + wtid;
          int target_local_rank = params.index_args.token_scatterd_local_rank[pos];
          int64_t scatter_row_offset = params.index_args.token_scatterd_pos[pos];
          Element *tmp_ptr = reinterpret_cast<Element *>(params.internal_ptrs[target_local_rank]) +
                             scatter_row_offset * N_DIM;
          internal_src_ptr[wtid] = tmp_ptr;
        }
        __syncwarp();
        // #pragma unroll
        for (int n_iter = 0; n_iter < N_UNROLL; n_iter++) {
          // clear the acc
          // #pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            acc[i] = 0.0f;
          }
          // perform the topk reduction here
          for (int top_id = 0; top_id < token_act_topk; top_id++) {
            load_128bit_sys(
                internal_src_ptr[top_id] + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC,
                &reg128[0]);
            // FETCH_128bit(&reg128[0]) = FETCH_128bit(
            //     internal_src_ptr[top_id] + n_iter * N_ELEMENT_PER_ITER + wtid *
            //     N_ELEMENT_PER_VEC);
            // #pragma unroll
            for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
              acc[i] += element_to_float(reg128[i]);
            }
          }
          // #pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            reg128[i] = float_to_element<Element>(acc[i]);
          }
          // write the output to the output buffer
          // flatten the data within the same block
          int64_t pos = token_pos * N_DIM + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
          FETCH_128bit(&reduction_ptr[pos]) = FETCH_128bit(&reg128[0]);
        }
      }
      __threadfence_system();
      // call nvshmem put mem nbi here to send the data to the target node
      int64_t src_pos_m = send_node * max_token + token_block_idx * BLOCK_M;
      void *src_ptr = &reduction_ptr[src_pos_m * N_DIM];
      int64_t dst_pos_m = params.index_args.cur_node_id * max_token + token_block_idx * BLOCK_M;
      void *dst_ptr = output_ptr + dst_pos_m * N_DIM;
      const int msg_size = N_DIM * sizeof(Element) * n_token_in_block;
      int dst_rank =
          send_node * params.index_args.local_world_size + local_rank + params.global_rank_offset;
      nvshmemx_putmem_nbi_warp(dst_ptr, src_ptr, msg_size, dst_rank);
    }
  }
  {
    // put the topk reduction of current at the last to better hide the I/O
    int send_node = params.index_args.cur_node_id;
    int wid_send = wid + blockIdx.x * NUM_WARP;
    int block_to_send = params.index_args.block_count_send[send_node];
    // each warp is responsible for a block
    for (int block_offset = wid_send; block_offset < block_to_send;
         block_offset += NUM_WARP * gridDim.x) {
      // keep performing the reduction and sending until there are not any data left
      int token_block_idx = params.index_args.block_idx_send[send_node * max_token + block_offset];
      int n_token_in_block =
          params.index_args.block_n_tokens[send_node * max_token + block_offset];
      for (int token_count = 0; token_count < n_token_in_block; token_count++) {
        int token_pos = send_node * max_token + token_block_idx * BLOCK_M + token_count;
        int local_token_idx = params.index_args.token_idx_send[token_pos];
        int token_act_topk = params.index_args.token_topk_count[token_pos];
        // preload the ptrs into the shared memory
        if (wtid < token_act_topk) {
          int pos = send_node * max_token * TOPK + local_token_idx * TOPK + wtid;
          int target_local_rank = params.index_args.token_scatterd_local_rank[pos];
          int64_t scatter_row_offset = params.index_args.token_scatterd_pos[pos];
          Element *tmp_ptr = reinterpret_cast<Element *>(params.internal_ptrs[target_local_rank]) +
                             scatter_row_offset * N_DIM;
          internal_src_ptr[wtid] = tmp_ptr;
        }
        __syncwarp();
        // #pragma unroll
        for (int n_iter = 0; n_iter < N_UNROLL; n_iter++) {
          // clear the acc
          // #pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            acc[i] = 0.0f;
          }
          // perform the topk reduction here
          for (int top_id = 0; top_id < token_act_topk; top_id++) {
            load_128bit_sys(
                internal_src_ptr[top_id] + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC,
                &reg128[0]);
            // FETCH_128bit(&reg128[0]) = FETCH_128bit(
            //     internal_src_ptr[top_id] + n_iter * N_ELEMENT_PER_ITER + wtid *
            //     N_ELEMENT_PER_VEC);
            // #pragma unroll
            for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
              acc[i] += element_to_float(reg128[i]);
            }
          }
          // #pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            reg128[i] = float_to_element<Element>(acc[i]);
          }
          // write the output to the output buffer
          // flatten the data within the same block
          int64_t pos = token_pos * N_DIM + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
          FETCH_128bit(&reduction_ptr[pos]) = FETCH_128bit(&reg128[0]);
        }
      }
      __threadfence_system();
      if (n_nodes > 1) {
        // call nvshmem put mem nbi here to send the data to the target node
        int64_t src_pos_m = send_node * max_token + token_block_idx * BLOCK_M;
        void *src_ptr = &reduction_ptr[src_pos_m * N_DIM];
        int64_t dst_pos_m = params.index_args.cur_node_id * max_token + token_block_idx * BLOCK_M;
        void *dst_ptr = output_ptr + dst_pos_m * N_DIM;
        const int msg_size = N_DIM * sizeof(Element) * n_token_in_block;
        int dst_rank = send_node * params.index_args.local_world_size + local_rank +
                       params.global_rank_offset;
        nvshmemx_putmem_nbi_warp(dst_ptr, src_ptr, msg_size, dst_rank);
      }
    }
  }
}

template <typename Element, const int NUM_THREADS, const int TOPK>
struct __align__(16) ShareStorage {
  static_assert(NUM_THREADS % 32 == 0);
  static constexpr int NUM_WARPS = NUM_THREADS / 32;
  static constexpr int ELEMENT_SIZE = sizeof(Element);
  static constexpr int VEC_SIZE = 16;
  static constexpr int N_VEC = VEC_SIZE / ELEMENT_SIZE;
  Element data[NUM_WARPS * TOPK][32 * N_VEC];
  Element *row_ptr[NUM_WARPS * TOPK];
};

template <
    typename Element,
    const int NUM_THREADS,
    const int64_t N_DIM,
    const int BLOCK_M,
    const int TOPK>
__global__ void
dis_scatter_backward_kernel_v2(const DisScatterBackwardParams params) {
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int VEC_SIZE = 16;
  constexpr int WARP_SIZE = 32;
  extern __shared__ float4 smem_buf[];
  auto sstore = reinterpret_cast<ShareStorage<Element, NUM_THREADS, TOPK> *>(smem_buf);
  constexpr int NUM_WARP = NUM_THREADS / WARP_SIZE;
  static_assert(VEC_SIZE % ELEMENT_SIZE == 0);
  constexpr int N_ELEMENT_PER_VEC = VEC_SIZE / ELEMENT_SIZE;
  constexpr int N_ELEMENT_PER_ITER = WARP_SIZE * N_ELEMENT_PER_VEC;
  static_assert(N_DIM % N_ELEMENT_PER_ITER == 0, "N_DIM must be divisible by N_ELEMENT_PER_ITER");
  constexpr int N_UNROLL = N_DIM / N_ELEMENT_PER_ITER;
  static_assert(TOPK < WARP_SIZE);

  float acc[N_ELEMENT_PER_VEC];
  Element reg128[N_ELEMENT_PER_VEC];
  int wid = threadIdx.x / 32;
  int wtid = threadIdx.x % 32;
  Element **internal_src_ptr = &sstore->row_ptr[TOPK * wid];
  int smem_data_base = __cvta_generic_to_shared(&sstore->data[0]);
  const int max_token = params.index_args.max_token_per_rank;
  int local_rank = params.index_args.rank % params.index_args.local_world_size;
  Element *reduction_ptr;
  reduction_ptr = reinterpret_cast<Element *>(params.reduction_buffer);

  Element *output_ptr = reinterpret_cast<Element *>(params.output_ptrs[local_rank]);

  for (int node_offset = 1; node_offset < params.index_args.n_nodes; node_offset++) {
    int send_node = (params.index_args.cur_node_id + node_offset) % params.index_args.n_nodes;
    int wid_send = wid + blockIdx.x * NUM_WARP;
    int block_to_send = params.index_args.block_count_send[send_node];
    // each warp is responsible for a block
    for (int block_offset = wid_send; block_offset < block_to_send;
         block_offset += NUM_WARP * gridDim.x) {
      // keep performing the reduction and sending until there are not any data left
      int token_block_idx = params.index_args.block_idx_send[send_node * max_token + block_offset];
      int n_token_in_block =
          params.index_args.block_n_tokens[send_node * max_token + block_offset];
      for (int token_count = 0; token_count < n_token_in_block; token_count++) {
        int token_pos = send_node * max_token + token_block_idx * BLOCK_M + token_count;
        int local_token_idx = params.index_args.token_idx_send[token_pos];
        int token_act_topk = params.index_args.token_topk_count[token_pos];
        // preload the ptrs into the shared memory
        if (wtid < token_act_topk) {
          int pos = send_node * max_token * TOPK + local_token_idx * TOPK + wtid;
          int target_local_rank = params.index_args.token_scatterd_local_rank[pos];
          int64_t scatter_row_offset = params.index_args.token_scatterd_pos[pos];
          Element *tmp_ptr = reinterpret_cast<Element *>(params.internal_ptrs[target_local_rank]) +
                             scatter_row_offset * N_DIM;
          internal_src_ptr[wtid] = tmp_ptr;
        }
        __syncwarp();
        // #pragma unroll
        for (int n_iter = 0; n_iter < N_UNROLL; n_iter++) {
          // #pragma unroll
          // perform the topk reduction here
          for (int top_id = 0; top_id < token_act_topk; top_id++) {
            int pos = (wid * TOPK + top_id) * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
            int dst_shm = smem_data_base + pos * sizeof(Element);
            Element *src_gmem =
                internal_src_ptr[top_id] + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
            cutlass::arch::async_load<VEC_SIZE>(dst_shm, src_gmem, 1);
          }
          asm("cp.async.commit_group;\n" ::);
          asm("cp.async.wait_group 0;\n" ::);
          // clear the acc
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            acc[i] = 0.0f;
          }
          for (int top_id = 0; top_id < token_act_topk; top_id++) {
            FETCH_128bit(&reg128[0]) =
                FETCH_128bit(&sstore->data[wid * TOPK + top_id][wtid * N_ELEMENT_PER_VEC]);
#pragma unroll
            for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
              acc[i] += element_to_float(reg128[i]);
            }
          }
#pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            reg128[i] = float_to_element<Element>(acc[i]);
          }
          // write the output to the output buffer
          // flatten the data within the same block
          int64_t pos = token_pos * N_DIM + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
          FETCH_128bit(&reduction_ptr[pos]) = FETCH_128bit(&reg128[0]);
        }
      }
      __threadfence();
      // call nvshmem put mem nbi here to send the data to the target node
      int64_t src_pos_m = send_node * max_token + token_block_idx * BLOCK_M;
      void *src_ptr = &reduction_ptr[src_pos_m * N_DIM];
      int64_t dst_pos_m = params.index_args.cur_node_id * max_token + token_block_idx * BLOCK_M;
      void *dst_ptr = output_ptr + dst_pos_m * N_DIM;
      const int msg_size = N_DIM * sizeof(Element) * n_token_in_block;
      int dst_rank =
          send_node * params.index_args.local_world_size + local_rank + params.global_rank_offset;
      nvshmemx_putmem_nbi_warp(dst_ptr, src_ptr, msg_size, dst_rank);
    }
  }
  {
    reduction_ptr = output_ptr;
    // put the topk reduction of current at the last to better hide the I/O
    int send_node = params.index_args.cur_node_id;
    int wid_send = wid + blockIdx.x * NUM_WARP;
    int block_to_send = params.index_args.block_count_send[send_node];
    // each warp is responsible for a block
    for (int block_offset = wid_send; block_offset < block_to_send;
         block_offset += NUM_WARP * gridDim.x) {
      // keep performing the reduction and sending until there are not any data left
      int token_block_idx = params.index_args.block_idx_send[send_node * max_token + block_offset];
      int n_token_in_block =
          params.index_args.block_n_tokens[send_node * max_token + block_offset];
      for (int token_count = 0; token_count < n_token_in_block; token_count++) {
        int token_pos = send_node * max_token + token_block_idx * BLOCK_M + token_count;
        int local_token_idx = params.index_args.token_idx_send[token_pos];
        int token_act_topk = params.index_args.token_topk_count[token_pos];
        // preload the ptrs into the shared memory
        if (wtid < token_act_topk) {
          int pos = send_node * max_token * TOPK + local_token_idx * TOPK + wtid;
          int target_local_rank = params.index_args.token_scatterd_local_rank[pos];
          int64_t scatter_row_offset = params.index_args.token_scatterd_pos[pos];
          Element *tmp_ptr = reinterpret_cast<Element *>(params.internal_ptrs[target_local_rank]) +
                             scatter_row_offset * N_DIM;
          internal_src_ptr[wtid] = tmp_ptr;
        }
        __syncwarp();
        for (int n_iter = 0; n_iter < N_UNROLL; n_iter++) {
          // #pragma unroll
          // perform the topk reduction here
          for (int top_id = 0; top_id < token_act_topk; top_id++) {
            int pos = (wid * TOPK + top_id) * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
            int dst_shm = smem_data_base + pos * sizeof(Element);
            Element *src_gmem =
                internal_src_ptr[top_id] + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
            cutlass::arch::async_load<VEC_SIZE>(dst_shm, src_gmem, 1);
          }
          asm("cp.async.commit_group;\n" ::);
          asm("cp.async.wait_group 0;\n" ::);
// clear the acc
#pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            acc[i] = 0.0f;
          }
          for (int top_id = 0; top_id < token_act_topk; top_id++) {
            FETCH_128bit(&reg128[0]) =
                FETCH_128bit(&sstore->data[wid * TOPK + top_id][wtid * N_ELEMENT_PER_VEC]);
#pragma unroll
            for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
              acc[i] += element_to_float(reg128[i]);
            }
          }
#pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            reg128[i] = float_to_element<Element>(acc[i]);
          }
          // write the output to the output buffer
          // flatten the data within the same block
          int64_t pos = token_pos * N_DIM + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
          FETCH_128bit(&reduction_ptr[pos]) = FETCH_128bit(&reg128[0]);
        }
      }
      // if (n_nodes > 1) {
      //   // call nvshmem put mem nbi here to send the data to the target node
      //   int src_pos_m = send_node * max_token + token_block_idx * BLOCK_M;
      //   void *src_ptr = &reduction_ptr[src_pos_m * N_DIM];
      //   int dst_pos_m = params.index_args.cur_node_id * max_token + token_block_idx * BLOCK_M;
      //   void *dst_ptr = output_ptr + dst_pos_m * N_DIM;
      //   const int msg_size = N_DIM * sizeof(Element) * n_token_in_block;
      //   int dst_rank = send_node * params.index_args.local_world_size + local_rank;
      //   nvshmemx_putmem_nbi_warp(dst_ptr, src_ptr, msg_size, dst_rank);
      // }
    }
  }
}
template <typename Element, const int NUM_THREADS, const int64_t N_DIM, const int TOPK>
__global__ void
dis_scatter_backward_single_node_kernel(const DisScatterBackwardSingleNodeParams params) {
  static_assert(NUM_THREADS % 32 == 0);
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int VEC_SIZE = 16;
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static_assert(VEC_SIZE % ELEMENT_SIZE == 0);
  constexpr int N_ELEMENT_PER_VEC = VEC_SIZE / ELEMENT_SIZE;
  constexpr int N_ELEMENT_PER_ITER = WARP_SIZE * N_ELEMENT_PER_VEC;
  constexpr int N_UNROLL = N_DIM / N_ELEMENT_PER_ITER;
  static_assert(TOPK < 32);

  __shared__ Element *smem_ptr[NUM_WARPS * TOPK];
  __shared__ int task_blk_rank[ALL2ALL_MAX_TASK_BLOCKS];
  __shared__ int task_blk_start_in_rank[ALL2ALL_MAX_TASK_BLOCKS];
  __shared__ int task_blk_end_in_rank[ALL2ALL_MAX_TASK_BLOCKS];
  __shared__ int task_blk_token_acc[kMaxLocalWorldSize];
  __shared__ int total_task_blk[1];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int wid = tid / 32;
  int wtid = tid % 32;
  int rank = params.index_args.rank;
  int world_size = params.index_args.world_size;
  if (tid == 0) {
    int n_tasks_acc = 0;
    int total_token_to_send = 0;
    for (int i = 0; i < world_size; i++) {
      total_token_to_send += params.index_args.token_count_send[i];
      task_blk_token_acc[i] = 0;
    }
    int n_tasks_for_avg_token = ALL2ALL_MAX_TASK_BLOCKS - params.index_args.world_size;
    int avg_token_per_task =
        (total_token_to_send + n_tasks_for_avg_token - 1) / n_tasks_for_avg_token;
    int remained_ranks = world_size;
    while (remained_ranks > 0) {
      remained_ranks = 0;
      for (int rank_offset = 0; rank_offset < world_size; rank_offset++) {
        int tgt_rank = (rank + rank_offset) % world_size;
        int n_token_to_send = params.index_args.token_count_send[tgt_rank];
        if (task_blk_token_acc[tgt_rank] < n_token_to_send) {
          // still got remained tokens to sent in current rank
          remained_ranks++;
          task_blk_rank[n_tasks_acc] = tgt_rank;
          task_blk_start_in_rank[n_tasks_acc] = task_blk_token_acc[tgt_rank];
          task_blk_end_in_rank[n_tasks_acc] =
              min(task_blk_token_acc[tgt_rank] + avg_token_per_task, n_token_to_send);
          task_blk_token_acc[tgt_rank] = task_blk_end_in_rank[n_tasks_acc];
          n_tasks_acc++;
        }
      }
    }
    total_task_blk[0] = n_tasks_acc;
  }
  __syncthreads();
  // return;
  int n_total_tasks = total_task_blk[0];
  float acc[N_ELEMENT_PER_VEC];
  Element reg128[N_ELEMENT_PER_VEC];
  Element **internal_src_ptr = &smem_ptr[wid * TOPK];

  int max_token_per_rank = params.index_args.max_token_per_rank;
  for (int task_id = bid; task_id < n_total_tasks; task_id += gridDim.x) {
    int tgt_rank = task_blk_rank[task_id];
    Element *tgt_output_ptr = reinterpret_cast<Element *>(params.output_ptrs[tgt_rank]);
    int global_token_start = params.index_args.get_global_token_start(tgt_rank);
    int global_token_end = params.index_args.get_global_token_end(tgt_rank);
    int token_start_in_rank = task_blk_start_in_rank[task_id];
    int token_end_in_rank = task_blk_end_in_rank[task_id];
    int n_token_to_send_cur_task = token_end_in_rank - token_start_in_rank;
    int gwid = wid;

    for (int cur_task_token_id = gwid; cur_task_token_id < n_token_to_send_cur_task;
         cur_task_token_id += NUM_WARPS) {
      // local idx is the local token offset within the rank
      int local_idx = cur_task_token_id + token_start_in_rank;
      int global_idx = params.index_args.token_idx_send[tgt_rank * max_token_per_rank + local_idx];
      int act_topk = params.index_args.token_local_reduce_pos[global_idx * (TOPK + 1) + TOPK];
      if (wtid < act_topk) {
        int local_scatter_pos =
            params.index_args.token_local_reduce_pos[global_idx * (TOPK + 1) + wtid];
        Element *tmp_ptr =
            reinterpret_cast<Element *>(params.input_ptr) + local_scatter_pos * N_DIM;
        internal_src_ptr[wtid] = tmp_ptr;
      }
      __syncwarp();
      // #pragma unroll
      for (int n_iter = 0; n_iter < N_UNROLL; n_iter++) {
// clear the acc
#pragma unroll
        for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
          acc[i] = 0.0f;
        }
        // perform the topk reduction here
        for (int top_id = 0; top_id < act_topk; top_id++) {
          FETCH_128bit(&reg128[0]) = FETCH_128bit(
              internal_src_ptr[top_id] + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC);
#pragma unroll
          for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
            acc[i] += element_to_float(reg128[i]);
          }
        }
#pragma unroll
        for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
          reg128[i] = float_to_element<Element>(acc[i]);
        }
        // write the output to the output buffer directly
        int64_t pos = (global_idx - global_token_start + max_token_per_rank * rank) * N_DIM +
                      n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC;
        FETCH_128bit(tgt_output_ptr + pos) = FETCH_128bit(&reg128[0]);
      }
    }
  }
}

template <typename Element, const int NUM_THREADS, const int64_t N_DIM, const int TOPK>
__global__ void
topk_reduce_kernel(int M, int32_t *reduction_idx, Element *input, Element *output) {
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int VEC_SIZE = 16;
  constexpr int WARP_SIZE = 32;

  constexpr int NUM_WARP = NUM_THREADS / WARP_SIZE;
  static_assert(VEC_SIZE % ELEMENT_SIZE == 0);
  constexpr int N_ELEMENT_PER_VEC = VEC_SIZE / ELEMENT_SIZE;
  constexpr int N_ELEMENT_PER_ITER = WARP_SIZE * N_ELEMENT_PER_VEC;
  static_assert(N_DIM % N_ELEMENT_PER_ITER == 0, "N_DIM must be divisible by N_ELEMENT_PER_ITER");
  constexpr int N_UNROLL = N_DIM / N_ELEMENT_PER_ITER;
  __shared__ __align__(16) Element *shared_buffer[TOPK * NUM_WARP];
  float acc[N_ELEMENT_PER_VEC];
  Element reg128[N_ELEMENT_PER_VEC];
  int wtid = threadIdx.x % 32;
  int wid = threadIdx.x / WARP_SIZE;
  int bid = blockIdx.x;
  int token_idx = wid + bid * NUM_WARP;
  Element **ptr = &shared_buffer[TOPK * wid];
  if (token_idx < M) {
    int topk_remained = reduction_idx[token_idx * (TOPK + 1) + TOPK];
    // preprocess the ptrs into shared memory
    if (wtid < topk_remained) {
      ptr[wtid] = input + reduction_idx[token_idx * (TOPK + 1) + wtid] * N_DIM;
    }
    __syncwarp();
#pragma unroll
    for (int n_iter = 0; n_iter < N_UNROLL; n_iter++) {
#pragma unroll
      for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
        acc[i] = 0.0f;
      }
      for (int i = 0; i < topk_remained; i++) {
        FETCH_128bit(&reg128[0]) =
            FETCH_128bit(ptr[i] + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC);
#pragma unroll
        for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
          acc[i] += element_to_float(reg128[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < N_ELEMENT_PER_VEC; i++) {
        reg128[i] = float_to_element<Element>(acc[i]);
      }
      FETCH_128bit(
          output + N_DIM * token_idx + n_iter * N_ELEMENT_PER_ITER + wtid * N_ELEMENT_PER_VEC) =
          FETCH_128bit(&reg128[0]);
    }
  }
}

void
dis_scatter_backward_impl(const DisScatterBackwardParams &params, cudaStream_t stream) {
  constexpr int NUM_THREADS = 1024;
  constexpr int BLOCK_SIZE_M = ALL2ALL_COMBINE_BLOCK;
  dim3 block_dim(NUM_THREADS);
  dim3 grid_dim(params.n_threadblocks);
  // TODO add the dispatch logic here
  assert(params.hidden_dim % 256 == 0);
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(
              cute::Int<768>{},
              cute::Int<1024>{},
              cute::Int<2048>{},
              cute::Int<3072>{},
              cute::Int<4096>{},
              cute::Int<4608>{},
              cute::Int<5120>{},
              cute::Int<5632>{},
              cute::Int<6144>{},
              cute::Int<7168>{},
              cute::Int<8192>{},
              cute::Int<9216>{},
              cute::Int<10240>{}),
          cute::make_tuple(
              cute::_4{}, cute::_5{}, cute::_6{}, cute::Int<8>{}, cute::Int<10>{})),  // topk
      [&](auto tup) {
        auto [cndim, ctopk] = tup;
        return cndim == params.hidden_dim and ctopk == params.index_args.topk;
      },
      [&](auto tup) {
        auto [cndim, ctopk] = tup;
        constexpr int TOPK = decltype(ctopk){};
        constexpr int NDIM = decltype(cndim){};
        // TODO: currently only suport BF16, but support other data types can be easily extended
        using Element = nv_bfloat16;
        int smem_size = sizeof(ShareStorage<Element, NUM_THREADS, TOPK>);
        smem_size = (smem_size + 255) / 256 * 256;
        CUDA_CHECK(cudaFuncSetAttribute(
            dis_scatter_backward_kernel_v2<Element, NUM_THREADS, NDIM, BLOCK_SIZE_M, TOPK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
        dis_scatter_backward_kernel_v2<Element, NUM_THREADS, NDIM, BLOCK_SIZE_M, TOPK>
            <<<grid_dim, block_dim, smem_size, stream>>>(params);
        // dis_scatter_backward_kernel<Element, NUM_THREADS, NDIM, BLOCK_SIZE_M, TOPK>
        //     <<<grid_dim, block_dim, 0, stream>>>(params);
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for topk=" << params.index_args.topk
                          << " n_dim:" << params.hidden_dim;
      });
}
void
dis_scatter_backward_single_node_impl(
    const DisScatterBackwardSingleNodeParams &params, cudaStream_t stream) {
  constexpr int NUM_THREADS = 1024;
  dim3 block_dim(NUM_THREADS);
  dim3 grid_dim(params.n_threadblocks);
  FLUX_CHECK(params.hidden_dim % 256 == 0);
  FLUX_CHECK(ALL2ALL_MAX_TASK_BLOCKS >= params.n_threadblocks + params.index_args.world_size);
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(
              cute::Int<768>{},
              cute::Int<1024>{},
              cute::Int<2048>{},
              cute::Int<3072>{},
              cute::Int<4096>{},
              cute::Int<4608>{},
              cute::Int<5120>{},
              cute::Int<5632>{},
              cute::Int<6144>{},
              cute::Int<7168>{},
              cute::Int<8192>{},
              cute::Int<9216>{},
              cute::Int<10240>{}),
          cute::make_tuple(
              cute::_4{}, cute::_5{}, cute::_6{}, cute::Int<8>{}, cute::Int<10>{})),  // topk
      [&](auto tup) {
        auto [cndim, ctopk] = tup;
        return cndim == params.hidden_dim and ctopk == params.index_args.topk;
      },
      [&](auto tup) {
        auto [cndim, ctopk] = tup;
        constexpr int TOPK = decltype(ctopk){};
        constexpr int NDIM = decltype(cndim){};
        // TODO: currently only suport BF16, but support other data types can be easily extended
        using Element = nv_bfloat16;
        dis_scatter_backward_single_node_kernel<Element, NUM_THREADS, NDIM, TOPK>
            <<<grid_dim, block_dim, 0, stream>>>(params);
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for topk=" << params.index_args.topk
                          << " n_dim:" << params.hidden_dim;
      });
}

void
dis_scatter_backward_build_index_impl(
    const DisScatterBackwardBuildIndexParams &params, cudaStream_t stream) {
  constexpr int NUM_THREADS = 512;
  // each thread process contiguous BLOCK_SIZE_M tokens
  // so that we can better saturate the IB bandwidth
  constexpr int BLOCK_SIZE_M = ALL2ALL_COMBINE_BLOCK;
  constexpr int TOKENS_PER_BLOCK = NUM_THREADS * BLOCK_SIZE_M;
  dim3 block_dim(NUM_THREADS);
  int n_threadblocks = (params.max_token_per_rank + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
  dim3 grid_dim(n_threadblocks);
  dis_scatter_backward_build_index_kernel<NUM_THREADS, BLOCK_SIZE_M>
      <<<grid_dim, block_dim, 0, stream>>>(params);
}

void
dis_scatter_backward_build_index_single_node_impl(
    const DisScatterBackwardBuildIndexSingleNodeParams &params, cudaStream_t stream) {
  constexpr int NUM_THREADS = 512;
  int n_threadblocks = (params.max_token_per_rank + NUM_THREADS - 1) / NUM_THREADS;
  dim3 block_dim(NUM_THREADS);
  dim3 grid_dim(n_threadblocks);
  dis_scatter_backward_build_index_single_node_kernel<NUM_THREADS>
      <<<grid_dim, block_dim, 0, stream>>>(params);
}

void
topk_reduce_impl(
    void *input,
    void *output,
    int32_t *reduce_token_idx,
    int M,
    int N,
    int topk,
    cudaStream_t stream) {
  constexpr int NUM_THREADS = 1024;
  static_assert(NUM_THREADS % 32 == 0);
  constexpr int NUM_WARP = NUM_THREADS / 32;

  dim3 block_dim(NUM_THREADS);
  int n_threadblocks = (M + NUM_WARP - 1) / NUM_WARP;
  dim3 grid_dim(n_threadblocks);
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(
              cute::Int<768>{},
              cute::Int<1024>{},
              cute::Int<2048>{},
              cute::Int<3072>{},
              cute::Int<4096>{},
              cute::Int<4608>{},
              cute::Int<5120>{},
              cute::Int<5632>{},
              cute::Int<6144>{},
              cute::Int<7168>{},
              cute::Int<8192>{},
              cute::Int<9216>{},
              cute::Int<10240>{}),
          cute::make_tuple(
              cute::_4{}, cute::_5{}, cute::_6{}, cute::Int<8>{}, cute::Int<10>{})),  // topk
      [&](auto tup) {
        auto [cndim, ctopk] = tup;
        return cndim == N and ctopk == topk;
      },
      [&](auto tup) {
        auto [cndim, ctopk] = tup;
        constexpr int TOPK = decltype(ctopk){};
        constexpr int NDIM = decltype(cndim){};
        // TODO: currently only suport BF16, but support other data types can be easily extended
        using Element = nv_bfloat16;
        Element *input_ptr = reinterpret_cast<Element *>(input);
        Element *output_ptr = reinterpret_cast<Element *>(output);
        topk_reduce_kernel<Element, NUM_THREADS, NDIM, TOPK>
            <<<grid_dim, block_dim, 0, stream>>>(M, reduce_token_idx, input_ptr, output_ptr);
      },
      [&]() { FLUX_CHECK(false) << "unsupported for topk=" << topk << " n_dim:" << N; });
}

}  // namespace flux
}  // namespace bytedance
