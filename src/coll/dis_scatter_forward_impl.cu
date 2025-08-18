//===- dis_scatter_forward_impl.cu --------------------------------------- C++ ---===//
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

#include "dis_scatter_forward_impl.hpp"
#include "flux/flux.h"

namespace bytedance {
namespace flux {

__device__ int
ld_acquire(int *ptr) {
  int state = 0;

#if (__CUDA_ARCH__ >= 700)
  /// SM70 and newer use memory consistency qualifiers

  // Acquire pattern using acquire modifier
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));

#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif  // (__CUDA_ARCH__ >= 700)

  return state;
}

template <const int NUM_THREADS, const int N_NODES, const int BLOCK_SIZE>
__global__ void
dis_scatter_forward_build_index_flatten_kernel(const DisScatterForwardBuildIndexParams params) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be divisible by 32");

  __shared__ int shared_buffer[NUM_THREADS * BLOCK_SIZE * N_NODES];
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int token_block_idx = bid * blockDim.x + tid;
  int token_idx_start = token_block_idx * BLOCK_SIZE;
  int token_idx_end = token_idx_start + BLOCK_SIZE;
  int experts_per_node = params.total_num_experts / params.n_nodes;
  int *shared_idx = &shared_buffer[tid * BLOCK_SIZE * N_NODES];
  int token_node_count[N_NODES];
  int node_flag[N_NODES];
  const int max_token = params.max_token_per_rank;

  // preprocess the indexes from the perspective of sending
#pragma unroll
  for (int i = 0; i < N_NODES; i++) {
    token_node_count[i] = 0;
  }
  if (token_idx_start < params.n_tokens_cur_ep) {
    int node_start = params.get_global_token_start(params.cur_node_id);
    int global_token_start = token_idx_start + node_start;
    int global_token_end = min(params.n_tokens_cur_ep, token_idx_end) + node_start;
    for (int global_idx = global_token_start; global_idx < global_token_end; global_idx++) {
      for (int i = 0; i < params.n_nodes; i++) {
        node_flag[i] = 0;
      }
      for (int i = 0; i < params.topk; i++) {
        int load_pos = global_idx * params.topk + i;
        int exp_id = params.ag_exp_indices[load_pos];
        int exp_node_id = exp_id / experts_per_node;
        if (node_flag[exp_node_id] == 0) {
          int write_pos = token_node_count[exp_node_id] + exp_node_id * BLOCK_SIZE;
          shared_idx[write_pos] = global_idx - node_start;
          token_node_count[exp_node_id] += 1;
          node_flag[exp_node_id] = 1;
        }
      }
    }
  }
  for (int node_id = 0; node_id < params.n_nodes; node_id++) {
    if (token_node_count[node_id] > 0) {
      int block_offset = atomicAdd(params.token_count_send + node_id, 1);
      params.block_idx_to_send[node_id * max_token + block_offset] = token_block_idx;
      params.token_count_in_block[node_id * max_token + block_offset] = token_node_count[node_id];
      for (int i = 0; i < token_node_count[node_id]; i++) {
        int write_pos = node_id * max_token + token_block_idx * BLOCK_SIZE + i;
        int load_pos = node_id * BLOCK_SIZE + i;
        params.sorted_token_idx_send[write_pos] = shared_idx[load_pos];
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N_NODES; i++) {
    token_node_count[i] = 0;
  }
  // preprocess the indexes from the perspective of receiving
  for (int nid = 0; nid < params.n_nodes; nid++) {
    int node_start = params.get_global_token_start(nid);
    int token_counts = params.get_global_token_end(nid) - node_start;
    if (token_idx_start < token_counts) {
      int global_token_start = token_idx_start + node_start;
      int global_token_end = min(token_counts, token_idx_end) + node_start;
      for (int global_idx = global_token_start; global_idx < global_token_end; global_idx++) {
        node_flag[nid] = 0;
        for (int i = 0; i < params.topk; i++) {
          int load_pos = global_idx * params.topk + i;
          int exp_id = params.ag_exp_indices[load_pos];
          int exp_node_id = exp_id / experts_per_node;
          if (exp_node_id == params.cur_node_id && node_flag[nid] == 0) {
            int write_pos = token_node_count[nid] + nid * BLOCK_SIZE;
            shared_idx[write_pos] = global_idx - node_start;
            token_node_count[nid] += 1;
            node_flag[nid] = 1;
          }
        }
      }
      if (token_node_count[nid] > 0) {
        int global_offset = atomicAdd(params.token_count_recv + nid, token_node_count[nid]);
        for (int i = 0; i < token_node_count[nid]; i++) {
          int write_offset = global_offset + i + nid * max_token;
          params.sorted_token_idx_recv[write_offset] = shared_idx[nid * BLOCK_SIZE + i];
        }
      }
    }
  }
}

template <const int NUM_THREADS, const int N_NODES, const int BLOCK_SIZE, const int SPLITS>
__global__ void
dis_scatter_forward_build_index_flatten_kernel_v2(const DisScatterForwardBuildIndexParams params) {
  static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be divisible by 32");

  __shared__ int shared_buffer[NUM_THREADS * BLOCK_SIZE * N_NODES];
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int token_block_idx = bid * blockDim.x + tid;
  int token_idx_start = token_block_idx * BLOCK_SIZE;
  int token_idx_end = token_idx_start + BLOCK_SIZE;
  int experts_per_node = params.total_num_experts / params.n_nodes;
  int *shared_idx = &shared_buffer[tid * BLOCK_SIZE * N_NODES];
  int token_node_count[N_NODES];
  int node_flag[N_NODES];
  const int max_token = params.max_token_per_rank;
  int n_nodes = params.n_nodes;
  // preprocess the indexes from the perspective of sending
#pragma unroll
  for (int i = 0; i < N_NODES; i++) {
    token_node_count[i] = 0;
  }
  int n_tokens_per_split = (params.n_tokens_cur_ep + SPLITS - 1) / SPLITS;
  int split_id = token_idx_start / n_tokens_per_split;
  if (token_idx_start < params.n_tokens_cur_ep) {
    int node_start = params.get_global_token_start(params.cur_node_id);
    int global_token_start = token_idx_start + node_start;
    int global_token_end = min(params.n_tokens_cur_ep, token_idx_end) + node_start;
    // use the first token in the block to determine the split id
    for (int global_idx = global_token_start; global_idx < global_token_end; global_idx++) {
      for (int i = 0; i < n_nodes; i++) {
        node_flag[i] = 0;
      }
      for (int i = 0; i < params.topk; i++) {
        int load_pos = global_idx * params.topk + i;
        int exp_id = params.ag_exp_indices[load_pos];
        int exp_node_id = exp_id / experts_per_node;
        if (node_flag[exp_node_id] == 0 && exp_node_id < n_nodes) {
          int write_pos = token_node_count[exp_node_id] + exp_node_id * BLOCK_SIZE;
          shared_idx[write_pos] = global_idx - node_start;
          token_node_count[exp_node_id] += 1;
          node_flag[exp_node_id] = 1;
        }
      }
    }
  }
  for (int node_id = 0; node_id < n_nodes; node_id++) {
    if (token_node_count[node_id] > 0) {
      int block_offset = atomicAdd(params.token_count_send + split_id * n_nodes + node_id, 1);
      params.block_idx_to_send[(split_id * n_nodes + node_id) * max_token + block_offset] =
          token_block_idx;
      params.token_count_in_block[(split_id * n_nodes + node_id) * max_token + block_offset] =
          token_node_count[node_id];
      for (int i = 0; i < token_node_count[node_id]; i++) {
        int write_pos = node_id * max_token + token_block_idx * BLOCK_SIZE + i;
        int load_pos = node_id * BLOCK_SIZE + i;
        params.sorted_token_idx_send[write_pos] = shared_idx[load_pos];
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N_NODES; i++) {
    token_node_count[i] = 0;
  }
  // preprocess the indexes from the perspective of receiving
  for (int nid = 0; nid < params.n_nodes; nid++) {
    int node_start = params.get_global_token_start(nid);
    int token_counts = params.get_global_token_end(nid) - node_start;
    n_tokens_per_split = (token_counts + SPLITS - 1) / SPLITS;
    split_id = token_idx_start / n_tokens_per_split;
    if (token_idx_start < token_counts) {
      int global_token_start = token_idx_start + node_start;
      int global_token_end = min(token_counts, token_idx_end) + node_start;
      for (int global_idx = global_token_start; global_idx < global_token_end; global_idx++) {
        node_flag[nid] = 0;
        for (int i = 0; i < params.topk; i++) {
          int load_pos = global_idx * params.topk + i;
          int exp_id = params.ag_exp_indices[load_pos];
          int exp_node_id = exp_id / experts_per_node;
          if (exp_node_id == params.cur_node_id && node_flag[nid] == 0) {
            int write_pos = token_node_count[nid] + nid * BLOCK_SIZE;
            shared_idx[write_pos] = global_idx - node_start;
            token_node_count[nid] += 1;
            node_flag[nid] = 1;
          }
        }
      }
      if (token_node_count[nid] > 0) {
        int global_offset =
            atomicAdd(params.token_count_recv + split_id * n_nodes + nid, token_node_count[nid]);
        for (int i = 0; i < token_node_count[nid]; i++) {
          int write_offset = global_offset + i + (split_id * n_nodes + nid) * max_token;
          params.sorted_token_idx_recv[write_offset] = shared_idx[nid * BLOCK_SIZE + i];
        }
      }
    }
  }
}

template <typename Element, const int N_SEND_WARP, const int N_DISPATCH_WARP, const int BLOCK_SIZE>
__global__
__launch_bounds__(1024, 1) void dis_scatter_forward_flatten_kernel(
    const DisScatterForwardParams params) {
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int VEC_SIZE = 16;
  static_assert(VEC_SIZE % ELEMENT_SIZE == 0);

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int wid = tid / 32;
  int wtid = tid % 32;
  int cur_node_id = params.index_args.cur_node_id;
  int n_nodes = params.index_args.n_nodes;
  int local_rank = params.rank % params.local_world_size;
  int max_token = params.index_args.max_token_per_rank;
  int expert_per_node = params.index_args.total_num_experts / n_nodes;
  int expert_per_rank = params.index_args.total_num_experts / params.world_size;
  int topk = params.index_args.topk;
  Element *symm_internal_buffer = reinterpret_cast<Element *>(params.internal_ptrs[local_rank]);
  // Element *symm_output_buffer = reinterpret_cast<Element *>(params.output_ptrs[local_rank]);
  int64_t hidden_size = params.hidden_dim;
  const int msg_size = hidden_size * sizeof(Element);
  const int *ep_cum_sum_ptr =
      params.ep_cum_sum_gpu_ptr != nullptr ? params.ep_cum_sum_gpu_ptr : params.ep_cum_sum;

  if (wid < N_SEND_WARP) {
    // send warp start from the next rank
    // time step 0: 0->1, 1->2, 2->3, 3->0
    // time step 1: 0->2, 1->3, 2->0, 3->1
    // time step 2: 0->3, 1->0, 2->1, 3->2
    int wid_send = wid % N_SEND_WARP + bid * N_SEND_WARP;
    for (int node_offset = 1; node_offset < n_nodes; node_offset++) {
      int send_node = (cur_node_id + node_offset) % n_nodes;
      int n_block_to_send = params.index_args.token_count_send[send_node];
      int dst_rank_in_team = local_rank + send_node * params.local_world_size;
      int dst_rank =
          nvshmem_team_translate_pe(params.nvshmem_team, dst_rank_in_team, NVSHMEM_TEAM_WORLD);
      for (int offset = wid_send; offset < n_block_to_send; offset += N_SEND_WARP * gridDim.x) {
        int block_info_pos = max_token * send_node + offset;
        int token_block_idx = params.index_args.block_idx_to_send[block_info_pos];
        int n_token_in_block = params.index_args.token_count_in_block[block_info_pos];
        int l_pos = 0;
        int r_pos = 0;
        int *token_idx_ptr =
            &params.index_args
                 .sorted_token_idx_send[token_block_idx * BLOCK_SIZE + send_node * max_token];
        while (l_pos < n_token_in_block) {
          int token_idx_left = token_idx_ptr[l_pos];
          r_pos = l_pos + 1;
          while (r_pos < n_token_in_block &&
                 token_idx_ptr[r_pos] - token_idx_left == (r_pos - l_pos))
            r_pos++;
          // token in [lpos, rpos) are contiguous
          Element *ptr =
              symm_internal_buffer + (token_idx_left + cur_node_id * max_token) * hidden_size;
          nvshmemx_putmem_nbi_warp(ptr, ptr, msg_size * (r_pos - l_pos), dst_rank);
          l_pos = r_pos;
        }
        // arrive increase barrier
      }
      __threadfence();
      // never mind ibrc fence is no-op, run this in a non-gloabl team does not effect the
      // performance
      nvshmem_fence();
      // put signal here
      if (wtid == 0) {
        nvshmemx_signal_op(
            reinterpret_cast<uint64_t *>(params.barrier_ptrs[local_rank]) + cur_node_id,
            1,
            NVSHMEM_SIGNAL_ADD,
            dst_rank);
      }
    }
  } else {
    int wid_dispatch = wid % N_DISPATCH_WARP + bid * N_DISPATCH_WARP;
    // dispatch warp
    for (int node_offset = 0; node_offset < params.index_args.n_nodes; node_offset++) {
      // dispatch starts from
      int recv_node = (cur_node_id + n_nodes - node_offset) % n_nodes;
      int global_token_start = params.index_args.get_global_token_start(recv_node);
      __threadfence();
      nvshmem_fence();
      if (recv_node != cur_node_id) {
        // no need to wait data for current node
        // wait data to be ready
        nvshmem_signal_wait_until(
            reinterpret_cast<uint64_t *>(params.barrier_ptrs[local_rank]) + recv_node,
            NVSHMEM_CMP_EQ,
            N_SEND_WARP * gridDim.x);
      }
      __threadfence();
      nvshmem_fence();
      // perform dispatch here
      int n_token_to_recv = params.index_args.token_count_recv[recv_node];

      for (int offset = wid_dispatch; offset < n_token_to_recv;
           offset += N_DISPATCH_WARP * gridDim.x) {
        int token_idx = params.index_args.sorted_token_idx_recv[recv_node * max_token + offset];
        int global_token_idx = token_idx + global_token_start;
        for (int topk_idx = 0; topk_idx < topk; topk_idx++) {
          int exp_id = params.index_args.ag_exp_indices[global_token_idx * topk + topk_idx];
          int exp_node_id = exp_id / expert_per_node;

          if (exp_node_id == cur_node_id) {
            int scatter_idx = params.index_args.ag_scatter_idx[global_token_idx * topk + topk_idx];
            // calculate the local_rank of current rank
            int target_ep_rank_this_team = exp_id / expert_per_rank;
            int target_ep_rank = nvshmem_team_translate_pe(
                params.nvshmem_team, target_ep_rank_this_team, NVSHMEM_TEAM_WORLD);
            int target_local_rank = target_ep_rank % params.local_world_size;
            Element *src_ptr =
                symm_internal_buffer + hidden_size * (token_idx + recv_node * max_token);
            int output_offset_m = scatter_idx - ep_cum_sum_ptr[target_local_rank];
            Element *dst_ptr = reinterpret_cast<Element *>(params.output_ptrs[local_rank]) +
                               hidden_size * output_offset_m;
            nvshmemx_putmem_nbi_warp(dst_ptr, src_ptr, msg_size, target_ep_rank);
          }
        }
      }
    }
  }
}

template <
    typename Element,
    const int N_SEND_WARP,
    const int N_DISPATCH_WARP,
    const int BLOCK_SIZE,
    const int SPLITS>
__global__
__launch_bounds__(1024, 1) void dis_scatter_forward_flatten_kernel_v2(
    const DisScatterForwardParams params) {
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int VEC_SIZE = 16;
  static_assert(VEC_SIZE % ELEMENT_SIZE == 0);

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int wid = tid / 32;
  int wtid = tid % 32;
  int cur_node_id = params.index_args.cur_node_id;
  int n_nodes = params.index_args.n_nodes;
  int local_rank = params.rank % params.local_world_size;
  int max_token = params.index_args.max_token_per_rank;
  int expert_per_node = params.index_args.total_num_experts / n_nodes;
  int expert_per_rank = params.index_args.total_num_experts / params.world_size;
  int topk = params.index_args.topk;
  Element *symm_internal_buffer = reinterpret_cast<Element *>(params.internal_ptrs[local_rank]);
  // Element *symm_output_buffer = reinterpret_cast<Element *>(params.output_ptrs[local_rank]);
  int64_t hidden_size = params.hidden_dim;
  const int msg_size = hidden_size * sizeof(Element);
  const int *ep_cum_sum_ptr =
      params.ep_cum_sum_gpu_ptr != nullptr ? params.ep_cum_sum_gpu_ptr : params.ep_cum_sum;

  if (wid < N_SEND_WARP) {
    // send warp start from the next rank
    // time step 0: 0->1, 1->2, 2->3, 3->0
    // time step 1: 0->2, 1->3, 2->0, 3->1
    // time step 2: 0->3, 1->0, 2->1, 3->2
    int wid_send = wid % N_SEND_WARP + bid * N_SEND_WARP;
    for (int node_offset = 1; node_offset < n_nodes; node_offset++) {
      int send_node = (cur_node_id + node_offset) % n_nodes;
      // send the data split by split
      for (int split_id = 0; split_id < SPLITS; split_id++) {
        int n_block_to_send = params.index_args.token_count_send[split_id * n_nodes + send_node];
        int dst_rank_in_team = local_rank + send_node * params.local_world_size;
        int dst_rank =
            nvshmem_team_translate_pe(params.nvshmem_team, dst_rank_in_team, NVSHMEM_TEAM_WORLD);
        for (int offset = wid_send; offset < n_block_to_send; offset += N_SEND_WARP * gridDim.x) {
          int block_info_pos = max_token * (split_id * n_nodes + send_node) + offset;
          int token_block_idx = params.index_args.block_idx_to_send[block_info_pos];
          int n_token_in_block = params.index_args.token_count_in_block[block_info_pos];
          int l_pos = 0;
          int r_pos = 0;
          int *token_idx_ptr =
              &params.index_args
                   .sorted_token_idx_send[token_block_idx * BLOCK_SIZE + send_node * max_token];
          while (l_pos < n_token_in_block) {
            int token_idx_left = token_idx_ptr[l_pos];
            r_pos = l_pos + 1;
            while (r_pos < n_token_in_block &&
                   token_idx_ptr[r_pos] - token_idx_left == (r_pos - l_pos))
              r_pos++;
            // token in [lpos, rpos) are contiguous
            Element *ptr =
                symm_internal_buffer + (token_idx_left + cur_node_id * max_token) * hidden_size;
            nvshmemx_putmem_nbi_warp(ptr, ptr, msg_size * (r_pos - l_pos), dst_rank);
            l_pos = r_pos;
          }
          // arrive increase barrier
        }
        __threadfence();
        nvshmem_fence();
        // put signal here
        if (wtid == 0) {
          nvshmemx_signal_op(
              reinterpret_cast<uint64_t *>(params.barrier_ptrs[local_rank]) + split_id * n_nodes +
                  cur_node_id,
              1,
              NVSHMEM_SIGNAL_ADD,
              dst_rank);
        }
      }
    }
  } else {
    int wid_dispatch = wid % N_DISPATCH_WARP + bid * N_DISPATCH_WARP;
    // dispatch warp
    for (int node_offset = 0; node_offset < params.index_args.n_nodes; node_offset++) {
      // dispatch starts from
      int recv_node = (cur_node_id + n_nodes - node_offset) % n_nodes;
      int global_token_start = params.index_args.get_global_token_start(recv_node);
      // dispatch the data split by split
      for (int split_id = 0; split_id < SPLITS; split_id++) {
        __threadfence();
        nvshmem_fence();
        if (recv_node != cur_node_id) {
          // no need to wait data for current node
          // wait data to be ready
          nvshmem_signal_wait_until(
              reinterpret_cast<uint64_t *>(params.barrier_ptrs[local_rank]) + split_id * n_nodes +
                  recv_node,
              NVSHMEM_CMP_EQ,
              N_SEND_WARP * gridDim.x);
        }
        __threadfence();
        nvshmem_fence();
        // perform dispatch here
        int n_token_to_recv = params.index_args.token_count_recv[split_id * n_nodes + recv_node];

        for (int offset = wid_dispatch; offset < n_token_to_recv;
             offset += N_DISPATCH_WARP * gridDim.x) {
          int token_idx =
              params.index_args
                  .sorted_token_idx_recv[(split_id * n_nodes + recv_node) * max_token + offset];
          int global_token_idx = token_idx + global_token_start;
          for (int topk_idx = 0; topk_idx < topk; topk_idx++) {
            int exp_id = params.index_args.ag_exp_indices[global_token_idx * topk + topk_idx];
            int exp_node_id = exp_id / expert_per_node;

            if (exp_node_id == cur_node_id) {
              int scatter_idx =
                  params.index_args.ag_scatter_idx[global_token_idx * topk + topk_idx];
              // calculate the local_rank of current rank
              int target_ep_rank_this_team = exp_id / expert_per_rank;
              int target_ep_rank = nvshmem_team_translate_pe(
                  params.nvshmem_team, target_ep_rank_this_team, NVSHMEM_TEAM_WORLD);
              int target_local_rank = target_ep_rank % params.local_world_size;
              Element *src_ptr =
                  symm_internal_buffer + hidden_size * (token_idx + recv_node * max_token);
              int output_offset_m = scatter_idx - ep_cum_sum_ptr[target_local_rank];
              Element *dst_ptr = reinterpret_cast<Element *>(params.output_ptrs[local_rank]) +
                                 hidden_size * output_offset_m;
              nvshmemx_putmem_nbi_warp(dst_ptr, src_ptr, msg_size, target_ep_rank);
            }
          }
        }
      }
    }
  }
}

__device__ void
copy_on_element_size(void *dst, void *src, int dst_pos, int src_pos, int element_size) {
  if (element_size == 4) {
    reinterpret_cast<int32_t *>(dst)[dst_pos] = reinterpret_cast<int32_t *>(src)[src_pos];
  } else if (element_size == 8) {
    reinterpret_cast<int64_t *>(dst)[dst_pos] = reinterpret_cast<int64_t *>(src)[src_pos];
  } else if (element_size == 2) {
    reinterpret_cast<nv_bfloat16 *>(dst)[dst_pos] = reinterpret_cast<nv_bfloat16 *>(src)[src_pos];
  }
}

__global__ void
dis_scatter_pre_comm_index_kernel(const DisScatterPreCommIndexParams params) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gtid = bid * blockDim.x + tid;
  int num_threads = blockDim.x * gridDim.x;
  // copy the data to the communication buffer
  // splits to comm_buffer
  const int *ep_cum_sum_ptr =
      params.ep_cum_sum_gpu_ptr != nullptr ? params.ep_cum_sum_gpu_ptr : params.ep_cum_sum;

  // topk indice and value copy
  int token_start = ep_cum_sum_ptr[params.rank];
  int token_end = ep_cum_sum_ptr[params.rank + 1];
  int idx_offset_idx = token_start * params.indice_n_dim;
  int idx_offset_val = token_start * params.values_n_dim;
  int n_tokens_cur_rank = token_end - token_start;
  int n_index_idx = n_tokens_cur_rank * params.indice_n_dim;
  int n_index_val = n_tokens_cur_rank * params.values_n_dim;
  int index_element_size = params.index_element_size;
  int value_element_size = params.value_element_size;
  // copy the data into the nvshmem communication buffer
  for (int i = gtid; i < n_index_idx; i += num_threads) {
    copy_on_element_size(
        params.ag_topk_idx_comm, params.cur_topk_idx, idx_offset_idx + i, i, index_element_size);
  }
  for (int i = gtid; i < n_index_val; i += num_threads) {
    copy_on_element_size(
        params.ag_topk_val_comm, params.cur_topk_val, idx_offset_val + i, i, value_element_size);
  }
  // wait the data copy is done
  __syncthreads();
  __threadfence();
  if (tid == 0) {
    atomicAdd(params.local_barrier, 1);
    __threadfence();
    while (ld_acquire(params.local_barrier) != gridDim.x) {
    }
  }
  __syncthreads();
  __threadfence();

  // send the data
  for (int tgt_rank = bid; tgt_rank < params.world_size; tgt_rank += gridDim.x) {
    // send data to the tgt rank
    // send the data for topk indices and values
    if (tgt_rank != params.rank) {
      char *topk_index_ptr =
          reinterpret_cast<char *>(params.ag_topk_idx_comm) + idx_offset_idx * index_element_size;
      int msg_size_idx = n_index_idx * index_element_size;
      nvshmemx_putmem_nbi_block(topk_index_ptr, topk_index_ptr, msg_size_idx, tgt_rank);
      char *topk_val_ptr =
          reinterpret_cast<char *>(params.ag_topk_val_comm) + idx_offset_val * value_element_size;
      int msg_size_val = n_index_val * value_element_size;
      nvshmemx_putmem_nbi_block(topk_val_ptr, topk_val_ptr, msg_size_val, tgt_rank);
    }
  }
}

__global__ void
dis_scatter_pre_comm_index_kernel_copy_out(const DisScatterPreCommIndexParams params) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gtid = bid * blockDim.x + tid;
  int num_threads = blockDim.x * gridDim.x;
  // copy the data to the communication buffer
  // splits to comm_buffer
  const int *ep_cum_sum_ptr =
      params.ep_cum_sum_gpu_ptr != nullptr ? params.ep_cum_sum_gpu_ptr : params.ep_cum_sum;

  // topk indice and value copy
  int index_element_size = params.index_element_size;
  int value_element_size = params.value_element_size;

  // copy the data to the target tensor
  int total_token = ep_cum_sum_ptr[params.world_size];
  int n_index_idx = total_token * params.indice_n_dim;
  int n_index_val = total_token * params.values_n_dim;
  for (int i = gtid; i < n_index_idx; i += num_threads) {
    copy_on_element_size(params.ag_topk_idx_re, params.ag_topk_idx_comm, i, i, index_element_size);
  }
  for (int i = gtid; i < n_index_val; i += num_threads) {
    copy_on_element_size(params.ag_topk_val_re, params.ag_topk_val_comm, i, i, value_element_size);
  }
}

__global__ void
shape_info_cum_sum_kernel(
    int *splits,
    int *token_per_rank,
    int *splits_cum_sum,
    int *splits_cum_sum_per_rank,
    int *token_per_rank_cum_sum,
    int total_experts,
    int rank,
    int world_size,
    int local_world_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  if (tid == 0 && bid == 0) {
    if (splits != nullptr) {
      splits_cum_sum[0] = 0;
      for (int i = 0; i < total_experts; i++) {
        splits_cum_sum[i + 1] = splits_cum_sum[i] + splits[i];
      }
    }
    if (token_per_rank != nullptr) {
      token_per_rank_cum_sum[0] = 0;
      for (int i = 0; i < world_size; i++) {
        token_per_rank_cum_sum[i + 1] = token_per_rank_cum_sum[i] + token_per_rank[i];
      }
    }
    if (splits != nullptr) {
      int num_expert_per_rank = total_experts / world_size;
      int node_id = rank / local_world_size;
      for (int i = 0; i < local_world_size; i++) {
        // dispatch only need rank-wise cum sum on its
        // local node, see shape process in the forward_impl
        // for reference
        int target_rank = i + node_id * local_world_size;
        splits_cum_sum_per_rank[i] = splits_cum_sum[target_rank * num_expert_per_rank];
      }
    }
  }
}

void
dis_scatter_forward_flatten_impl(const DisScatterForwardParams &params, cudaStream_t stream) {
  constexpr int BLOCK_SIZE = ALL2ALL_DISPATCH_BLOCK;
  if (params.world_size / params.local_world_size > 1) {
    constexpr int N_SEND_WARP = 4;
    constexpr int N_DISPATCH_WARP = 28;
    constexpr int N_THREADS = 32 * (N_SEND_WARP + N_DISPATCH_WARP);
    FLUX_CHECK(kMaxLocalWorldSize >= params.index_args.n_nodes);
    FLUX_CHECK(kMaxLocalWorldSize >= params.local_world_size);
    dim3 grid_dim(params.n_threadblocks);
    dim3 block_dim(N_THREADS);
    // dis_scatter_forward_flatten_kernel<nv_bfloat16, N_SEND_WARP, N_DISPATCH_WARP, BLOCK_SIZE>
    //     <<<grid_dim, block_dim, 0, stream>>>(params);
    dis_scatter_forward_flatten_kernel_v2<
        nv_bfloat16,
        N_SEND_WARP,
        N_DISPATCH_WARP,
        BLOCK_SIZE,
        ALL2ALL_DISPATCH_SPLITS><<<grid_dim, block_dim, 0, stream>>>(params);
  } else {
    // we only have one node, no need to use send warp
    constexpr int N_SEND_WARP = 0;
    constexpr int N_DISPATCH_WARP = 32;
    constexpr int N_THREADS = 32 * (N_SEND_WARP + N_DISPATCH_WARP);
    FLUX_CHECK(kMaxLocalWorldSize >= params.index_args.n_nodes);
    FLUX_CHECK(kMaxLocalWorldSize >= params.local_world_size);
    dim3 grid_dim(params.n_threadblocks);
    dim3 block_dim(N_THREADS);
    // dis_scatter_forward_flatten_kernel<nv_bfloat16, N_SEND_WARP, N_DISPATCH_WARP, BLOCK_SIZE>
    //     <<<grid_dim, block_dim, 0, stream>>>(params);
    dis_scatter_forward_flatten_kernel_v2<
        nv_bfloat16,
        N_SEND_WARP,
        N_DISPATCH_WARP,
        BLOCK_SIZE,
        ALL2ALL_DISPATCH_SPLITS><<<grid_dim, block_dim, 0, stream>>>(params);
  }
}

void
dis_scatter_forward_build_index_flatten_impl(
    const DisScatterForwardBuildIndexParams &params, cudaStream_t stream) {
  FLUX_CHECK(params.total_num_experts % params.n_nodes == 0);
  constexpr int BLOCK_SIZE = ALL2ALL_DISPATCH_BLOCK;
  constexpr int N_THREADS = 128;
  constexpr int TOKENS_PER_BLOCK = N_THREADS * BLOCK_SIZE;
  constexpr int MAX_N_NODES = 8;
  dim3 block_dim(N_THREADS);
  int n_threadblocks = (params.max_token_per_rank + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
  FLUX_CHECK(params.n_nodes <= MAX_N_NODES);
  dim3 grid_dim(n_threadblocks);
  // dis_scatter_forward_build_index_flatten_kernel<N_THREADS, MAX_N_NODES, BLOCK_SIZE>
  //     <<<grid_dim, block_dim, 0, stream>>>(params);
  dis_scatter_forward_build_index_flatten_kernel_v2<
      N_THREADS,
      MAX_N_NODES,
      BLOCK_SIZE,
      ALL2ALL_DISPATCH_SPLITS><<<grid_dim, block_dim, 0, stream>>>(params);
}

void
dis_scatter_pre_comm_index_impl(const DisScatterPreCommIndexParams &params, cudaStream_t stream) {
  constexpr int N_THREADS = 1024;
  dim3 block_dim(N_THREADS);
  dim3 grid_dim(params.n_threadblocks);
  dis_scatter_pre_comm_index_kernel<<<grid_dim, block_dim, 0, stream>>>(params);
  nvshmemx_barrier_on_stream(params.nvshmem_team, stream);
  dim3 grid_dim_copy_out(128);
  dis_scatter_pre_comm_index_kernel_copy_out<<<grid_dim_copy_out, block_dim, 0, stream>>>(params);
}

void
shape_info_cum_sum_impl(
    int *splits,
    int *token_per_rank,
    int *splits_cum_sum,
    int *splits_cum_sum_per_rank,
    int *token_per_rank_cum_sum,
    int total_experts,
    int rank,
    int world_size,
    int local_world_size,
    cudaStream_t stream) {
  dim3 block_dim(1);
  dim3 grid_dim(1);
  shape_info_cum_sum_kernel<<<grid_dim, block_dim, 0, stream>>>(
      splits,
      token_per_rank,
      splits_cum_sum,
      splits_cum_sum_per_rank,
      token_per_rank_cum_sum,
      total_experts,
      rank,
      world_size,
      local_world_size);
}

}  // namespace flux
}  // namespace bytedance
