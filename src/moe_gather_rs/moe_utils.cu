//===- moe_utils.cu ----------------------------------------------- C++ ---===//
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

#include <cub/cub.cuh>

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/array.h"
#include "cutlass/device_kernel.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/reduce_utils.cuh"
#include "flux/flux.h"
#include "moe_gather_rs/moe_utils.h"

namespace bytedance::flux {

template <const int NUM_THREADS>
__global__ void
ep_index_filter_kernel(
    int32_t *scatter_idx,          // input
    int32_t *pos_filtered,         // output
    int32_t *token_idx_filtered,   // output
    int32_t *total_token_acc,      // output
    int32_t *ep_n_tokens_cum_sum,  // input
    int32_t *splits_gpu_cum_sum,   // input
    int32_t *reduce_token_idx,     // output
    int32_t max_token_per_rank,
    int32_t topk,
    int32_t total_num_experts,
    int32_t world_size,
    int32_t ep_world_size,
    int32_t tp_world_size,
    int32_t cur_rank,
    int n_blocks_per_rank) {
  __shared__ int token_acc_shm[NUM_THREADS / 32];
  static_assert(NUM_THREADS % 32 == 0);
  int topk_acc = 0;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bid_in_rank = bid % n_blocks_per_rank;
  int rank_id = bid / n_blocks_per_rank;
  int wid = threadIdx.x / 32;
  int wtid = threadIdx.x % 32;
  // the token id within its rank
  int local_token_idx = bid_in_rank * blockDim.x + tid;
  int token_start_cur_rank = ep_n_tokens_cum_sum[rank_id];
  int token_end_cur_rank = ep_n_tokens_cum_sum[rank_id + 1];
  int n_token_cur_rank = token_end_cur_rank - token_start_cur_rank;
  int cur_ep_rank = cur_rank / tp_world_size;
  int global_offset = -1;
  int local_offset = -1;
  int num_experts_per_rank = total_num_experts / ep_world_size;
  int expert_rank_start = splits_gpu_cum_sum[cur_ep_rank * num_experts_per_rank];
  int expert_rank_end = splits_gpu_cum_sum[(cur_ep_rank + 1) * num_experts_per_rank];
  if (wtid == 0) {
    // perform the reduction within the warp first
    token_acc_shm[wid] = 0;
  }
  if (local_token_idx < n_token_cur_rank) {
    int global_token_idx = local_token_idx + token_start_cur_rank;
    for (int i = 0; i < topk; i++) {
      int pos = global_token_idx * topk + i;
      int scattered_pos = scatter_idx[pos];
      if (scattered_pos >= expert_rank_start && scattered_pos < expert_rank_end) {
        pos_filtered[(topk + 3) * global_token_idx + topk_acc] = scattered_pos;
        topk_acc += 1;
      }
    }
    if (topk_acc > 0) {
      // topk x [scatterd pos to be reduces] + [activated topk, local token idx to be write to,
      // rank id]
      pos_filtered[global_token_idx * (topk + 3) + topk] = topk_acc;
      pos_filtered[global_token_idx * (topk + 3) + topk + 1] = local_token_idx;
      pos_filtered[global_token_idx * (topk + 3) + topk + 2] = rank_id;
      local_offset = atomicAdd(&token_acc_shm[wid], 1);
    }
  }
  if (wtid == 0 && token_acc_shm[wid] > 0) {
    global_offset = atomicAdd(total_token_acc, token_acc_shm[wid]);
  }
  global_offset = __shfl_sync(0xFFFFFFFF, global_offset, 0);
  // write
  if (local_offset >= 0) {
    int global_token_idx = local_token_idx + token_start_cur_rank;
    token_idx_filtered[global_offset + local_offset] = global_token_idx;
  }
  // perform the index computation for the topk_reduce kernel
  if (rank_id == cur_rank && local_token_idx < n_token_cur_rank) {
    int global_token_idx = local_token_idx + token_start_cur_rank;
    int hit_rank = 0;
    for (int tgt_rank = 0; tgt_rank < world_size; tgt_rank++) {
      int tgt_ep_rank = tgt_rank / tp_world_size;
      int tgt_rank_expert_start = splits_gpu_cum_sum[tgt_ep_rank * num_experts_per_rank];
      int tgt_rank_expert_end = splits_gpu_cum_sum[(tgt_ep_rank + 1) * num_experts_per_rank];
      int hit_topk = 0;
      for (int top_id = 0; top_id < topk; top_id++) {
        int pos = global_token_idx * topk + top_id;
        int scattered_pos = scatter_idx[pos];
        if (scattered_pos >= tgt_rank_expert_start && scattered_pos < tgt_rank_expert_end) {
          hit_topk += 1;
        }
      }
      if (hit_topk > 0) {
        // when tp is enabled, the number of activated rank may be larger than topk
        int pos = tgt_rank * max_token_per_rank + local_token_idx;
        int write_pos = local_token_idx * (world_size + 1) + hit_rank;
        reduce_token_idx[write_pos] = pos;
        hit_rank += 1;
      }
    }
    // there are `hit_rank` tokens to be reduced in the second round
    reduce_token_idx[local_token_idx * (world_size + 1) + world_size] = hit_rank;
  }
}

void
ep_index_filter_impl(
    int32_t *scatter_idx,
    int32_t *pos_filtered,
    int32_t *token_idx_filtered,
    int32_t *total_token_acc,
    int32_t *ep_n_tokens_cum_sum,
    int32_t *splits_gpu_cum_sum,
    int32_t *reduce_token_idx,
    int32_t max_token_per_rank,
    int32_t topk,
    int32_t total_num_experts,
    int32_t world_size,
    int32_t ep_world_size,
    int32_t tp_world_size,
    int32_t cur_rank,
    cudaStream_t stream) {
  const int num_threads = 1024;
  const int n_blocks_per_rank = (max_token_per_rank + num_threads - 1) / num_threads;
  dim3 block_dim(num_threads);
  dim3 grid_dim(world_size * n_blocks_per_rank);
  ep_index_filter_kernel<num_threads><<<grid_dim, block_dim, 0, stream>>>(
      scatter_idx,
      pos_filtered,
      token_idx_filtered,
      total_token_acc,
      ep_n_tokens_cum_sum,
      splits_gpu_cum_sum,
      reduce_token_idx,
      max_token_per_rank,
      topk,
      total_num_experts,
      world_size,
      ep_world_size,
      tp_world_size,
      cur_rank,
      n_blocks_per_rank);
}

////////////////////////////////////////////
// copied from dis_scatter_backward, should be megerd in the future
template <typename Element, const int NUM_THREADS, const int N_DIM, const int TOPK>
__global__ void
ep_topk_reduce_kernel(int M, int32_t *reduction_idx, Element *input, Element *output) {
  constexpr int ELEMENT_SIZE = sizeof(Element);
  constexpr int VEC_SIZE = 16;
  constexpr int WARP_SIZE = 32;

  constexpr int NUM_WARP = NUM_THREADS / WARP_SIZE;
  static_assert(VEC_SIZE % ELEMENT_SIZE == 0);
  constexpr int N_ELEMENT_PER_VEC = VEC_SIZE / ELEMENT_SIZE;
  constexpr int N_ELEMENT_PER_ITER = WARP_SIZE * N_ELEMENT_PER_VEC;
  static_assert(N_DIM % N_ELEMENT_PER_ITER == 0, "N_DIM must be divisible by N_ELEMENT_PER_ITER");
  constexpr int N_UNROLL = N_DIM / N_ELEMENT_PER_ITER;
  __shared__ Element *shared_buffer[TOPK * NUM_WARP];
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
ep_topk_reduce_impl(
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
              cute::Int<1024>{},
              cute::Int<2048>{},
              cute::Int<3072>{},
              cute::Int<4096>{},
              cute::Int<5120>{},
              cute::Int<6144>{},
              cute::Int<7168>{},
              cute::Int<8192>{},
              cute::Int<10240>{}),
          cute::make_tuple(
              cute::_2{},
              cute::_4{},
              cute::_5{},
              cute::_6{},
              cute::Int<8>{},
              cute::Int<10>{},
              cute::Int<16>{})),  // topk
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
        ep_topk_reduce_kernel<Element, NUM_THREADS, NDIM, TOPK>
            <<<grid_dim, block_dim, 0, stream>>>(M, reduce_token_idx, input_ptr, output_ptr);
      },
      [&]() { FLUX_CHECK(false) << "unsupported for topk=" << topk << " n_dim:" << N; });
}

///////////////////////////////////////////

template <typename Element>
struct topk_reduce_scatter_args {
  Element *ptrs[8];
  int32_t *scatter_idx;
  int groups;
  int topk;
  Element *output;
  int M;
  int N;
};

template <typename Element, const int N, const int TOPK, const int N_THREADS, const int GROUPS>
__global__ void
topk_reduce_scatter_kernel(topk_reduce_scatter_args<Element> args) {
  static_assert(TOPK < 32);
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int wid = tid / 32;
  int wtid = tid % 32;
  constexpr int N_REG = 8;
  Element reg128[N_REG];
  constexpr int ROW_PER_BLOCK = N_THREADS / 32;
  __shared__ int scatter_shm[TOPK * ROW_PER_BLOCK];
  constexpr int SIZE_PER_STEP_COL = 32 * N_REG;
  constexpr int COL_UNROLL = N / SIZE_PER_STEP_COL;
  int SIZE_PER_STEP_ROW = ROW_PER_BLOCK * gridDim.x;
  int row_start = bid * ROW_PER_BLOCK + wid;
  Element *out = args.output;
  for (int row = row_start; row < args.M; row += SIZE_PER_STEP_ROW) {
    if (wtid < TOPK) {
      scatter_shm[wid * TOPK + wtid] = args.scatter_idx[row * TOPK + wtid];
    }

#pragma unroll
    for (int col = 0; col < COL_UNROLL; col++) {
      int col_offset = col * SIZE_PER_STEP_COL + wtid * N_REG;
      float acc[N_REG] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int group = 0; group < GROUPS; group++) {
        Element *ptr = args.ptrs[group];
#pragma unroll
        for (int topk_idx = 0; topk_idx < TOPK; topk_idx++) {
          int pos = scatter_shm[wid * TOPK + topk_idx] * N + col_offset;

          FETCH_128bit(&reg128[0]) = FETCH_128bit(ptr + pos);
#pragma unroll
          for (int i = 0; i < N_REG; i++) {
            acc[i] += element_to_float(reg128[i]);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < N_REG; i++) {
        reg128[i] = float_to_element<Element>(acc[i]);
      }
      FETCH_128bit(out + row * N + col_offset) = FETCH_128bit(&reg128[0]);
    }
  }
}

void
topk_reduce_scatter_impl(
    void **ptrs,
    int groups,
    DataTypeEnum dtype,
    int32_t *scatter_idx,
    int32_t topk,
    void *output_ptr,
    int M,
    int N) {
  constexpr int NTHREADS = 768;
  dim3 block_dim(NTHREADS);
  dim3 grid_dim(264);
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(cute::_4{}, cute::_5{}),
          cute::make_tuple(
              cute::_256{},
              cute::_512{},
              cute::_1024{},
              cute::_2048{},
              cute::Int<3072>{},
              cute::_4096{},
              cute::_8192{}),
          cute::make_tuple(cute::_1{}, cute::_2{})),
      [&](auto tup) {
        auto [cdtype, ctopk, c_ndim, c_groups] = tup;
        return cdtype == dtype and ctopk == topk and groups == c_groups and c_ndim == N;
      },
      [&](auto tup) {
        auto [cdtype, ctopk, c_ndim, c_groups] = tup;
        constexpr int TOPK = decltype(ctopk){};
        constexpr int NDIM = decltype(c_ndim){};
        constexpr int INPUTGROUPS = decltype(c_groups){};
        using Element = decltype(to_cuda_dtype(cdtype));
        topk_reduce_scatter_args<Element> args;
        for (int i = 0; i < groups; i++) {
          args.ptrs[i] = reinterpret_cast<Element *>(ptrs[i]);
        }
        args.groups = groups;
        args.scatter_idx = scatter_idx;

        args.topk = topk;
        args.output = reinterpret_cast<Element *>(output_ptr);
        args.M = M;
        args.N = N;
        topk_reduce_scatter_kernel<Element, NDIM, TOPK, NTHREADS, INPUTGROUPS>
            <<<grid_dim, block_dim>>>(args);
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for topk=" << topk << " dtype:" << dtype << " Ndim:" << N
                          << "input groups:" << groups << "\n";
      });
}

/////////////////////////////////////////////////////////////////////////////////////

void
sort_impl(
    int64_t num_elems,
    uint64_t *key_in,
    uint64_t *key_out,
    uint64_t *val_in,
    uint64_t *val_out,
    cudaStream_t stream) {
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  int begin_bit = 0;
  int end_bit = sizeof(uint64_t) * 8;
  cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      key_in,
      key_out,
      val_in,
      val_out,
      num_elems,
      begin_bit,
      end_bit,
      stream);
}

// index: [4096, ]
// finish_gather: [20480, ]
__global__ void
index_put_kernel(int64_t num_tokens, int64_t topk, index_t *index, bool *data, bool value) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  index_t idx = index[thread_idx];
  data[idx] = value;
}

void
index_put_impl(
    int64_t num_tokens,
    int64_t topk,
    index_t *index,
    bool *data,
    bool value,
    cudaStream_t stream) {
  index_put_kernel<<<num_tokens / 1024, 1024, 0, stream>>>(num_tokens, topk, index, data, value);
}

// (32, num_block), (1024, )
// scan index
// if in [offset, offset + bin_size]
// count += 1
__global__ void
calculate_prepared_nums_kernel(
    int64_t num_experts,
    int64_t num_tokens,
    int32_t *splits,
    int32_t *offsets,
    index_t *last_src,
    int32_t *prepared_nums) {
  int expert_idx = blockIdx.x;
  int num_threads_per_expert = gridDim.y * blockDim.x;
  int thread_idx = blockIdx.y * blockDim.x + threadIdx.x;
  int32_t min = 0;
  if (expert_idx != 0) {
    min = offsets[expert_idx - 1];
  }
  int32_t max = offsets[expert_idx];

  int tokens_per_thread = num_tokens / num_threads_per_expert;
  int32_t count = 0;
  for (int i = 0; i < tokens_per_thread; ++i) {
    int idx = thread_idx * tokens_per_thread + i;
    index_t src = last_src[idx];
    if (min <= src && src < max) {
      count += 1;
    }
  }
  atomicAdd(&prepared_nums[expert_idx], count);
}

void
calculate_prepared_nums_impl(
    int64_t num_experts,
    int64_t num_tokens,
    int32_t *splits,
    int32_t *offsets,
    index_t *last_src,
    int32_t *prepared_nums,
    cudaStream_t stream) {
  constexpr int num_blocks = 1;
  dim3 blocks_per_grid(num_experts, num_blocks);
  calculate_prepared_nums_kernel<<<blocks_per_grid, 1024, 0, stream>>>(
      num_experts, num_tokens, splits, offsets, last_src, prepared_nums);
}

/*
 * transport_nums: [num_experts, world_size]
 * prepared_order: [num_tokens, ]
 * prepared_offsets: [num_experts, ]
 */
__global__ void
calculate_transport_info_kernel(
    int64_t num_experts,
    int64_t world_size,
    index_t *prepared_order,
    index_t *prepared_offsets,
    index_t *transport_nums) {
  int eid = threadIdx.y;
  int rid = threadIdx.x;
  int num = 0;
  index_t begin_idx = 0;
  if (eid > 0) {
    begin_idx = prepared_offsets[eid - 1];
  }
  index_t end_idx = prepared_offsets[eid];

  for (size_t i = begin_idx; i < end_idx; ++i) {
    int dst_rank = prepared_order[i];
    if (dst_rank == rid) {
      num += 1;
    }
  }

  transport_nums[eid * world_size + rid] = num;
}

void
calculate_transport_info_impl(
    int64_t num_experts,
    int64_t world_size,
    index_t *prepared_order,
    index_t *prepared_offsets,
    index_t *transport_nums,
    cudaStream_t stream) {
  dim3 threads_per_block(world_size, num_experts);
  calculate_transport_info_kernel<<<1, threads_per_block, 0, stream>>>(
      num_experts, world_size, prepared_order, prepared_offsets, transport_nums);
}

/*
 * index: [world_size, tokens_per_rank]
 * input: [world_size, tokens_per_rank, hidden_size]
 * output: [world_size, tokens_per_rank, hidden_size]
 */
// constant_ranks = torch.arange(self.world_size, device=self.device).unsqueeze(1)
// output[constant_ranks, index] = input
template <int COPY_SIZE_PER_THREAD, int DTYPE_SIZE>
__global__ void
scatter_add_kernel(
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    index_t *index,
    void *input,
    void *output) {
  //[world_size * tokens_per_rank * num_threads_per_row] * copy_size_per_thread
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int num_threads_per_row = hidden_size / COPY_SIZE_PER_THREAD;
  int hidden_idx = global_idx % num_threads_per_row;
  int token_idx = (global_idx / num_threads_per_row) % tokens_per_rank;
  int rank_idx = global_idx / (num_threads_per_row * tokens_per_rank);

  index_t out_token_idx = index[rank_idx * tokens_per_rank + token_idx];
  int out_elem_offset = (rank_idx * tokens_per_rank + out_token_idx) * hidden_size +
                        hidden_idx * COPY_SIZE_PER_THREAD;
  int inp_elem_offset =
      (rank_idx * tokens_per_rank + token_idx) * hidden_size + hidden_idx * COPY_SIZE_PER_THREAD;
  uint4 *src_uint4 = reinterpret_cast<uint4 *>((char *)input + inp_elem_offset * DTYPE_SIZE);
  uint4 *dst_uint4 = reinterpret_cast<uint4 *>((char *)output + out_elem_offset * DTYPE_SIZE);
  // output[src_rank, out_row_idx, col_idx] = input[src_rank, row_idx, col_idx];
  // for (int i = 0; i < COPY_SIZE_PER_THREAD; ++i) {
  //   output[out_elem_idx + i] = input[inp_elem_idx + i];
  // }
  __stcg(dst_uint4, __ldcg(src_uint4));
}

/*
 * index: [tokens_per_rank, world_size]
 * input: [world_size, tokens_per_rank, hidden_size]
 * output: [world_size, tokens_per_rank, hidden_size]
 */
template <int NELEMS_PER_THREAD>
__global__ void
scatter_add_kernel1(
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    index_t *index,
    void *input,
    void *output) {
  constexpr int elem_size = 2;
  // int num_threads = gridDim.x * blockDim.x;
  int global_tidx = blockIdx.x * blockDim.x + threadIdx.x;
  // assert(total_nbytes % num_threads == 0);
  // assert(hidden_size * elem_size % NBYTES_PER_THREAD == 0);

  // (tokens_per_rank, hidden_size // NELEMS_PER_THREAD)
  int num_threads_per_row = hidden_size / NELEMS_PER_THREAD;
  int token_offset = global_tidx / num_threads_per_row;
  int hidden_offset = global_tidx % num_threads_per_row * NELEMS_PER_THREAD;

  __half2 sum{0, 0};
  for (int in_rank = 0; in_rank < world_size; in_rank++) {
    index_t in_token_offset = index[token_offset * world_size + in_rank];
    int in_elem_offset =
        ((in_rank * tokens_per_rank + in_token_offset) * hidden_size + hidden_offset) * elem_size;
    __half2 inp_elem = *reinterpret_cast<__half2 *>((char *)input + in_elem_offset);
    sum += inp_elem;
  }
  int out_elem_offset = (token_offset * hidden_size + hidden_offset) * elem_size;
  *reinterpret_cast<__half2 *>((char *)output + out_elem_offset) = sum;

  // index_t out_token_idx = index[rank_idx * tokens_per_rank + token_idx];
  // int inp_elem_offset = ((rank_idx * tokens_per_rank + token_idx) * hidden_size + hidden_idx) *
  // elem_size; int out_elem_offset = ((rank_idx * tokens_per_rank + out_token_idx) * hidden_size +
  // hidden_idx) * elem_size; constexpr int copy_times = NBYTES_PER_THREAD / sizeof(uint4);

  // #pragma unroll
  // for (int i = 0; i < copy_times; ++i) {
  //   uint4* src_uint4 = reinterpret_cast<uint4*>((char*)input + inp_elem_offset + i *
  //   sizeof(uint4)); uint4* dst_uint4 = reinterpret_cast<uint4*>((char*)output + out_elem_offset
  //   + i * sizeof(uint4));
  //   __stcg(dst_uint4, __ldcg(src_uint4));
  // }
}

void
scatter_add_impl(
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    index_t *index,
    void *input,
    void *output,
    cudaStream_t stream) {
  // constexpr static int COPY_SIZE_PER_THREAD = 8;
  // int blocks_per_grid = world_size * tokens_per_rank * hidden_size / 1024 /
  // COPY_SIZE_PER_THREAD; scatter_add_kernel<COPY_SIZE_PER_THREAD, 2><<<blocks_per_grid, 1024, 0,
  // stream>>>(world_size,
  //                                                   tokens_per_rank,
  //                                                   hidden_size,
  //                                                   index,
  //                                                   input,
  //                                                   output);

  constexpr int NELEMS_PER_THREAD = 2;
  // constexpr int elem_size = 2;
  int blocks_per_grid = tokens_per_rank * hidden_size / 1024 / NELEMS_PER_THREAD;
  scatter_add_kernel1<NELEMS_PER_THREAD><<<blocks_per_grid, 1024, 0, stream>>>(
      world_size, tokens_per_rank, hidden_size, index, input, output);
}

constexpr static int ELEM_SIZE = 2;

// printf("%" PRIu64 "\n", t);

// send_buffer: [world_size, tokens_per_rank, hidden_size]
// recv_buffer: [world_size, tokens_per_rank, hidden_size]
// transport_offsets: [world_size, ]
// transport_nbytes: [world_size, ]
__global__ void
transport_kernel(
    int64_t src_rank,
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    void *send_buffer,
    index_t *transport_offsets,
    index_t *transport_nbytes,
    void *dev_recv_buffer_ptrs[]) {
  int dst_rank = blockIdx.x;
  index_t nbytes = transport_nbytes[dst_rank];
  if (nbytes == 0) {
    return;
  }
  int num_threads_per_rank = gridDim.y * blockDim.x;
  int copy_times = ceilf(float(nbytes) / sizeof(uint4) / num_threads_per_rank);
  // printf("copy_times: %d\n", copy_times);

  int thread_idx_per_rank = blockIdx.y * blockDim.x + threadIdx.x;
  int nbytes_per_thread = copy_times * sizeof(uint4);
  // assert(nbytes % nbytes_per_thread == 0);
  if (thread_idx_per_rank * nbytes_per_thread >= nbytes) {
    return;
  }

  index_t offset = transport_offsets[dst_rank];
  void *src_ptr = reinterpret_cast<char *>(send_buffer) +
                  (dst_rank * tokens_per_rank + offset) * hidden_size * ELEM_SIZE;
  src_ptr = reinterpret_cast<char *>(src_ptr) + thread_idx_per_rank * nbytes_per_thread;

  void *dst_ptr = reinterpret_cast<char *>(dev_recv_buffer_ptrs[dst_rank]) +
                  (src_rank * tokens_per_rank + offset) * hidden_size * ELEM_SIZE;
  dst_ptr = reinterpret_cast<char *>(dst_ptr) + thread_idx_per_rank * nbytes_per_thread;

  for (int i = 0; i < copy_times; ++i) {
    uint4 *src_uint4 = reinterpret_cast<uint4 *>(src_ptr) + i;
    uint4 *dst_uint4 = reinterpret_cast<uint4 *>(dst_ptr) + i;
    __stcg(dst_uint4, __ldcg(src_uint4));
  }
  // memcpy(dst_ptr, src_ptr, nbytes_per_thread);
  // nvshmem_uint8_put_signal_nbi(static_cast<uint8_t*>(dst_ptr),
  //                              static_cast<uint8_t*>(src_ptr),
  //                              nbytes_per_thread,
  //                             reinterpret_cast<uint64_t*>(sig_addr),               // uint64_t*
  //                             1,                      // uint64_t
  //                             NVSHMEM_SIGNAL_ADD,     // int
  //                             dst_rank                // int
  //                             );

  // if (blockIdx.y == 0 && threadIdx.x == 0) {
  //   printf("src_rank: %d, dst_rank: %d, nbytes_per_thread: %d\n", static_cast<int>(src_rank),
  //   dst_rank, nbytes_per_thread);
  //   // printf("on device, send_buffer: %" PRIu64 "\n", reinterpret_cast<uint64_t>(send_buffer));
  //   // printf("on device, recv_buffer: %" PRIu64 "\n",
  //   reinterpret_cast<uint64_t>(dev_recv_buffer_ptrs[dst_rank]));
  //   // printf("on device, offset: %" PRId64 "\n", offset);
  //   // printf("on device, nbytes: %" PRId64 "\n", nbytes);
  //   printf("on device, signal address: %" PRIu64 "\n", reinterpret_cast<uint64_t>(sig_addr));
  //   // printf("on device, src_rank: %d, dst_rank: %d, send_buffer: %" PRIu64 ", recv_buffer: %"
  //   PRIu64 ", offset: %" PRId64 ", nbytes: %" PRId64 "\n",
  //   //       src_rank,
  //   //       dst_rank,
  //   //       reinterpret_cast<uint64_t>(send_buffer),
  //   //       reinterpret_cast<uint64_t>(dev_recv_buffer_ptrs[dst_rank]),
  //   //       offset,
  //   //       nbytes);
  // }
  // cudaMemcpyAsync(dst_ptr, src_ptr, nbytes, cudaMemcpyDeviceToDevice);
}

// self.transport_op.forward(self.send_buffer, transport_offsets_cpu, transport_nbytes_cpu, idx,
// do_barrier)
void
transport_impl(
    int64_t src_rank,
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    void *send_buffer,
    index_t *transport_offsets,
    index_t *transport_nbytes,
    void *dev_recv_buffer_ptrs[],
    cudaStream_t stream) {
  // for (int i = 0; i < world_size; ++i) {
  //   printf("on host, transport_impl, src_rank: %d, dst_rank: %d, send_buffer: %" PRIu64 "\n",
  //         static_cast<int>(src_rank),
  //         i,
  //         reinterpret_cast<uint64_t>(send_buffer));
  // }
  // constexpr int num_blocks_per_rank = 96;
  constexpr int num_blocks_per_rank = 120;
  dim3 blocks_per_grid(world_size, num_blocks_per_rank);
  transport_kernel<<<blocks_per_grid, 1024, 0, stream>>>(
      src_rank,
      world_size,
      tokens_per_rank,
      hidden_size,
      send_buffer,
      transport_offsets,
      transport_nbytes,
      dev_recv_buffer_ptrs);
}

__global__ void
calc_moe_triton_blocked_gather_a_kernel(
    const int32_t *__restrict__ splits,
    int32_t ep_start,
    int32_t ep_count,
    int32_t block_size_m,
    int32_t *__restrict__ gather_a_index,
    int32_t *__restrict__ expert_index) {
  extern __shared__ char shm_buf[];
  const int *ep_splits = splits + ep_start;
  int32_t *ep_splits_acc = reinterpret_cast<int32_t *>(shm_buf);
  int32_t *ep_splits_pad_acc = ep_splits_acc + ep_count;
  block_prefix_sum_and_sync(ep_splits, ep_splits_acc, ep_count);
  aligned_block_prefix_sum_and_sync(ep_splits, ep_splits_pad_acc, ep_count, block_size_m);

  for (int expert_id = blockIdx.x; expert_id < ep_count; expert_id += gridDim.x) {
    const int m_start = expert_id == 0 ? 0 : ep_splits_acc[expert_id - 1];
    const int m_start_pad = expert_id == 0 ? 0 : ep_splits_pad_acc[expert_id - 1];
    const int mlen = ep_splits[expert_id];
    const int tile_m_start = m_start_pad / block_size_m;
    const int tile_count = (mlen + block_size_m - 1) / block_size_m;
    for (int i = threadIdx.x; i < tile_count * block_size_m; i += blockDim.x) {
      int m = m_start + i;
      int m_pad = m_start_pad + i;
      gather_a_index[m_pad] = i < mlen ? m : INT32_MAX;
    }
    for (int i = threadIdx.x; i < tile_count; i += blockDim.x) {
      expert_index[i + tile_m_start] = expert_id;
    }
  }
}

void
calc_moe_triton_blocked_gather_a(
    const int32_t *splits,
    int32_t ep_start,
    int32_t ep_count,
    int32_t block_size_m,
    int32_t *gather_a_index,
    int32_t *expert_index,
    int num_blocks,
    int num_threads,
    cudaStream_t stream) {
  int shared_mem_size = sizeof(int) * ep_count * 2;
  calc_moe_triton_blocked_gather_a_kernel<<<num_blocks, num_threads, shared_mem_size, stream>>>(
      splits, ep_start, ep_count, block_size_m, gather_a_index, expert_index);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace bytedance::flux
