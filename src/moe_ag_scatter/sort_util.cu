//===- sort_util.cu ----------------------------------------------- C++ ---===//
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

#include <cuda_fp16.h>
#include <cutlass/device_kernel.h>

#include <algorithm>
#include <climits>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <iterator>
#include <numeric>
#include <set>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/reduce_utils.cuh"
#include "flux/flux.h"
#include "flux/utils.h"
#include "sort_util.h"

namespace bytedance::flux {
using namespace cute;

template <int MaxNExperts>
__global__ void
calc_gather_index_kernel(
    int32_t nexperts,
    int32_t ntokens,
    int32_t topk,
    int32_t expert_idx_start,
    int32_t expert_idx_end,
    const int32_t *splits,
    const int32_t *scatter_index,
    int32_t *gather_index_ep,
    int32_t *total_nrows_ep) {
  __shared__ int32_t splits_presum[MaxNExperts];

  auto splits_ts = make_tensor(make_gmem_ptr(splits), make_shape(nexperts));
  int const warp_idx = threadIdx.x / 32;
  int const lane_idx = threadIdx.x % 32;

  if (warp_idx == 0) {
    int cur_offset = 0;
    int experts_pad = pad_to(nexperts, 32);
    for (int i = lane_idx; i < experts_pad; i += 32) {
      int len = i < nexperts ? splits_ts(i) : 0;
      int temp_offset = warp_prefix_sum(threadIdx.x, len);
      if (i < nexperts) {
        splits_presum[i] = cur_offset + temp_offset;
      }
      cur_offset += __shfl_sync(0xffffffff, temp_offset, 31);
    }
  }
  __syncthreads();

  int rows_start = expert_idx_start == 0 ? 0 : splits_presum[expert_idx_start - 1];
  int rows_end = expert_idx_end == 0 ? 0 : splits_presum[expert_idx_end - 1];

  int const total_threads = gridDim.x * blockDim.x;
  int const thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_idx == 0) {
    *total_nrows_ep = rows_end - rows_start;
  }

  Tensor gather_ts = make_tensor(make_gmem_ptr(gather_index_ep), make_shape(ntokens * topk));
  Tensor scatter_ts =
      make_tensor(make_gmem_ptr(scatter_index), make_shape(make_shape(topk, ntokens)));

  for (int i = thread_idx; i < ntokens * topk; i += total_threads) {
    int dst_row = scatter_ts(i);
    auto [topki, source_row] = get<0>(scatter_ts.get_hier_coord(i));

    if (rows_start <= dst_row and dst_row < rows_end) {
      gather_ts(dst_row - rows_start) = source_row;
    }
  }
}

template <int MaxNExperts>
__global__ void
calc_gather_index_kernel_v2(
    int32_t ntokens,
    int32_t topk,
    int32_t rows_start,
    int32_t rows_end,
    const int32_t *scatter_index,
    int32_t *gather_index_ep) {
  int const total_threads = gridDim.x * blockDim.x;
  int const thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int M_this_ep = rows_end - rows_start;

  Tensor gather_ts = make_tensor(make_gmem_ptr(gather_index_ep), make_shape(M_this_ep));
  Tensor scatter_ts =
      make_tensor(make_gmem_ptr(scatter_index), make_shape(make_shape(topk, ntokens)));

  for (int i = thread_idx; i < ntokens * topk; i += total_threads) {
    int dst_row = scatter_ts(i);
    auto [topki, source_row] = get<0>(scatter_ts.get_hier_coord(i));

    if (rows_start <= dst_row && dst_row < rows_end) {
      gather_ts(dst_row - rows_start) = source_row;
    }
  }
}

void
calc_gather_index_impl_v2(
    int32_t nexperts,
    int32_t ntokens,
    int32_t topk,
    int32_t rows_start,
    int32_t rows_end,
    int32_t const *scatter_index,
    int32_t *gather_index_ep,
    cudaStream_t stream) {
  dim3 grid;
  grid.x = 32;
  dim3 block;
  block.x = 256;

  constexpr int MaxNExperts = 1024;
  FLUX_CHECK_LE(nexperts, MaxNExperts);
  calc_gather_index_kernel_v2<MaxNExperts><<<grid, block, 0, stream>>>(
      ntokens, topk, rows_start, rows_end, scatter_index, gather_index_ep);
}

void
calc_gather_index_impl(
    int32_t nexperts,
    int32_t ntokens,
    int32_t topk,
    int32_t expert_idx_start,
    int32_t expert_idx_end,
    int32_t const *splits,
    int32_t const *scatter_index,
    int32_t *gather_index_ep,
    int32_t *total_nrows_ep_gpu,  // scalar
    cudaStream_t stream) {
  dim3 grid;
  grid.x = 32;
  dim3 block;
  block.x = 256;

  constexpr int MaxNExperts = 1024;
  FLUX_CHECK_LE(nexperts, MaxNExperts);
  calc_gather_index_kernel<MaxNExperts><<<grid, block, 0, stream>>>(
      nexperts,
      ntokens,
      topk,
      expert_idx_start,
      expert_idx_end,
      splits,
      scatter_index,
      gather_index_ep,
      total_nrows_ep_gpu);
}

struct AgScatterSortOp {
  static constexpr int ThreadsInWarp = 32;
  static constexpr int Warps = 8;
  static constexpr int ThreadsInBlock = ThreadsInWarp * Warps;
  static constexpr unsigned FullMask = 0xffffffff;
  static constexpr int MaxNExperts = 1024;
  static constexpr int MaxTpRanks = 64;

  using Arguments = AGScatterSortOpArguments;
  struct Params : public Arguments {
    int32_t ntokens_perrank;
    int32_t tp_size;
  };

  struct SharedStorage {
    // in splitted problem, number of rows of (cur_rank,expert_id)
    int rows_counter[MaxTpRanks];
    // presum of splits, used to find the expert_id of a given row idx
    int splits_presum[MaxNExperts];
  };

  static Params
  to_underlying_arguments(Arguments const &args) {
    int32_t tp_size = args.dist_env.world_size;
    FLUX_CHECK(args.nexperts_ep <= MaxNExperts);
    FLUX_CHECK(args.ntokens % tp_size == 0) << args.ntokens << " % " << tp_size << " != 0";
    int32_t ntokens_perrank = args.ntokens / tp_size;
    return {args, ntokens_perrank, tp_size};
  }

  static dim3
  get_grid_shape(Arguments const &args) {
    dim3 grid;
    grid.x = args.nexperts_ep;
    return grid;
  }

  static dim3
  get_block_shape(Arguments const &args) {
    dim3 block;
    block.x = ThreadsInBlock;
    return block;
  }

  static constexpr int
  get_smem_size() {
    return sizeof(SharedStorage);
  }

  int const thread_idx;
  int const lane_idx;
  int const warp_idx;
  int const cur_expert_id;

  CUTLASS_DEVICE
  AgScatterSortOp()
      : thread_idx(threadIdx.x),
        lane_idx(thread_idx % ThreadsInWarp),
        warp_idx(thread_idx / ThreadsInWarp),
        cur_expert_id(blockIdx.x) {}

  CUTLASS_DEVICE void
  operator()(Params const &params, char *smem_buf) {
    this->calc_splits_presum(params, smem_buf);
    this->calc_outputs(params, smem_buf);
  }

 private:
  CUTLASS_DEVICE void
  calc_splits_presum(Params const &params, char *smem_buf) {
    auto &smem = *reinterpret_cast<SharedStorage *>(smem_buf);
    auto splits = make_tensor(make_gmem_ptr(params.splits_ep), make_shape(params.nexperts_ep));

    if (warp_idx == 0) {
      int cur_offset = 0;
      int experts_pad = pad_to(params.nexperts_ep, ThreadsInWarp);
      for (int i = lane_idx; i < experts_pad; i += ThreadsInWarp) {
        int len = i < params.nexperts_ep ? splits[i] : 0;
        int temp_offset = warp_prefix_sum(thread_idx, len);
        if (i < params.nexperts_ep) {
          smem.splits_presum[i] = cur_offset + temp_offset;
        }
        cur_offset += __shfl_sync(FullMask, temp_offset, 31);
      }
      for (int i = lane_idx; i < params.tp_size; i += ThreadsInWarp) {
        if (i < params.tp_size) {
          smem.rows_counter[i] = 0;
        }
      }
    }
    __syncthreads();
  }

  CUTLASS_DEVICE void
  calc_outputs(Params const &params, char *smem_buf) {
    auto &smem = *reinterpret_cast<SharedStorage *>(smem_buf);
    // calculate offset of each problem
    // iterate rows, update counter and get offset of rows in the problem
    Tensor gather_index = make_tensor(make_gmem_ptr(params.gather_index_ep), make_shape(0));
    Tensor sorted_splits = make_tensor(
        make_gmem_ptr(params.sorted_splits), make_shape(params.tp_size, params.nexperts_ep));
    Tensor sorted_scatter_index =
        make_tensor(make_gmem_ptr(params.sorted_scatter_index), make_shape(0));
    // use sorted_gather_index as temp workspace to store
    // mappings from original total_nrows to sorted.
    Tensor sorted_gather_index =
        make_tensor(make_gmem_ptr(params.sorted_gather_index), make_shape(0));

    int const row_end = smem.splits_presum[cur_expert_id];
    int const row_start = cur_expert_id == 0 ? 0 : smem.splits_presum[cur_expert_id - 1];
    // int const source_row_offset = params.ntokens_perrank * cur_rank;

    for (int i = row_start + thread_idx; i < row_end; i += ThreadsInBlock) {
      int source_row = gather_index[i];
      int source_rank = source_row / params.ntokens_perrank;
      int shift_rank = shift_rank_to_order(source_rank, params.dist_env);
      int idx = atomicAdd(&smem.rows_counter[shift_rank], 1);
      sorted_gather_index[i] = idx;  // row -> inner index <Expert, Rank>
    }

    __syncthreads();
    for (int i = thread_idx; i < params.tp_size; i += ThreadsInBlock) {
      sorted_splits(i, cur_expert_id) = smem.rows_counter[i];
    }

    // calculate presum of rows_counter as offsets
    if (warp_idx == 0) {
      int cur_offset = 0;
      int tp_size_pad = pad_to(params.tp_size, ThreadsInWarp);
      for (int i = lane_idx; i < tp_size_pad; i += ThreadsInWarp) {
        int len = i < params.tp_size ? smem.rows_counter[i] : 0;
        int temp_offset = warp_prefix_sum(thread_idx, len);
        if (i < params.tp_size) {
          smem.rows_counter[i] = cur_offset + temp_offset - len;  // row_counter cumsum
        }
        cur_offset += __shfl_sync(FullMask, temp_offset, 31);
      }
    }
    __syncthreads();

    for (int i = row_start + thread_idx; i < row_end; i += ThreadsInBlock) {
      int source_row = gather_index[i];
      int source_rank = source_row / params.ntokens_perrank;
      int shift_rank = shift_rank_to_order(source_rank, params.dist_env);
      int idx = sorted_gather_index[i];  // local index
      idx += smem.rows_counter[shift_rank];
      idx += row_start;  // global index of sorted_gather_index
      // mapping f: sorted_gather_index -> gather_index. output[i] = output_sorted[f(i)]
      sorted_scatter_index[idx] = i;
    }
    __syncthreads();

    // from sorted_scatter_index -> sorted_gather_index
    for (int i = row_start + thread_idx; i < row_end; i += ThreadsInBlock) {
      int scatter_row = sorted_scatter_index[i];  //
      int source_row = gather_index[scatter_row];
      // sorted_gather_index: mapping from sorted_scatter_index to token_index
      // input_token[i] = input_sorted[sorted_gather_index[i]]
      sorted_gather_index[i] = source_row;
    }
    __syncthreads();
  }

 public:
  // kernel launch related
  static constexpr int MaxThreadsPerBlock = ThreadsInBlock;
  static constexpr int MinBlocksPerMultiprocessor = 0;
};

struct AgScatterSortOpV2 {
  static constexpr int ThreadsInWarp = 32;
  static constexpr int Warps = 8;
  static constexpr int ThreadsInBlock = ThreadsInWarp * Warps;
  static constexpr unsigned FullMask = 0xffffffff;
  static constexpr int MaxNExperts = 1024;
  static constexpr int MaxTpRanks = 64;

  using Arguments = AGScatterSortOpArgumentsV2;
  struct Params : public Arguments {
    int32_t ntokens_perrank;
    int32_t tp_size;
  };

  struct SharedStorage {
    // in splitted problem, number of rows of (cur_rank,expert_id)
    int rows_counter[MaxTpRanks];
    // presum of splits, used to find the expert_id of a given row idx
    int splits_presum[MaxNExperts];
  };

  static Params
  to_underlying_arguments(Arguments const &args) {
    int32_t tp_size = args.world_size;
    FLUX_CHECK(args.nexperts_ep <= MaxNExperts);
    FLUX_CHECK(args.ntokens % tp_size == 0) << args.ntokens << " % " << tp_size << " != 0";
    int32_t ntokens_perrank = args.ntokens / tp_size;
    return {args, ntokens_perrank, tp_size};
  }

  static dim3
  get_grid_shape(Arguments const &args) {
    dim3 grid;
    grid.x = args.nexperts_ep;
    return grid;
  }

  static dim3
  get_block_shape(Arguments const &args) {
    dim3 block;
    block.x = ThreadsInBlock;
    return block;
  }

  static constexpr int
  get_smem_size() {
    return sizeof(SharedStorage);
  }

  int const thread_idx;
  int const lane_idx;
  int const warp_idx;
  int const cur_expert_id;

  CUTLASS_DEVICE
  AgScatterSortOpV2()
      : thread_idx(threadIdx.x),
        lane_idx(thread_idx % ThreadsInWarp),
        warp_idx(thread_idx / ThreadsInWarp),
        cur_expert_id(blockIdx.x) {}

  CUTLASS_DEVICE void
  operator()(Params const &params, char *smem_buf) {
    this->calc_splits_presum(params, smem_buf);
    this->calc_outputs(params, smem_buf);
  }

 private:
  CUTLASS_DEVICE void
  calc_splits_presum(Params const &params, char *smem_buf) {
    auto &smem = *reinterpret_cast<SharedStorage *>(smem_buf);
    for (int i = lane_idx; i < params.tp_size; i += ThreadsInWarp) {
      smem.rows_counter[i] = 0;
    }
    block_prefix_sum_and_sync(params.splits_ep, &smem.splits_presum[0], params.nexperts_ep);
  }

  CUTLASS_DEVICE void
  calc_outputs(Params const &params, char *smem_buf) {
    auto &smem = *reinterpret_cast<SharedStorage *>(smem_buf);
    // calculate offset of each problem
    // iterate rows, update counter and get offset of rows in the problem
    Tensor gather_index = make_tensor(make_gmem_ptr(params.gather_index_ep), make_shape(0));
    Tensor sorted_splits = make_tensor(
        make_gmem_ptr(params.sorted_splits), make_shape(params.tp_size, params.nexperts_ep));
    Tensor sorted_splits_cumsum = make_tensor(
        make_gmem_ptr(params.sorted_splits_cumsum),
        make_shape(params.tp_size, params.nexperts_ep));
    Tensor sorted_scatter_index =
        make_tensor(make_gmem_ptr(params.sorted_scatter_index), make_shape(0));
    // use sorted_gather_index as temp workspace to store
    // mappings from original total_nrows to sorted.
    Tensor sorted_gather_index =
        make_tensor(make_gmem_ptr(params.sorted_gather_index), make_shape(0));

    int const row_end = smem.splits_presum[cur_expert_id];
    int const row_start = cur_expert_id == 0 ? 0 : smem.splits_presum[cur_expert_id - 1];
    // int const source_row_offset = params.ntokens_perrank * cur_rank;

    for (int i = row_start + thread_idx; i < row_end; i += ThreadsInBlock) {
      int source_row = gather_index[i];
      int source_rank = source_row / params.ntokens_perrank;
      int idx = atomicAdd(&smem.rows_counter[source_rank], 1);
      sorted_gather_index[i] = idx;  // row -> inner index <Expert, Rank>
    }

    __syncthreads();
    for (int i = thread_idx; i < params.tp_size; i += ThreadsInBlock) {
      sorted_splits(i, cur_expert_id) = smem.rows_counter[i];
    }

    // calculate presum of rows_counter as offsets
    if (warp_idx == 0) {
      int cur_offset = 0;
      int tp_size_pad = pad_to(params.tp_size, ThreadsInWarp);
      for (int i = lane_idx; i < tp_size_pad; i += ThreadsInWarp) {
        int len = i < params.tp_size ? smem.rows_counter[i] : 0;
        int temp_offset = warp_prefix_sum(thread_idx, len);
        if (i < params.tp_size) {
          smem.rows_counter[i] = cur_offset + temp_offset - len;  // row_counter cumsum
        }
        cur_offset += __shfl_sync(FullMask, temp_offset, 31);
      }
    }
    __syncthreads();
    for (int i = thread_idx; i < params.tp_size; i += ThreadsInBlock) {
      sorted_splits_cumsum(i, cur_expert_id) =
          smem.rows_counter[i] + sorted_splits(i, cur_expert_id);  // real cumsum
    }

    for (int i = row_start + thread_idx; i < row_end; i += ThreadsInBlock) {
      int source_row = gather_index[i];
      int source_rank = source_row / params.ntokens_perrank;
      int idx = sorted_gather_index[i];  // local index
      idx += smem.rows_counter[source_rank];
      idx += row_start;  // global index of sorted_gather_index
      // mapping f: sorted_gather_index -> gather_index. output[i] = output_sorted[f(i)]
      sorted_scatter_index[idx] = i;
    }
    __syncthreads();

    // from sorted_scatter_index -> sorted_gather_index
    for (int i = row_start + thread_idx; i < row_end; i += ThreadsInBlock) {
      int scatter_row = sorted_scatter_index[i];  //
      int source_row = gather_index[scatter_row];
      // sorted_gather_index: mapping from sorted_scatter_index to token_index
      // input_token[i] = input_sorted[sorted_gather_index[i]]
      sorted_gather_index[i] = source_row;
    }
    __syncthreads();
  }

 public:
  // kernel launch related
  static constexpr int MaxThreadsPerBlock = ThreadsInBlock;
  static constexpr int MinBlocksPerMultiprocessor = 0;
};

void
ag_scatter_sort_impl(AGScatterSortOpArguments const &args, cudaStream_t stream) {
  using Op = AgScatterSortOp;
  auto params = Op::to_underlying_arguments(args);
  dim3 const grid = Op::get_grid_shape(args);
  dim3 const block = Op::get_block_shape(args);
  constexpr int smem_size = Op::get_smem_size();
  cutlass::device_kernel<Op><<<grid, block, smem_size, stream>>>(params);
}

void
ag_scatter_sort_impl_v2(AGScatterSortOpArgumentsV2 const &args, cudaStream_t stream) {
  using Op = AgScatterSortOpV2;
  auto params = Op::to_underlying_arguments(args);
  dim3 const grid = Op::get_grid_shape(args);
  dim3 const block = Op::get_block_shape(args);
  constexpr int smem_size = Op::get_smem_size();
  cutlass::device_kernel<Op><<<grid, block, smem_size, stream>>>(params);
}

std::vector<ProblemSchedule>
get_sorted_problem_schedule(
    std::vector<int32_t> const &sorted_splits_cpu,
    DistEnv const &dist_env,
    int32_t nexperts_ep,
    int32_t tile_size,
    int32_t ntiles) {
  int tp_size = dist_env.world_size;
  FLUX_CHECK_EQ(sorted_splits_cpu.size(), nexperts_ep * tp_size);

  std::vector<int32_t> splits_presum(sorted_splits_cpu.size() + 1);
  splits_presum[0] = 0;
  std::partial_sum(sorted_splits_cpu.begin(), sorted_splits_cpu.end(), splits_presum.begin() + 1);
  Tensor splits = make_tensor(sorted_splits_cpu.data(), make_shape(tp_size, nexperts_ep));

  // current and next states of experts
  std::vector<int32_t> split_idxs(nexperts_ep, 0);
  std::vector<int32_t> nxt_split_idxs(nexperts_ep, 0);
  std::vector<int32_t> rem_split_sizes(nexperts_ep, 0);
  std::vector<int32_t> nxt_rem_split_sizes(nexperts_ep, 0);
  std::vector<int32_t> offsets(nexperts_ep, 0);
  std::vector<int32_t> nxt_offsets(nexperts_ep, 0);

  auto is_next_split_used = [&](int e) {
    return int(
        nxt_split_idxs[e] < tp_size and nxt_rem_split_sizes[e] != splits(nxt_split_idxs[e], e));
  };

  auto comp = [&](int i, int j) {
    int require_i = nxt_split_idxs[i] + is_next_split_used(i);
    int require_j = nxt_split_idxs[j] + is_next_split_used(j);
    return std::make_tuple(require_i, i) < std::make_tuple(require_j, j);
  };

  // store index of experts, make sure that when an expert is in the set,
  // its states must not be modified
  std::set<int, decltype(comp)> pool(comp);

  auto update_next_states = [&](int e) {
    FLUX_CHECK(pool.find(e) == pool.end());
    int exp_problem_size_rem = tile_size * ntiles;
    int split_size_rem = rem_split_sizes[e];
    int split_idx = split_idxs[e];
    int offset = offsets[e];

    if (split_idx >= tp_size) {
      return;
    }

    if (split_size_rem > exp_problem_size_rem) {
      // current split is more thant enough
      nxt_split_idxs[e] = split_idx;
      nxt_rem_split_sizes[e] = split_size_rem - exp_problem_size_rem;
      nxt_offsets[e] = offset + exp_problem_size_rem;
      return;
    }

    // need a new split
    int split_idx_end = tp_size;
    int accum = split_size_rem;
    for (int i = split_idx + 1; i <= tp_size; ++i) {
      if (accum >= exp_problem_size_rem) {
        split_idx_end = i;
        break;
      }
      if (i < tp_size) {
        accum += splits(i, e);
      }
    }
    // consume remaining of current split
    offset += split_size_rem;
    ++split_idx;
    exp_problem_size_rem -= split_size_rem;
    split_size_rem = split_idx < tp_size ? splits(split_idx, e) : 0;

    while (split_idx < split_idx_end) {
      if (split_size_rem <= exp_problem_size_rem) {
        exp_problem_size_rem -= split_size_rem;
        offset += split_size_rem;
        ++split_idx;
        split_size_rem = split_idx < tp_size ? splits(split_idx, e) : 0;
      } else {
        offset += exp_problem_size_rem;
        split_size_rem -= exp_problem_size_rem;
        break;
      }
    }

    nxt_split_idxs[e] = split_idx;
    nxt_rem_split_sizes[e] = split_size_rem;
    nxt_offsets[e] = offset;
  };

  // initialize states
  for (int e = 0; e < nexperts_ep; ++e) {
    split_idxs[e] = 0;
    rem_split_sizes[e] = splits(0, e);
    offsets[e] = splits_presum[e * tp_size];
    update_next_states(e);
    pool.insert(e);
  }

  std::vector<ProblemSchedule> schedule;
  while (!pool.empty()) {
    int e = *pool.begin();
    pool.erase(pool.begin());

    schedule.emplace_back(
        ProblemSchedule{
            .expert_id = e,
            .m_start = offsets[e],
            .m_end = nxt_offsets[e],
            .source_rank_start = split_idxs[e],
            .source_rank_end = nxt_split_idxs[e] + is_next_split_used(e)});

    split_idxs[e] = nxt_split_idxs[e];
    rem_split_sizes[e] = nxt_rem_split_sizes[e];
    offsets[e] = nxt_offsets[e];
    update_next_states(e);
    if (split_idxs[e] < tp_size) {
      pool.insert(e);
    }
  }

  return schedule;
}

std::vector<ProblemSchedule>
get_sorted_problem_schedule_v2(
    const int32_t *const splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups) {
  // start from `rank` segment, and leaves the tile cross bounder to last stage
  // what about tile that cross multi segments?
  //  for first stage: always leaves the tile cross rank to next stages. won't wait for more
  //  signals for other stage: if tile all in this stage, process it, otherwise leaves the tile to
  //  next stages
  std::vector<ProblemSchedule> problem_schedules;
  problem_schedules.reserve(tp_size * num_weight_groups * ep_nexperts);
  for (int stage = 0; stage < tp_size; stage++) {
    int segment = (stage + rank) % tp_size;
    for (int gid = 0; gid < num_weight_groups; gid++) {
      for (int eid = 0; eid < ep_nexperts; eid++) {
        const int *cumsum_this_rank = cumsum_per_rank_ptr + eid * tp_size;
        auto get_cumsum_this_rank_with_zero_pad = [=](int segment) {
          return segment == 0 ? 0 : cumsum_this_rank[segment - 1];
        };
        auto get_rank_id = [=](int m) {
          auto iter = std::upper_bound(cumsum_this_rank, cumsum_this_rank + tp_size, m);
          return std::distance(cumsum_this_rank, iter);
        };
        auto get_stage_for_tile = [=](int tiled_m) {
          int m_start_this_tile = tiled_m * tiled_m_size;
          int m_end_this_tile = std::min(splits[eid + ep_start], (tiled_m + 1) * tiled_m_size) - 1;
          int segment_start_this_tile = get_rank_id(m_start_this_tile);
          int segment_end_this_tile = get_rank_id(m_end_this_tile);
          int stage_max = 0;
          for (int sid = segment_start_this_tile; sid <= segment_end_this_tile; sid++) {
            int stage = (sid - rank + tp_size) % tp_size;
            int ntokens_this_rank = sid == 0 ? cumsum_this_rank[0]
                                             : (cumsum_this_rank[sid] - cumsum_this_rank[sid - 1]);
            if (ntokens_this_rank != 0) {
              stage_max = std::max(stage, stage_max);
            }
          }
          return stage_max;
        };
        const int m_start = get_cumsum_this_rank_with_zero_pad(segment);
        const int m_end = get_cumsum_this_rank_with_zero_pad(segment + 1) - 1;
        int ntokens_this_segment = get_cumsum_this_rank_with_zero_pad(segment + 1) -
                                   get_cumsum_this_rank_with_zero_pad(segment);
        if (ntokens_this_segment == 0) {
          continue;
        }
        int tiled_m_start = m_start / tiled_m_size;
        int tiled_m_end = m_end / tiled_m_size;
        int start_stage = get_stage_for_tile(tiled_m_start);
        int end_stage = get_stage_for_tile(tiled_m_end);
        bool own_start = stage == start_stage;
        bool own_end = stage == end_stage;
        if (!own_start)
          tiled_m_start++;
        if (!own_end)
          tiled_m_end--;
        if (tiled_m_start > tiled_m_end) {
          continue;
        }
        ProblemSchedule sched;
        sched.expert_id = eid + ep_nexperts * gid;
        sched.m_start = tiled_m_start;
        sched.m_end = tiled_m_end;
        sched.source_rank_start = get_cumsum_this_rank_with_zero_pad(stage);
        sched.source_rank_end = get_cumsum_this_rank_with_zero_pad(stage + 1);
        problem_schedules.push_back(sched);
      }
    }
  }
  return problem_schedules;
}

template <int MaxNExperts>
__global__ void
sort_scatter_index_to_per_expert_kernel(
    int *sorted_scatter_index, int *splits_gpu, int ep_start, int ep_nexperts) {
  __shared__ int32_t splits_presum[MaxNExperts];
  splits_gpu += ep_start;
  block_prefix_sum_and_sync(splits_gpu, &splits_presum[0], ep_nexperts);

  int current_ep = blockIdx.x;
  int ep_row_offset = splits_presum[current_ep] - splits_gpu[current_ep];
  sorted_scatter_index += ep_row_offset;
  for (int i = threadIdx.x; i < splits_gpu[current_ep]; i += blockDim.x) {
    sorted_scatter_index[i] -= ep_row_offset;
  }
}

void
sort_scatter_index_to_per_expert(
    int *sorted_scatter_index,
    int *splits_gpu,
    int ep_start,
    int ep_nexperts,
    cudaStream_t stream) {
  constexpr int MaxNExperts = 1024;
  FLUX_CHECK_LE(ep_nexperts, MaxNExperts);
  sort_scatter_index_to_per_expert_kernel<MaxNExperts>
      <<<ep_nexperts, 1024, 0, stream>>>(sorted_scatter_index, splits_gpu, ep_start, ep_nexperts);
  CUDA_CHECK(cudaGetLastError());
}

std::vector<ProblemSchedule>
get_relax_sorted_problem_schedule_v2(
    std::vector<int32_t> const &splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups,
    const int nfold) {
  // start from `rank` segment, and leaves the tile cross bounder to last stage
  // what about tile that cross multi segments?
  //  for first stage: always leaves the tile cross rank to next stages. won't wait for more
  //  signals for other stage: if tile all in this stage, process it, otherwise leaves the tile to
  //  next stages
  std::vector<bytedance::flux::ProblemSchedule> problem_schedules;
  problem_schedules.reserve(tp_size * ep_nexperts);
  for (int iter = 0; iter < tp_size; iter += nfold) {
    for (int gid = 0; gid < num_weight_groups; gid++) {
      for (int eid = 0; eid < ep_nexperts; eid++) {
        for (int stage = iter; stage < min(tp_size, iter + nfold); stage++) {
          int segment = (stage + rank) % tp_size;
          const int *cumsum_this_rank = cumsum_per_rank_ptr + eid * tp_size;
          auto get_cumsum_this_rank_with_zero_pad = [=](int segment) {
            return segment == 0 ? 0 : cumsum_this_rank[segment - 1];
          };
          auto get_rank_id = [=](int m) {
            auto iter = std::upper_bound(cumsum_this_rank, cumsum_this_rank + tp_size, m);
            return std::distance(cumsum_this_rank, iter);
          };
          auto get_stage_for_tile = [=](int tiled_m) {
            int m_start_this_tile = tiled_m * tiled_m_size;
            int m_end_this_tile =
                std::min(splits[eid + ep_start], (tiled_m + 1) * tiled_m_size) - 1;
            int segment_start_this_tile = get_rank_id(m_start_this_tile);
            int segment_end_this_tile = get_rank_id(m_end_this_tile);
            int stage_max = 0;
            for (int sid = segment_start_this_tile; sid <= segment_end_this_tile; sid++) {
              int stage = (sid - rank + tp_size) % tp_size;
              int ntokens_this_rank = sid == 0
                                          ? cumsum_this_rank[0]
                                          : (cumsum_this_rank[sid] - cumsum_this_rank[sid - 1]);
              if (ntokens_this_rank != 0) {
                stage_max = std::max(stage, stage_max);
              }
            }
            return stage_max;
          };
          const int m_start = get_cumsum_this_rank_with_zero_pad(segment);
          const int m_end = get_cumsum_this_rank_with_zero_pad(segment + 1) - 1;
          int ntokens_this_segment = get_cumsum_this_rank_with_zero_pad(segment + 1) -
                                     get_cumsum_this_rank_with_zero_pad(segment);
          if (ntokens_this_segment == 0) {
            continue;
          }
          int tiled_m_start = m_start / tiled_m_size;
          int tiled_m_end = m_end / tiled_m_size;
          int start_stage = get_stage_for_tile(tiled_m_start);
          int end_stage = get_stage_for_tile(tiled_m_end);
          bool own_start = stage == start_stage;
          bool own_end = stage == end_stage;
          if (!own_start)
            tiled_m_start++;
          if (!own_end)
            tiled_m_end--;
          if (tiled_m_start > tiled_m_end) {
            continue;
          }
          ProblemSchedule sched;
          sched.expert_id = eid + ep_nexperts * gid;
          sched.m_start = tiled_m_start;
          sched.m_end = tiled_m_end;
          sched.source_rank_start = get_cumsum_this_rank_with_zero_pad(stage);
          sched.source_rank_end = get_cumsum_this_rank_with_zero_pad(stage + 1);
          problem_schedules.push_back(sched);
        }
      }
    }
  }
  return problem_schedules;
}

int
get_last_tile_for_segment_no_x(
    const std::vector<int> &accum_per_rank, int segment, int tile_size_m) {
  int segment_start_next = accum_per_rank[segment];
  int tile_m_next = segment_start_next / tile_size_m;
  return std::max(0, tile_m_next - 1);
}

int
get_start_tile_for_segment_no_x(
    const std::vector<int> &accum_per_rank, int segment, int tile_size_m) {
  int segment_start = segment == 0 ? 0 : accum_per_rank[segment - 1];
  if (segment_start == 0) {
    return 0;
  }
  int segment_end_prev = segment_start - 1;
  int tile_idx_m_end_prev = segment_end_prev / tile_size_m;
  int last_tile = (accum_per_rank.back() - 1) / tile_size_m;
  return std::min(tile_idx_m_end_prev + 1, last_tile);
}

int
get_last_tile_for_segment_x(const std::vector<int> &accum_per_rank, int segment, int tile_size_m) {
  int segment_end = accum_per_rank[segment] - 1;
  int tile_m_end = segment_end / tile_size_m;
  return tile_m_end;
}

struct ProblemScheduleWorker {
  std::vector<int> accum_per_rank;
  int problem_idx;
  int rank;
  int world_size;
  int stage;
  int tile_size_m;
  int tile_idx_m_start;
  int dispatched;
  int m_end;
  int tile_idx_m_last;

  ProblemScheduleWorker(
      const std::vector<int> &accum_per_rank, int problem_idx, int rank, int tile_size_m)
      : accum_per_rank(accum_per_rank),
        problem_idx(problem_idx),
        rank(rank),
        world_size(accum_per_rank.size()),
        stage(0),
        tile_size_m(tile_size_m),
        tile_idx_m_start(get_start_tile_for_segment_no_x(accum_per_rank, segment(), tile_size_m)),
        dispatched(0),
        m_end(accum_per_rank.back()) {
    tile_idx_m_last = get_last_tile_for_segment_x(
        this->accum_per_rank,
        (this->rank + this->world_size - 1) % this->world_size,
        this->tile_size_m);
  }

  int
  segment_id(int m) const {
    auto iter = std::upper_bound(this->accum_per_rank.begin(), this->accum_per_rank.end(), m);
    return std::distance(this->accum_per_rank.begin(), iter);
  }

  int
  segment() const {
    return (this->stage + this->rank) % this->world_size;
  }

  ProblemSchedule
  dispatch() {
    int tile_idx_m_end_ =
        this->stage == this->world_size ? tile_idx_m_last : this->tile_idx_m_end();
    ProblemSchedule segment_info{
        this->problem_idx,
        this->tile_idx_m_start,
        tile_idx_m_end_,
        segment_id(
            this->tile_idx_m_start * tile_size_m),  // TODO(houqi.1993) BUG here. to be fixed later
        segment_id(tile_idx_m_end_ * tile_size_m + tile_size_m - 1)};
    this->dispatched += tile_idx_m_end_ - this->tile_idx_m_start + 1;
    this->tile_idx_m_start = tile_idx_m_end_ + 1;
    if (this->tile_idx_m_start >= (this->m_end - 1) / this->tile_size_m) {
      this->tile_idx_m_start = 0;
    }
    this->stage++;
    return segment_info;
  }

  void
  advance_head() {
    this->stage++;
  }

  std::vector<ProblemSchedule>
  dispatch_all() {
    std::vector<ProblemSchedule> segments;
    segments.push_back(dispatch());
    return segments;
  }

  int
  tile_idx_m_end() const {
    return get_last_tile_for_segment_no_x(this->accum_per_rank, segment(), this->tile_size_m);
  }

  bool
  has_more_work() const {
    int m_tiles = (m_end - 1 + this->tile_size_m) / this->tile_size_m;
    return this->dispatched < m_tiles;
  }

  bool
  should_dispatch(int K) {
    return this->segment() == this->world_size - 1 ||
           (this->tile_idx_m_end() - this->tile_idx_m_start + 1) >= K;
  }
};

std::vector<ProblemSchedule>
get_sorted_problem_schedule_v2_with_ntiles_limit(
    std::vector<int32_t> const &splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups,
    const int ntiles_limit) {
  // start from `rank` segment, and leaves the tile cross bounder to last stage
  // what about tile that cross multi segments?
  //  for first stage: always leaves the tile cross rank to next stges. won't wait for more
  //  signals for other stage: if tile all in this stage, process it, otherwise leaves the tile to
  //  next stages
  std::vector<ProblemSchedule> problem_schedules;

  std::vector<ProblemScheduleWorker> problem_schedule_workers;
  int problem_count = ep_nexperts * num_weight_groups;
  for (int eid = 0; eid < problem_count; eid++) {
    const int *split_accum_this_rank = cumsum_per_rank_ptr + (eid % ep_nexperts) * tp_size;
    problem_schedule_workers.emplace_back(
        std::vector<int>(split_accum_this_rank, split_accum_this_rank + tp_size),
        eid,
        rank,
        tiled_m_size);
  }

  int stage = 0;
  while (stage != tp_size) {
    bool has_dispatched = false;
    for (auto &worker : problem_schedule_workers) {
      if (worker.has_more_work() && worker.should_dispatch(ntiles_limit)) {
        problem_schedules.push_back(worker.dispatch());
        has_dispatched = true;
      }
    }
    if (!has_dispatched) {
      stage++;
      for (auto &worker : problem_schedule_workers) {
        if (worker.stage < tp_size) {
          worker.advance_head();
        }
      }
    }
  }
  for (auto &worker : problem_schedule_workers) {
    while (worker.has_more_work()) {
      problem_schedules.push_back(worker.dispatch());
    }
  }
  // check if problem schedule is correct
  std::vector<int> num_tiles(problem_count, 0);
  for (const auto &problem_schedule : problem_schedules) {
    num_tiles[problem_schedule.expert_id] += problem_schedule.m_end - problem_schedule.m_start + 1;
  }
  for (int i = 0; i < problem_schedules.size(); i++) {
    int problem_idx = problem_schedules.at(i).expert_id;
    problem_idx = problem_idx % ep_nexperts;
    const int *accum_per_rank_ptr = cumsum_per_rank_ptr + problem_idx * tp_size;
    FLUX_CHECK_EQ(
        num_tiles[problem_idx],
        (accum_per_rank_ptr[tp_size - 1] + tiled_m_size - 1) / tiled_m_size)
        << " problem_idx " << i << " with " << problem_schedules.at(i);
  }
  return problem_schedules;
}
}  // namespace bytedance::flux
