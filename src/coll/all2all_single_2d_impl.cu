//===- all2all_single_2d_impl.cu ---------------------------------- C++ ---===//
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
#include <nvshmem.h>

#include "all2all_single_2d_impl.hpp"
#include "flux/flux.h"
#include "flux/cuda/cuda_common.h"
namespace bytedance {
namespace flux {

template <int32_t Alignment, typename VecType, int32_t BlockSize>
__device__ __forceinline__ size_t
copy_if_aligned(void *__restrict__ dest, void *__restrict__ src, size_t nbytes) {
  if (nbytes == 0)
    return 0;
  int32_t thread_id = threadIdx.x;
  size_t copyed_bytes = 0;
  if ((uintptr_t)dest % Alignment == 0 && (uintptr_t)src % Alignment == 0) {
    VecType *__restrict__ dest_vec = reinterpret_cast<VecType *>(dest);
    VecType *__restrict__ src_vec = reinterpret_cast<VecType *>(src);
    size_t nelems = nbytes / sizeof(VecType);
    for (size_t i = thread_id; i < nelems; i += BlockSize) {
      dest_vec[i] = src_vec[i];
    }
    copyed_bytes = nelems * sizeof(VecType);
  }
  return copyed_bytes;
}

template <int32_t BlockSize>
__device__ __forceinline__ void
copy_contiguous_data(void *__restrict__ dest, void *__restrict__ src, size_t nbytes) {
  size_t copyed_bytes;
  // 16B
  copyed_bytes = copy_if_aligned<16, uint4, BlockSize>(dest, src, nbytes);
  nbytes -= copyed_bytes;
  dest = reinterpret_cast<char *>(dest) + copyed_bytes;
  src = reinterpret_cast<char *>(src) + copyed_bytes;
  if (nbytes == 0)
    return;

  // 8B
  copyed_bytes = copy_if_aligned<8, uint64_t, BlockSize>(dest, src, nbytes);
  nbytes -= copyed_bytes;
  dest = reinterpret_cast<char *>(dest) + copyed_bytes;
  src = reinterpret_cast<char *>(src) + copyed_bytes;
  if (nbytes == 0)
    return;

  // 4B
  copyed_bytes = copy_if_aligned<4, uint32_t, BlockSize>(dest, src, nbytes);
  nbytes -= copyed_bytes;
  dest = reinterpret_cast<char *>(dest) + copyed_bytes;
  src = reinterpret_cast<char *>(src) + copyed_bytes;
  if (nbytes == 0)
    return;

  // 2B
  copyed_bytes = copy_if_aligned<2, uint16_t, BlockSize>(dest, src, nbytes);
  nbytes -= copyed_bytes;
  dest = reinterpret_cast<char *>(dest) + copyed_bytes;
  src = reinterpret_cast<char *>(src) + copyed_bytes;
  if (nbytes == 0)
    return;

  // 1B
  copyed_bytes = copy_if_aligned<1, uint8_t, BlockSize>(dest, src, nbytes);
}

template <typename Element, int32_t BlockSize>
__global__ void
__launch_bounds__(1024, 1) a2a_single_kernel(const All2AllSingleParams params) {
  int32_t cur_node_id = params.rank / params.local_world_size;
  int32_t bid = blockIdx.x;
  int32_t num_blocks = gridDim.x;
  size_t size_per_m = sizeof(Element) * params.n_dim;
  // inter node: push mode
  int32_t num_ranks_inter_node = params.world_size - params.local_world_size;
  for (int32_t i = bid; i < num_ranks_inter_node; i += num_blocks) {
    int32_t node_offset = i / params.local_world_size;
    int32_t rank_offset = i % params.local_world_size;
    int32_t remote_rank = ((cur_node_id + 1 + node_offset) * params.local_world_size +
                           (params.local_rank + 1 + rank_offset) % params.local_world_size) %
                          params.world_size;
    int64_t input_offset = 0;
    // preprocess cusum_input_splits?
    for (int32_t j = 0; j < remote_rank; j++) {
      input_offset += params.input_splits[j];
    }
    size_t msg_sz = size_per_m * params.input_splits[remote_rank];
    Element *dest_ptr = reinterpret_cast<Element *>(params.output_comm_ptr) +
                        params.max_split * params.n_dim * params.rank;
    Element *src_ptr =
        reinterpret_cast<Element *>(params.input_comm_ptr) + input_offset * params.n_dim;
    nvshmemx_putmem_signal_nbi_block(
        dest_ptr,
        src_ptr,
        msg_sz,
        params.barrier_ptr + params.rank,
        1,
        NVSHMEM_SIGNAL_SET,
        remote_rank);
  }

  int64_t intra_node_signal = 1;
  // intra node: push mode
  if (num_blocks <= params.local_world_size) {
    for (int32_t i = bid; i < params.local_world_size; i += num_blocks) {
      int32_t remote_rank = (i + params.local_rank + 1) % params.local_world_size +
                            cur_node_id * params.local_world_size;
      int64_t input_offset = 0;
      for (int32_t j = 0; j < remote_rank; j++) {
        input_offset += params.input_splits[j];
      }
      size_t msg_sz = size_per_m * params.input_splits[remote_rank];
      Element *dest_ptr = reinterpret_cast<Element *>(params.output_comm_ptr) +
                          params.max_split * params.n_dim * params.rank;
      Element *src_ptr =
          reinterpret_cast<Element *>(params.input_comm_ptr) + input_offset * params.n_dim;
      nvshmemx_putmem_signal_nbi_block(
          dest_ptr,
          src_ptr,
          msg_sz,
          params.barrier_ptr + params.rank,
          1,
          NVSHMEM_SIGNAL_SET,
          remote_rank);
    }
  } else {
    int32_t num_blocks_per_local_rank = num_blocks / params.local_world_size;
    intra_node_signal = num_blocks_per_local_rank;
    int32_t rank_offset = bid / num_blocks_per_local_rank;
    int32_t remote_rank = (rank_offset + 1 + params.local_rank) % params.local_world_size +
                          cur_node_id * params.local_world_size;
    int64_t input_offset = 0;
    for (int32_t j = 0; j < remote_rank; j++) {
      input_offset += params.input_splits[j];
    }
    int32_t split = params.input_splits[remote_rank];
    int32_t tile_size = (split + num_blocks_per_local_rank - 1) / num_blocks_per_local_rank;
    tile_size = (tile_size + 15) / 16 * 16;
    int32_t block_start = min(tile_size * (bid % num_blocks_per_local_rank), split);
    int32_t block_end = min(block_start + tile_size, split);
    input_offset = input_offset + block_start;
    size_t msg_sz = size_per_m * max(0, block_end - block_start);
    Element *dest_ptr = reinterpret_cast<Element *>(params.output_comm_ptr) +
                        params.max_split * params.n_dim * params.rank + block_start * params.n_dim;
    Element *src_ptr =
        reinterpret_cast<Element *>(params.input_comm_ptr) + input_offset * params.n_dim;
    nvshmemx_putmem_signal_nbi_block(
        dest_ptr,
        src_ptr,
        msg_sz,
        params.barrier_ptr + params.rank,
        1,
        NVSHMEM_SIGNAL_ADD,
        remote_rank);
  }

  // copy back to user buf
  // TODO(zhengxuegui.0): need further optimization, e.g. use copy engine or directly write to user
  // buf(intra node)
  for (int32_t i = bid; i < params.world_size; i += num_blocks) {
    int64_t wait_signal = (i / params.local_world_size) == cur_node_id ? intra_node_signal : 1;
    nvshmem_signal_wait_until(params.barrier_ptr + i, NVSHMEM_CMP_EQ, wait_signal);
    int64_t output_offset = 0;
    for (int32_t j = 0; j < i; ++j)
      output_offset += params.output_splits[j];
    Element *dest_ptr =
        reinterpret_cast<Element *>(params.output_ptr) + output_offset * params.n_dim;
    Element *src_ptr =
        reinterpret_cast<Element *>(params.output_comm_ptr) + params.max_split * params.n_dim * i;
    copy_contiguous_data<BlockSize>(dest_ptr, src_ptr, size_per_m * params.output_splits[i]);
  }
}

template <typename Element, int32_t BlockSize>
__global__ void
__launch_bounds__(1024, 1) a2a_single_kernel_v2(const All2AllSingleParams params) {
  int32_t bid = blockIdx.x;
  int32_t num_blocks = gridDim.x;
  size_t size_per_m = sizeof(Element) * params.n_dim;
  for (int tgt_rank = bid; tgt_rank < params.world_size; tgt_rank += num_blocks) {
    int64_t input_offset = 0;
    for (int32_t j = 0; j < tgt_rank; j++) {
      input_offset += params.input_splits[j];
    }
    size_t msg_sz = size_per_m * params.input_splits[tgt_rank];
    Element *dest_ptr = reinterpret_cast<Element *>(params.output_comm_ptr) +
                        params.max_split * params.n_dim * params.rank;
    Element *src_ptr =
        reinterpret_cast<Element *>(params.input_comm_ptr) + input_offset * params.n_dim;
    int global_tgt_rank =
        nvshmem_team_translate_pe(params.nvshmem_team, tgt_rank, NVSHMEM_TEAM_WORLD);
    nvshmemx_putmem_nbi_block(dest_ptr, src_ptr, msg_sz, global_tgt_rank);
  }
}

template <typename Element, int32_t BlockSize>
__global__ void
__launch_bounds__(1024, 1) perform_data_copy(const All2AllSingleParams params) {
  int tid = threadIdx.x;
  int32_t bid = blockIdx.x;
  int64_t output_offset = 0;
  // calculate the output M offset here
  for (int j = 0; j < bid; j++) {
    output_offset += params.output_splits[j];
  }
  Element *dst = reinterpret_cast<Element *>(params.output_ptr) + output_offset * params.n_dim;
  Element *src =
      reinterpret_cast<Element *>(params.output_comm_ptr) + params.max_split * params.n_dim * bid;
  int64_t n_elements = params.output_splits[bid] * params.n_dim;
  for (int pos = tid; pos < n_elements; pos += blockDim.x)
    dst[pos] = src[pos];
}

void
a2a_single_impl(
    const All2AllSingleParams params,
    DataTypeEnum input_dtype,
    int32_t num_comm_sm,
    cudaStream_t stream) {
  static constexpr int32_t kThreadsPerBlock = 1024;
  int32_t local_world_size = params.local_world_size;
  if (num_comm_sm > local_world_size) {
    num_comm_sm = num_comm_sm / local_world_size * local_world_size;
  }
  FLUX_CHECK(num_comm_sm % local_world_size == 0 || num_comm_sm < local_world_size);
  tuple_return_if(
      cute::make_tuple(_BF16{}, _FP32{}),
      [&](auto cdtype) { return cdtype == input_dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cuda_dtype(cdtype));
        a2a_single_kernel_v2<Element, kThreadsPerBlock>
            <<<num_comm_sm, kThreadsPerBlock, 0, stream>>>(params);
      },
      [&]() { FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype << "\n"; });
  nvshmemx_barrier_on_stream(params.nvshmem_team, stream);
  tuple_return_if(
      cute::make_tuple(_BF16{}, _FP32{}),
      [&](auto cdtype) { return cdtype == input_dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cuda_dtype(cdtype));
        perform_data_copy<Element, kThreadsPerBlock>
            <<<params.world_size, kThreadsPerBlock, 0, stream>>>(params);
      },
      [&]() { FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype << "\n"; });
  nvshmemx_barrier_on_stream(params.nvshmem_team, stream);
}
}  // namespace flux
}  // namespace bytedance
