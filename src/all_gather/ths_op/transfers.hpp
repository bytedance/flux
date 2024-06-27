#pragma once

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>
#include <vector>
#include <cstdint>

#define CUStreamWaitValue(...) cuda_stub().cuStreamWaitValue32_v2(__VA_ARGS__)
#define CUStreamWriteValue(...) cuda_stub().cuStreamWriteValue32_v2(__VA_ARGS__)

#define SPLIT 1

namespace bytedance {
namespace flux {
namespace ths_op {

template <int dst_idx_, int src_idx_>
class Transfer {
 public:
  static constexpr int src_idx = src_idx_;
  static constexpr int dst_idx = dst_idx_;

 protected:
  template <typename FlagType>
  static void
  set_ready(
      int rank_,
      int segment,
      int split_index,
      std::vector<FlagType *> &barrier_ptrs,
      cudaStream_t stream) {
    CU_CHECK(CUStreamWriteValue(
        stream,
        (CUdeviceptr)((int32_t *)barrier_ptrs[rank_] + (segment * SPLIT + split_index)),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT));
  }

  template <typename FlagType>
  static void
  wait_ready(
      int rank_,
      int segment,
      int split_index,
      std::vector<FlagType *> &barrier_ptrs,
      cudaStream_t stream) {
    CU_CHECK(CUStreamWaitValue(
        stream,
        (CUdeviceptr)((int32_t *)barrier_ptrs[rank_] + (segment * SPLIT + split_index)),
        1,
        CU_STREAM_WRITE_VALUE_DEFAULT));
  }
};

template <int dst_idx_, int src_idx_>
class TransferCudaMemcpy : public Transfer<dst_idx_, src_idx_> {
 private:
  using ParentType = Transfer<dst_idx_, src_idx_>;

 public:
  static constexpr int src_idx = src_idx_;
  static constexpr int dst_idx = dst_idx_;

  template <typename FlagType>
  static void
  run(std::vector<void *> &input_ptrs,
      std::vector<FlagType *> &barrier_ptrs,
      int src_rank,
      int dst_rank,
      int chunk_size,
      int split_chunk_size,
      at::cuda::CUDAStream stream) {
    for (int j = 0; j < SPLIT; ++j) {
      auto split_offset = j * split_chunk_size;
      if (src_rank == src_idx) {
        ParentType::wait_ready(src_rank, src_idx, j, barrier_ptrs, stream);
      }
      CUDA_CHECK(cudaMemcpyAsync(
          ptr_offset(input_ptrs[dst_rank], src_idx * chunk_size + split_offset),
          ptr_offset(input_ptrs[src_rank], src_idx * chunk_size + split_offset),
          split_chunk_size,
          cudaMemcpyDeviceToDevice,
          stream));
      ParentType::set_ready(dst_rank, src_idx, j, barrier_ptrs, stream);
    }
  }
};

template <int dst_idx_, int src_idx_>
class TransferCudaMemcpyPush : public TransferCudaMemcpy<dst_idx_, src_idx_> {
 public:
  static constexpr int exec_rank = src_idx_;
};

template <int dst_idx_, int src_idx_>
class TransferCudaMemcpyPull : public TransferCudaMemcpy<dst_idx_, src_idx_> {
 public:
  static constexpr int exec_rank = dst_idx_;
};

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
