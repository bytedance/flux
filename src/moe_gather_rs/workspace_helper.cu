
//===- workspace_helper.cu ------------------------------------- C++ ------===//
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
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>

#include "flux/cuda/cuda_common.h"
#include "flux/cuda/reduce_utils.cuh"
#include "workspace_helper.h"
namespace bytedance::flux {

__device__ __forceinline__ void *
ptr_with_offset(void *ptr, intptr_t offset) {
  return (void *)((char *)ptr + offset);
}

template <typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD>
__global__ void
make_workspace_kernel(
    const MoeGatherRSWorkspaceArgs args,
    int input_elem_size,
    int output_elem_size,
    void *workspace) {
  extern __shared__ char shared_storage[];
  int *ep_splits = args.splits_gpu + args.ep_start;
  int *ep_splits_acc = (int *)shared_storage;
  block_prefix_sum_and_sync(ep_splits, ep_splits_acc, args.ep_nexperts);
  int problem_per_split = args.num_groups * args.ep_nexperts;
  int problem_count = problem_per_split * args.N_split;
  int N = args.N, K = args.K;
  const int new_N = args.N / args.N_split;
  constexpr int kAlignment = 128;

  // the offsets
  int offset_problem_sizes = 0;
  int offset_ptr_A =
      pad_to(offset_problem_sizes + problem_count * sizeof(cutlass::gemm::GemmCoord), kAlignment);
  int offset_ptr_B = pad_to(offset_ptr_A + problem_count * sizeof(void *), kAlignment);
  int offset_ptr_C = pad_to(offset_ptr_B + problem_count * sizeof(void *), kAlignment);
  int offset_ptr_D = pad_to(offset_ptr_C + problem_count * sizeof(void *), kAlignment);
  int offset_lda = pad_to(offset_ptr_D + problem_count * sizeof(void *), kAlignment);
  int offset_ldb = pad_to(offset_lda + problem_count * sizeof(int64_t), kAlignment);
  int offset_ldc = pad_to(offset_ldb + problem_count * sizeof(int64_t), kAlignment);
  int offset_ldd = pad_to(offset_ldc + problem_count * sizeof(int64_t), kAlignment);
  int offset_ldr = pad_to(offset_ldd + problem_count * sizeof(int64_t), kAlignment);
  int offset_scale_A = pad_to(offset_ldr + problem_count * sizeof(int64_t), kAlignment);
  int offset_scale_B = pad_to(offset_scale_A + problem_count * sizeof(float *), kAlignment);
  int offset_non_empty_problem_count =
      pad_to(offset_scale_B + problem_count * sizeof(float *), kAlignment);
  // the ptrs
  cutlass::gemm::GemmCoord *problem_sizes =
      (cutlass::gemm::GemmCoord *)((char *)workspace + offset_problem_sizes);
  void **ptr_A = (void **)((char *)workspace + offset_ptr_A);
  void **ptr_B = (void **)((char *)workspace + offset_ptr_B);
  void **ptr_C = (void **)((char *)workspace + offset_ptr_C);
  void **ptr_D = (void **)((char *)workspace + offset_ptr_D);
  int64_t *lda = (int64_t *)((char *)workspace + offset_lda);
  int64_t *ldb = (int64_t *)((char *)workspace + offset_ldb);
  int64_t *ldc = (int64_t *)((char *)workspace + offset_ldc);
  int64_t *ldd = (int64_t *)((char *)workspace + offset_ldd);
  int64_t *ldr = (int64_t *)((char *)workspace + offset_ldr);
  float **scale_A = (float **)((char *)workspace + offset_scale_A);
  float **scale_B = (float **)((char *)workspace + offset_scale_B);
  int *non_empty_problem_count = (int *)((char *)workspace + offset_non_empty_problem_count);

  for (int i = threadIdx.x; i < problem_count; i += blockDim.x) {
    int sid = i / problem_per_split;
    int sr = i % problem_per_split;
    int gid = sr / args.ep_nexperts;
    int eid = sr % args.ep_nexperts;

    int Mi = ep_splits[eid];
    int M_acc = ep_splits_acc[eid] - Mi;

    problem_sizes[i] = cutlass::gemm::GemmCoord{Mi, new_N, K};
    ptr_A[i] = ptr_with_offset(args.input[gid], M_acc * K * input_elem_size);
    ptr_B[i] = ptr_with_offset(args.weights[gid], (eid * N + sid * new_N) * K * input_elem_size);
    ptr_C[i] = nullptr;
    ptr_D[i] = ptr_with_offset(args.output[gid], (M_acc * N + sid * new_N) * output_elem_size);
    lda[i] = LayoutA::packed({(int)Mi, (int)K}).stride(0);
    ldb[i] = LayoutB::packed({(int)K, (int)N}).stride(0);
    ldc[i] = LayoutC::packed({(int)Mi, (int)N}).stride(0);
    ldd[i] = LayoutD::packed({(int)Mi, (int)N}).stride(0);
    ldr[i] = 0;

    scale_A[i] = args.input_scales[gid] == nullptr ? nullptr : args.input_scales[gid];
    scale_B[i] = args.weight_scales[gid] == nullptr ? nullptr : args.weight_scales[gid] + eid;
  }
  __syncthreads();
  int *non_empty_splits = (int *)shared_storage;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *non_empty_splits = 0;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < args.ep_nexperts; i += blockDim.x) {
    atomicAdd(non_empty_splits, ep_splits[i] != 0);
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *non_empty_problem_count = *non_empty_splits * args.N_split * args.num_groups;
  }
}
void
make_workspace(
    const MoeGatherRSWorkspaceArgs &args,
    GemmLayoutEnum layout,
    int input_elem_size,
    int output_elem_size,
    void *workspace,
    cudaStream_t stream) {
  FLUX_CHECK_EQ(layout, GemmLayoutEnum::RCR);
  int shared_mem_size = std::max(
      sizeof(int),                      // non_empty_problem_count
      sizeof(int) * args.ep_nexperts);  // splits_acc
  make_workspace_kernel<
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor>
      <<<1, 768, shared_mem_size, stream>>>(args, input_elem_size, output_elem_size, workspace);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace bytedance::flux
