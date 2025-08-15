//===- add_kernel_test.cc ---------------------------------------- C++ ------===//
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
#include "flux_triton_kernel.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <vector>

#define CHECK_CUDA(x)                                                              \
  do {                                                                             \
    auto x_ = (x);                                                                 \
    if (x_ != cudaSuccess) {                                                       \
      fprintf(stderr, "CUDA runtime error %d at %s:%d\n", x_, __FILE__, __LINE__); \
      exit(1);                                                                     \
    }                                                                              \
  } while (0)

#define CHECK_CUDA_DRIVER(x)                                                      \
  do {                                                                            \
    auto x_ = (x);                                                                \
    if (x_ != CUDA_SUCCESS) {                                                     \
      fprintf(stderr, "CUDA driver error %d at %s:%d\n", x_, __FILE__, __LINE__); \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

void
test_add_kernel() {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  float *A, *B, *out;
  int N = 1024;
  CHECK_CUDA(cudaMalloc((void **)&A, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&B, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&out, N * sizeof(float)));

  std::vector<float> A_h(N);
  std::vector<float> B_h(N);
  std::vector<float> out_h(N);
  for (int i = 0; i < N; ++i) {
    A_h[i] = 1.0f * i;
    B_h[i] = 2.0f + i;
  }
  CHECK_CUDA(cudaMemcpy(A, A_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B, B_h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_DRIVER(vector_add_fp32_ex(
      stream,
      (CUdeviceptr)A,
      (CUdeviceptr)B,
      (CUdeviceptr)out,
      1,
      N,
      add_kernel__triton_algo_info_t{1024, 4, 3}));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUDA(cudaMemcpy(out_h.data(), out, N * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (out_h[i] != A_h[i] + B_h[i]) {
      fprintf(stderr, "out_h[%d] = %f, expected %f\n", i, out_h[i], A_h[i] + B_h[i]);
      exit(1);
    }
  }
  fprintf(stderr, "check passed\n");

  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(out));
}

int
main(int argc, char **argv) {
  CHECK_CUDA(cudaFree(0));

  printf("test on device 0\n");
  CHECK_CUDA(cudaSetDevice(0));
  test_add_kernel();
  printf("test on device 1\n");
  CHECK_CUDA(cudaSetDevice(1));
  test_add_kernel();

  return 0;
}
