//===- test_memory_bound.cc --------------------------------------- C++ ---===//
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

#include <atomic>
#include <chrono>
#include <sstream>
#include <thread>
#include <cublas_v2.h>
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "cutlass/util/device_memory.h"
#include "flux/args/comm_none.h"

namespace bytedance::flux {

void
init_peer_access(int tp) {
  for (int i = 0; i < tp; ++i) {
    cudaSetDevice(i);
    for (int j = 0; j < tp; ++j) {
      if (j != i)
        cudaDeviceEnablePeerAccess(j, 0);
    }
  }
}

void
test_memory_bound(int tp, int m, int n, int k, int iters, int gemm_type, int copy_type) {
  constexpr int kMaxTp = 8;
  if (tp > kMaxTp) {
    std::cerr << "tp > kMaxTp\n";
    exit(1);
  }

  constexpr int warmup_iters = 5;
  printf("iters=%d, warmup_iters=%d.\n", iters, warmup_iters);
  if (gemm_type == 0) {
    printf("GEMM: none.\n");
  } else {
    printf("GEMM: tp=%d m=%d n=%d k=%d.\n", tp, m, n, k);
  }

  if (copy_type == 0) {
    printf("COPY: none.\n");
  } else if (copy_type == 1) {
    printf("COPY: each rank copy to self rank.\n");
  } else if (copy_type == 2) {
    printf("COPY: each rank scatter to all ranks.\n");
  } else if (copy_type == 3) {
    printf("COPY: each rank gather from all ranks.\n");
  }

  auto arch = get_arch();
  auto sm_core = get_sm_core();
  auto meta = make_gemm_meta(
      _FP16{},
      arch,
      sm_core,
      _CommNone{},
      _RCR{},
      ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}());
  using Element = decltype(to_cutlass_element(meta.dtype()));
  cutlass::DeviceAllocation<Element> block_A[kMaxTp];
  cutlass::DeviceAllocation<Element> block_B[kMaxTp];
  cutlass::DeviceAllocation<Element> block_C[kMaxTp];
  cutlass::DeviceAllocation<Element> block_D[kMaxTp];
  cutlass::DeviceAllocation<Element> block_scatter[kMaxTp];

  for (int i = 0; i < tp; ++i) {
    cudaSetDevice(i);
    block_A[i].reset(m * k);
    block_B[i].reset(k * n);
    block_C[i].reset(m * n);
    block_D[i].reset(m * n);
    block_scatter[i].reset(m * n);
  }

  std::atomic<bool> start_run;
  start_run = false;

  std::vector<std::string> messages(tp);

  auto thread_fn = [&](int rank) {
    cudaSetDevice(rank);
    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    cudaStream_t copy_stream;
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    const GemmOnlyArguments args{
        m,
        n,
        k,
        1.0,
        0.0,
        block_A[rank].get(),
        block_B[rank].get(),
        block_C[rank].get(),
        block_D[rank].get()};
    while (!start_run) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    auto rt_conf = make_runtime_config(m, n, k);
    // begin running
    auto gemm_op = OpRegistry::instance().get_op(meta, rt_conf);

    GpuTimer timer_compute;
    GpuTimer timer_copy;
    for (int i = 0; i < iters + warmup_iters; ++i) {
      if (i == warmup_iters) {
        timer_compute.start(compute_stream);
        timer_copy.start(copy_stream);
      }
      if (gemm_type == 1) {
        // launch compute kernel
        gemm_op->run(args, nullptr, compute_stream);
      }
      if (copy_type != 0) {
        int m_per_rank = (m + tp - 1) / tp;
        // launch copy kernel
        int chunk_size = m_per_rank * n;

        for (int j = rank + 1; j < tp + rank; ++j) {
          int id = j % tp;
          auto dst = block_scatter[copy_type == 1 ? rank : id].get() + chunk_size * rank;
          auto src = block_scatter[rank].get() + chunk_size * id;
          if (copy_type == 3) {
            std::swap(src, dst);
          }
          cudaMemcpyAsync(dst, src, chunk_size * sizeof(Element), cudaMemcpyDefault, copy_stream);
        }
      }
    }
    timer_compute.stop();
    timer_copy.stop();
    float compute_elapsed = std::round(timer_compute.elapsed_millis() / iters * 1000) / 1000;
    float copy_elapsed = std::round(timer_copy.elapsed_millis() / iters * 1000) / 1000;
    std::stringstream ss;
    ss << "rank #" << rank << ": compute: " << compute_elapsed << "ms, copy: " << copy_elapsed
       << "ms.";
    messages[rank] = ss.str();
  };

  std::vector<std::thread> ths;

  for (int i = 0; i < tp; ++i) {
    ths.emplace_back(thread_fn, i);
  }

  start_run = true;

  for (int i = 0; i < tp; ++i) {
    ths[i].join();
    std::cout << messages[i] << std::endl;
  }
}

}  // namespace bytedance::flux

int
main(int argc, char *argv[]) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0] << "<tp> <m> <n> <k> <iters> <gemm_type> <copy_type>\n"
              << "gemm_type: 0 for no gemm, 1 for run gemm\n"
              << "copy_type: 0 for no copy, 1 for copy to local, 2 for scatter to all tp, 3 for "
                 "gather from all tp\n";
    return 1;
  }

  int tp = std::atoi(argv[1]);
  int m = std::atoi(argv[2]);
  int n = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);
  int iters = std::atoi(argv[5]);
  int gemm_type = std::atoi(argv[6]);
  int copy_type = std::atoi(argv[7]);

  if (m % tp != 0) {
    std::cerr << "m % tp != 0\n";
    return 1;
  }

  using namespace bytedance::flux;
  init_peer_access(tp);
  test_memory_bound(tp, m, n, k, iters, gemm_type, copy_type);
}
