//===- test_gemm_rs.cc -------------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
#include <exception>
#include <thread>
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/args/reduce_scatter.h"
#include "cutlass/util/device_memory.h"
#include "flux/runtime_config.h"

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
  cudaSetDevice(0);
}

void
run_rs(int m, int n, int k, int tp, int nnodes) {
  constexpr int kMaxTp = 8;
  FLUX_CHECK(tp <= kMaxTp) << "k=" << k << " kMaxTp=" << kMaxTp;
  FLUX_CHECK(k % tp == 0) << "k=" << k << " tp=" << tp;
  FLUX_CHECK(tp % nnodes == 0) << "tp=" << tp << " nnodes=" << nnodes;

  k = k / tp;
  auto arch = get_arch();
  auto meta = make_gemm_meta(
      _FP16{},
      arch,
      _ReduceScatter{},
      _RCR{},
      ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}(),
      None{},
      make_reduce_scatter_meta(true, _IntraNode{}));
  using Element = decltype(to_cutlass_element(meta.dtype()));
  auto rt_conf = make_runtime_config(m, n, k, make_reduce_scatter_runtime_config(tp, nnodes));

  cutlass::DeviceAllocation<Element> block_A[kMaxTp];
  cutlass::DeviceAllocation<Element> block_B[kMaxTp];
  cutlass::DeviceAllocation<Element> block_C[kMaxTp];
  cutlass::DeviceAllocation<Element> block_D[kMaxTp];
  cutlass::DeviceAllocation<Element> block_reduce_buffer[kMaxTp];
  cutlass::DeviceAllocation<int> block_barrier[kMaxTp];
  void *block_D_ptrs[kMaxTp];
  void *block_reduce_buffer_ptrs[kMaxTp];
  void *block_barrier_ptrs[kMaxTp];

  for (int i = 0; i < tp; ++i) {
    cudaSetDevice(i);
    block_A[i].reset(m * k);
    block_B[i].reset(k * n);
    block_C[i].reset(m * n);
    block_D[i].reset(m * n);
    block_reduce_buffer[i].reset(m / tp * nnodes * nnodes * n);
    block_D_ptrs[i] = block_D[i].get();
    block_reduce_buffer_ptrs[i] = block_reduce_buffer[i].get();
  }

  std::atomic<int> ready_count;
  ready_count = 0;

  auto thread_fn = [&](int rank) {
    cudaSetDevice(rank);
    auto gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    const GemmReduceScatterArguments args{
        m,
        n,
        k,
        rank,
        tp,
        nnodes,
        1.0,
        0.0,
        block_A[rank].get(),
        block_B[rank].get(),
        block_C[rank].get(),
        block_D_ptrs,
        block_reduce_buffer[rank].get(),
        block_barrier_ptrs};

    int flag_count = gemm_op->get_barrier_workspace_size(args) / sizeof(int);
    block_barrier[rank].reset(flag_count);
    block_barrier_ptrs[rank] = block_barrier[rank].get();
    cudaDeviceSynchronize();

    ++ready_count;
    while (ready_count < tp) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    GpuTimer timer;
    constexpr int warm_iters = 5;
    constexpr int iters = 10;
    for (int i = 0; i < warm_iters + iters; ++i) {
      if (i == warm_iters) {
        timer.start(stream);
      }
      gemm_op->run(args, nullptr, stream);
    }
    timer.stop();
    printf("#%d time elapsed: %.3f ms\n", rank, timer.elapsed_millis() / iters);
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  std::vector<std::thread> ths;
  for (int i = 0; i < tp; ++i) {
    ths.emplace_back(thread_fn, i);
  }
  for (int i = 0; i < tp; ++i) {
    ths[i].join();
  }
}

}  // namespace bytedance::flux

int
main(int argc, char *argv[]) {
  if (argc != 5 and argc != 6) {
    std::cerr << "Usage: " << argv[0] << "<m> <n> <k> <tp> <nnodes=1>\n";
    return 1;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  int tp = std::atoi(argv[4]);

  int nnodes = 1;
  if (argc == 6) {
    nnodes = std::atoi(argv[5]);
  }
  bytedance::flux::init_peer_access(tp);
  try {
    bytedance::flux::run_rs(m, n, k, tp, nnodes);
  } catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
  }
}
