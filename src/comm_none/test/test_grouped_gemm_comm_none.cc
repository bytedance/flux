//===- test_grouped_gemm_comm_none.cc ----------------------------- C++ ---===//
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

#include <exception>
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/op_registry.h"
#include "flux/args/comm_none.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/profiler/device_allocation.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/packed_stride.hpp"

namespace bytedance::flux {
using namespace cute;

template <class Element>
bool
initialize_block(cutlass::DeviceAllocation<Element> &block, uint64_t seed = 2023) {
  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = static_cast<Element>(2);
    scope_min = static_cast<Element>(0);
  } else if (bits_input <= 8) {
    scope_max = static_cast<Element>(2);
    scope_min = static_cast<Element>(-2);
  } else {
    scope_max = static_cast<Element>(8);
    scope_min = static_cast<Element>(-8);
  }

  // cutlass::reference::device::BlockFillRandomUniform(
  //   block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

template <class DType>
void
run_grouped_gemm_comm_none(int groups, int m, int n, int k) {
  cudaSetDevice(0);
  auto arch = get_arch();

  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(DType{}));
  bool fast_accum = (is_fp8_dtype(dt_conf.a()) and is_fp8_dtype(dt_conf.b())) ? _True{} : _False{};
  auto v3_meta = make_gemm_v3_meta(fast_accum);
  auto meta = make_gemm_meta(dt_conf, arch, _CommNone{}, _RCC{}, _GemmGroupedV3{}(), v3_meta);

  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementC = decltype(to_cutlass_element(dt_conf.c()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));
  using UnderlyingProblemShape = cute::Shape<int, int, int>;

  using ElementAccumulator = float;

  auto rt_conf = make_runtime_config(n, m, k);

  // prepare the input/weight tensor

  // Host-side allocations
  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<ElementAccumulator> alpha_host;
  std::vector<ElementAccumulator> beta_host;

  // Device-side allocations
  std::vector<UnderlyingProblemShape> problem_sizes_host;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementD> block_D;
  cutlass::DeviceAllocation<ElementD> block_ref_D;

  // random the problem sizes
  for (int i = groups; i > 0; i--) {
    problem_sizes_host.push_back({n, m, k});
  }

  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;

  for (int32_t i = 0; i < groups; ++i) {
    auto problem = problem_sizes_host.at(i);
    auto [N, M, K] = problem;

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);

  uint64_t seed = 2024;
  std::vector<void const *> ptr_A_host(groups);
  std::vector<void const *> ptr_B_host(groups);
  std::vector<void const *> ptr_C_host(groups);
  std::vector<void *> ptr_D_host(groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(groups);
  std::vector<ElementAccumulator *> ptr_beta_host(groups);

  for (int32_t i = 0; i < groups; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    // ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_C_host.at(i) = nullptr;

    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    alpha_host.push_back(1.0);
    beta_host.push_back(0.0);
  }

  // TODO support random initialization for the gpu buffer
  // initialize_block(block_A, seed + 2023);
  // initialize_block(block_B, seed + 2022);
  // initialize_block(block_C, seed + 2021);

  auto hparams = OpRegistry::instance().get_hparams(meta, rt_conf);
  auto gemm_op = OpRegistry::instance().get_op(meta, hparams);

  auto stream = nullptr;
  auto args = GemmGroupedV3Arguments{
      groups,
      1.0,
      0.0,
      problem_sizes_host.data(),
      // swap A,B
      ptr_B_host.data(),
      ptr_A_host.data(),
      ptr_C_host.data(),
      ptr_D_host.data()};

  int64_t workspace_size = gemm_op->get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  gemm_op->run(args, workspace.get(), stream);

  constexpr int warm_iters = 5;
  constexpr int iters = 10;
  GpuTimer timer;
  for (int i = 0; i < warm_iters + iters; ++i) {
    if (i == warm_iters) {
      timer.start(stream);
    }
    gemm_op->run(args, workspace.get(), stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  printf("op time elapsed: %.3f ms\n", timer.elapsed_millis() / iters);
}
}  // namespace bytedance::flux

int
main(int argc, char *argv[]) {
  // CUDA 12.3 is required for grouped gemm v3
#if defined(__CUDACC__)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 3)) {
    std::cerr << "This test requires CUDA 12.3 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#endif

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr << "This example requires a GPU of NVIDIA's Hopper Architecture or "
              << "later (compute capability 90 or greater).\n";
    return 0;
  }

  if ((argc != 5) and (argc != 6)) {
    std::cerr << "Usage: " << argv[0] << " <groups> <m> <n> <k> <dtype>\n";
    return 1;
  }
  int groups = std::atoi(argv[1]);
  int m = std::atoi(argv[2]);
  int n = std::atoi(argv[3]);
  int k = std::atoi(argv[4]);

  std::string dtype = "FP16";
  if (argc == 6) {
    dtype = argv[5];
  }

  try {
    using namespace bytedance::flux;
    if (dtype == "FP8" or dtype == "fp8") {
      bytedance::flux::run_grouped_gemm_comm_none<decltype(make_gemm_dtype_config(
          _E4M3{}, _E4M3{}, _BF16{}, _BF16{}))>(groups, m, n, k);
    } else if (dtype == "FP16" or dtype == "fp16") {
      bytedance::flux::run_grouped_gemm_comm_none<_FP16>(groups, m, n, k);
    } else if (dtype == "BF16" or dtype == "bf16") {
      bytedance::flux::run_grouped_gemm_comm_none<_BF16>(groups, m, n, k);
    } else {
      FLUX_CHECK(false) << "unsupported dtype: " << dtype;
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
  }
}
