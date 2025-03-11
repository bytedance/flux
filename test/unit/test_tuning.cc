//===- test_tuning.cc --------------------------------------------- C++ ---===//
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

#include <memory>
#include <sstream>
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/cuda/cuda_common.h"
#include "flux/op_registry.h"
#include "cutlass/util/device_memory.h"
#include "flux/args/comm_none.h"

namespace bytedance::flux {

void
tuning(int m, int n, int k) {
  auto arch = get_arch();

  auto meta = make_gemm_meta(
      _FP16{},
      arch,
      _CommNone{},
      _RCR{},
      ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}());
  using Element = decltype(to_cutlass_element(meta.dtype()));
  cutlass::DeviceAllocation<Element> block_A;
  cutlass::DeviceAllocation<Element> block_B;
  cutlass::DeviceAllocation<Element> block_C;
  cutlass::DeviceAllocation<Element> block_D;
  cutlass::DeviceAllocation<uint8_t> workspace;
  block_A.reset(m * k);
  block_B.reset(k * n);
  block_C.reset(m * n);
  block_D.reset(m * n);

  auto rt_conf = make_runtime_config(m, n, k);

  std::cout << "tuning for " << meta << std::endl;
  std::unique_ptr<UnifiedGemmHParams> best_hparams;
  float best_time = 0.0;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  OpRegistry::instance().visit_hparams(
      [&](UnifiedGemmHParams hparams) {
        auto gemm_op = OpRegistry::instance().get_op(meta, hparams);

        const GemmOnlyArguments args{
            m, n, k, 1.0, 0.0, block_A.get(), block_B.get(), block_C.get(), block_D.get()};
        auto ws_size = gemm_op->get_workspace_size(args);
        if (ws_size > workspace.size()) {
          workspace.reset(ws_size);
        }
        constexpr int warm_iters = 5;
        constexpr int iters = 10;
        float total_elapsed = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int iter = 0; iter < warm_iters + iters; ++iter) {
          GpuTimer timer;
          timer.start(stream);
          gemm_op->run(args, workspace.get(), stream);
          timer.stop();
          if (iter >= warm_iters) {
            total_elapsed += timer.elapsed_millis();
          }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
        std::cout << "time elapsed " << avg_elapsed << " ms for " << hparams << std::endl;
        if (best_hparams == nullptr) {
          best_hparams.reset(new UnifiedGemmHParams(hparams));
          best_time = avg_elapsed;
        } else if (avg_elapsed < best_time) {
          best_time = avg_elapsed;
          best_hparams.reset(new UnifiedGemmHParams(hparams));
        }
      },
      meta);

  std::cout << std::endl;
  std::cout << "Generated config code:\n\n";

  TuningConfigGenerator codegen("_config_gemm_v3_comm_none");
  codegen.add(meta, rt_conf, *best_hparams);
  std::cout << codegen.str() << std::endl;
}

}  // namespace bytedance::flux

int
main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << "<m> <n> <k>\n";
    return 1;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);
  bytedance::flux::tuning(m, n, k);
}
