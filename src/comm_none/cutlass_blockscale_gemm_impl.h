//===- cutlass_blockscale_gemm_impl.h ----------------------------- C++ ---===//
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

#include "flux/args/comm_none.h"

namespace bytedance {
namespace flux {

struct CutlassBlockScaleGemm {
  static void run(const BlockScaleGemmArguments &flux_args, void *workspace, cudaStream_t stream);

  static size_t get_workspace_size(const BlockScaleGemmArguments &flux_args);
};

}  // namespace flux
}  // namespace bytedance
