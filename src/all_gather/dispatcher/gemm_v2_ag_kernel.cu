//===- gemm_v2_ag_kernel.cu --------------------------------------- C++ ---===//
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

#include "all_gather/gemm_v2_ag_kernel.hpp"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"

namespace bytedance::flux {

static auto _registry_all_gather_gemm_op_dispatcher [[maybe_unused]] = []() {
  // TODO: add dispatcher logic here
  return 0;
}();

}  // namespace bytedance::flux
