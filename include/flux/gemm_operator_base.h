//===- gemm_operator_base.h --------------------------------------- C++ ---===//
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

#pragma once
#include <any>
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include <stdexcept>
#include <type_traits>

namespace bytedance::flux {

struct GemmOperatorBase {
 public:
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmOperatorBase)

  virtual ~GemmOperatorBase() = default;
  virtual void run(
      std::any const &args,
      void *workspace = nullptr,
      void *stream = nullptr,
      bool launch_with_pdl = false) = 0;

  virtual void run(void *stream = nullptr, bool launch_with_pdl = false) = 0;

  // for device allocation required by Gemm::Arguments
  virtual size_t
  get_args_workspace_size(std::any const &args) const {
    return 0;
  }

  virtual void
  initialize_args_workspace(
      std::any const &args, void *args_workspace = nullptr, void *stream = nullptr) const {
    // noop
  }

  virtual void initialize(
      std::any const &args, void *workspace = nullptr, void *stream = nullptr) = 0;

  // total workspace: args_workspace + gemm workspace
  virtual size_t
  get_workspace_size(std::any const &args) const {
    return 0;
  }

  virtual size_t
  get_barrier_workspace_size(std::any const &args) const {
    return 0;
  }

  virtual UnifiedGemmHParams
  get_runtime_gemm_hparams() const {
    throw std::logic_error("get_runtime_gemm_hparams not implemented");
  }
};

}  // namespace bytedance::flux
