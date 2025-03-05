//===- dispatch_policy.hpp ------------------------------------- C++ ------===//
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

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"

#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm {
using namespace cute;

struct KernelPtrArrayCpAsyncWarpSpecializedCooperative {};
struct KernelPtrArrayCpAsyncWarpSpecializedPingpong {};

template <
    int Stages_,
    class ClusterShape_ = Shape<_1, _1, _1>,
    class KernelSchedule = KernelPtrArrayCpAsyncWarpSpecializedCooperative>
struct MainloopSm90ArrayCpAsyncGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
  static_assert(
      cute::is_base_of_v<KernelPtrArrayCpAsyncWarpSpecializedCooperative, KernelSchedule> or
          cute::is_base_of_v<KernelPtrArrayCpAsyncWarpSpecializedPingpong, KernelSchedule>,
      "KernelSchedule must be one of the Ptr-Array or Grouped Gemm Cp async Warp Specialized "
      "Cooperative or Pingpong policies");
};

}  // namespace cutlass::gemm
