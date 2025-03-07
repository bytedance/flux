//===- sm90_ag_scatter_fetcher.hpp ----------------------------- C++ ------===//
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
#include "flux/flux.h"
#include "flux/gemm_operator_base.h"

#include "cute/int_tuple.hpp"
#include "cute/numeric/int.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/device_kernel.h"
#include "flux/utils.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include "moe_ag_scatter/sort_util.h"
#include "nvshmem.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/memory_utils.hpp"

namespace bytedance::flux {

using namespace cute;

struct NoAGFetch {
  template <class... Ts>
  CUTE_HOST_DEVICE constexpr NoAGFetch(Ts...){};

  template <class... Ts>
  CUTE_HOST_DEVICE constexpr void operator()(Ts...) const {};
};

/// Check data fetched by all gather
struct Sm90AgScatterFetcher {
 public:
  using BarrierSync = cutlass::detail::
      NamedBarrierSync<cutlass::NumThreadsPerWarpGroup, (int)FluxNamedBarriers::AGScatterFetcher>;
  using Barrier = cutlass::detail::CustomizedGenericBarrier<BarrierSync>;

  struct Params {
    DistEnv dist_env;
    ProblemSchedule const *problem_schedule;
    int *barrier_ptr;
  };

  struct TensorStorage {};

  Params params;

  CUTLASS_HOST_DEVICE
  Sm90AgScatterFetcher() {}

  Sm90AgScatterFetcher(
      DistEnv dist_env, ProblemSchedule const *problem_schedule, int *barrier_ptr = nullptr)
      : params(Params{
            .dist_env = dist_env,
            .problem_schedule = problem_schedule,
            .barrier_ptr = barrier_ptr}) {}

  CUTLASS_DEVICE void
  operator()(int problem_idx, int thread_idx) const {
    Tensor problem_schedule = make_tensor(make_gmem_ptr(params.problem_schedule), make_shape(0));
    ProblemSchedule sched = problem_schedule(problem_idx);

    int start = sched.source_rank_start;
    int end = sched.source_rank_end;
    do {
      int rank = revert_order_to_rank(start, params.dist_env);
      Barrier::wait_eq(params.barrier_ptr, thread_idx, rank, 1);
      ++start;
    } while (start != end);
  }
};

}  // namespace bytedance::flux
