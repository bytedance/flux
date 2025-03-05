//===- test_moe_ag_scatter.cc ---------------------------------- C++ ------===//
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

#include <algorithm>
#include <exception>
#include <numeric>
#include <random>
#include "cute/int_tuple.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_operator_base.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/op_registry.h"
#include "flux/args/moe_ag_scatter.h"
#include "cutlass/util/device_memory.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <mpi.h>
#include "moe_ag_scatter/sort_util.h"

namespace bytedance::flux {

template <class DType>
struct MoeMlp1Runner {
 protected:
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(DType{}));
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementD = decltype(to_cutlass_element(dt_conf.d()));

 private:
  const int b;
  const int s;
  const int h;
  const int ffn_size;
  const int nexperts;
  const int topk;

  const int tp_rank;
  const int tp_size;

  const int ntokens;
  const int ntokens_perrank;
  const int nexperts_perrank;

  std::vector<int32_t> splits;
  std::vector<int32_t> scatter_index;
  std::vector<int32_t> gather_index;
  std::vector<int32_t> sorted_splits;

  cutlass::DeviceAllocation<int32_t> dev_splits;
  cutlass::DeviceAllocation<int32_t> dev_scatter_index;
  cutlass::DeviceAllocation<int32_t> dev_gather_index;

  // inputs&outputs dummy tensors
  cutlass::DeviceAllocation<ElementA> inputs;
  cutlass::DeviceAllocation<ElementB> weights;
  cutlass::DeviceAllocation<ElementD> outputs;

  cutlass::DeviceAllocation<int32_t> dev_sorted_splits;
  cutlass::DeviceAllocation<int32_t> dev_sorted_gather_index;
  cutlass::DeviceAllocation<int32_t> dev_sorted_scatter_index;

  cutlass::DeviceAllocation<uint8_t> barrier;
  cutlass::DeviceAllocation<uint8_t> workspace;

  ElementA *input_buffer;

 private:
  void
  init_splits_scatter_index() {
    splits.resize(nexperts);
    scatter_index.resize(ntokens * topk);

    std::vector<int> experts(nexperts);
    std::iota(experts.begin(), experts.end(), 0);
    std::mt19937 generator(/*seed=*/2024);
    std::vector<int> choosen_expert(ntokens * topk);

    for (int i = 0; i < ntokens; ++i) {
      std::shuffle(experts.begin(), experts.end(), generator);
      for (int j = 0; j < topk; ++j) {
        int idx = i * topk + j;
        choosen_expert[idx] = experts[j];
        ++splits[choosen_expert[idx]];
      }
    }

    auto expert_offset = splits;
    int32_t presum = 0;
    for (int i = 0; i < nexperts; ++i) {
      expert_offset[i] = presum;
      presum += splits[i];
    }

    for (int i = 0; i < ntokens; ++i) {
      for (int j = 0; j < topk; ++j) {
        int idx = i * topk + j;
        scatter_index[idx] = expert_offset[choosen_expert[idx]]++;
      }
    }

    dev_splits.reset(splits.size());
    dev_splits.copy_from_host(splits.data());
    dev_scatter_index.reset(scatter_index.size());
    dev_scatter_index.copy_from_host(scatter_index.data());
  }

  void
  init_gather_index() {
    gather_index.resize(ntokens * topk);
    for (int i = 0; i < ntokens; ++i) {
      for (int j = 0; j < topk; ++j) {
        int src_idx = scatter_index[i * topk + j];
        gather_index[src_idx] = i;
      }
    }
    dev_gather_index.reset(gather_index.size());
    dev_gather_index.copy_from_host(gather_index.data());
  }

  void
  init_gemm_buffers() {
    inputs.reset(1LL * ntokens * topk * h);
    weights.reset(1LL * ffn_size * h * nexperts);
    outputs.reset(1LL * ntokens * topk * ffn_size);

    dev_sorted_splits.reset(nexperts * tp_size);
    dev_sorted_gather_index.reset(ntokens * topk);
    dev_sorted_scatter_index.reset(ntokens * topk);
  }

  void
  init_nvshmem_ptrs() {
    int64_t input_buffer_size = 1LL * ntokens * h * sizeof(ElementA);
    input_buffer = static_cast<ElementA *>(nvshmem_malloc(input_buffer_size));
  }

 public:
  MoeMlp1Runner(int b, int s, int h, int ffn_size, int nexperts, int topk)
      : b(b),
        s(s),
        h(h),
        ffn_size(ffn_size),
        nexperts(nexperts),
        topk(topk),
        tp_rank(nvshmem_my_pe()),
        tp_size(nvshmem_n_pes()),
        ntokens(b * s),
        ntokens_perrank(ntokens / tp_size),
        nexperts_perrank(nexperts / tp_size) {
    FLUX_CHECK(ntokens % tp_size == 0);
    FLUX_CHECK(nexperts >= topk);
    init_splits_scatter_index();
    init_gather_index();
    init_gemm_buffers();
    init_nvshmem_ptrs();
  }

  void
  operator()(cudaStream_t stream = nullptr) {
    auto meta = make_gemm_meta(dt_conf, get_arch(), _AGScatter{}, _RCR{}, _GemmGroupedV3{});
    auto rt_conf = make_runtime_config();
    OpRegistry::OpPtr op = OpRegistry::instance().get_op(meta, rt_conf);

    ag_scatter_sort_impl(
        AGScatterSortOpArguments{
            DistEnv(tp_rank, tp_size, 1),
            ntokens,
            nexperts,
            dev_splits.get(),
            dev_gather_index.get(),
            dev_sorted_splits.get(),
            dev_sorted_scatter_index.get(),
            dev_sorted_gather_index.get(),
        },
        stream);

    sorted_splits.resize(nexperts * tp_size);
    cudaMemcpyAsync(
        sorted_splits.data(),
        dev_sorted_splits.get(),
        sizeof(int32_t) * nexperts * tp_size,
        cudaMemcpyDeviceToHost,
        stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    using UnderlyingProblemShape = cute::Shape<int, int, int>;
    std::vector<UnderlyingProblemShape> problem_sizes;
    std::vector<void const *> ptr_A;
    std::vector<void const *> ptr_B;
    std::vector<void const *> ptr_C;
    std::vector<void *> ptr_D;
    std::vector<int32_t const *> ptr_gather_A;
    std::vector<int32_t const *> ptr_scatter_D;
    std::vector<ProblemSchedule> problem_schedules_arg;

    {
      // initialize args vectors
      int tile_M = cute::get<0>(op->get_runtime_gemm_hparams().tile_shape());
      auto problem_schedules = get_sorted_problem_schedule(
          sorted_splits, DistEnv(tp_rank, tp_size, 1), nexperts, tile_M);

      int const N = ffn_size / tp_size;
      int const K = h;
      int64_t const weight_bytes = 1LL * N * K * sizeof_dtype(dt_conf.a());

      for (std::size_t idx = 0; idx < problem_schedules.size(); ++idx) {
        auto const &problem_param = problem_schedules[idx];
        int Mi = problem_param.m_end - problem_param.m_start;
        if (Mi == 0) {
          continue;
        }
        problem_sizes.emplace_back(Mi, N, K);
        ptr_A.emplace_back(input_buffer);
        ptr_B.emplace_back(ptr_offset(
            weights.get(), 1LL * problem_param.expert_id * N * K * sizeof_dtype(dt_conf.b())));
        ptr_C.emplace_back(nullptr);
        ptr_D.emplace_back(reinterpret_cast<uint8_t *>(outputs.get()));
        ptr_gather_A.emplace_back(static_cast<int32_t const *>(ptr_offset(
            dev_sorted_gather_index.get(), 1LL * sizeof(int32_t) * problem_param.m_start)));
        ptr_scatter_D.emplace_back(static_cast<int32_t const *>(ptr_offset(
            dev_sorted_scatter_index.get(), 1LL * sizeof(int32_t) * problem_param.m_start)));
        problem_schedules_arg.emplace_back(problem_param);
      }
    }

    auto args = GemmGroupedAgScatterArguments{
        GemmGroupedV3Arguments{
            .problem_count = static_cast<int>(problem_sizes.size()),
            .alpha = 1.0f,
            .beta = 0.0f,
            .problem_sizes = problem_sizes.data(),
            .ptr_A = ptr_A.data(),
            .ptr_B = ptr_B.data(),
            .ptr_C = ptr_C.data(),
            .ptr_D = ptr_D.data()},
        DistEnv(tp_rank, tp_size, 1),
        ntokens,
        h,
        input_buffer,
        ptr_gather_A.data(),
        ptr_scatter_D.data(),
        problem_schedules_arg.data()};

    int64_t workspace_size = op->get_workspace_size(args);
    if (workspace_size > workspace.size()) {
      workspace.reset(workspace_size);
    }
    int64_t barrier_size = op->get_barrier_workspace_size(args);
    if (barrier_size > barrier.size()) {
      barrier.reset(barrier_size);
    }
    args.barrier_ptr = barrier.get();
    CUDA_CHECK(cudaMemsetAsync(barrier.get(), 0, barrier.size(), stream));
    // prepare data of current rank
    CUDA_CHECK(cudaMemcpyAsync(
        input_buffer + tp_rank * ntokens_perrank * h,
        inputs.get(),
        ntokens_perrank * h * sizeof(ElementA),
        cudaMemcpyDeviceToDevice,
        stream));
    nvshmemx_barrier_all_on_stream(stream);
    for (int i = 0; i < tp_size; ++i) {
      CU_CHECK(CUStreamWriteValue(
          stream,
          (CUdeviceptr)(ptr_offset(barrier.get(), i * sizeof(int))),
          1,
          CU_STREAM_WRITE_VALUE_DEFAULT));
    }
    GpuTimer timer;
    timer.start(stream);
    op->run(args, workspace.get(), stream);
    timer.stop();
    std::cout << "gemm:" << timer.elapsed_millis() << std::endl;
  }
};

template <class DType>
void
run_moe_ag_scatter(int b, int s, int h, int ffn_size, int nexperts, int topk) {
  MoeMlp1Runner<DType> runner(b, s, h, ffn_size, nexperts, topk);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  constexpr int warm_iters = 5;
  constexpr int iters = 10;
  GpuTimer timer;
  for (int i = 0; i < warm_iters + iters; ++i) {
    if (i == warm_iters) {
      timer.start(stream);
    }
    runner(stream);
  }
  timer.stop();
  printf("op time elapsed: %.3f ms\n", timer.elapsed_millis() / iters);
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace bytedance::flux

struct FinalizeRTTI {
  ~FinalizeRTTI() {
    nvshmem_finalize();
    MPI_Finalize();
  }
};

int
main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  cudaSetDevice(rank);

  auto mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  FinalizeRTTI fin_rtti [[maybe_unused]];

  if (argc != 7 and argc != 8) {
    std::cerr << "Usage: " << argv[0] << "<b> <s> <h> <ffn_size> <nexperts> <topk> <dtype=FP16>\n";
    return 1;
  }
  int b = std::atoi(argv[1]);
  int s = std::atoi(argv[2]);
  int h = std::atoi(argv[3]);
  int ffn_size = std::atoi(argv[4]);
  int nexperts = std::atoi(argv[5]);
  int topk = std::atoi(argv[6]);
  std::string dtype = "FP16";
  if (argc == 8) {
    dtype = argv[7];
  }

  try {
    using namespace bytedance::flux;
    if (dtype == "FP8" or dtype == "fp8") {
      bytedance::flux::run_moe_ag_scatter<decltype(make_gemm_dtype_config(
          _E4M3{}, _E4M3{}, _BF16{}, _BF16{}))>(b, s, h, ffn_size, nexperts, topk);
    } else if (dtype == "FP16" or dtype == "fp16") {
      bytedance::flux::run_moe_ag_scatter<decltype(make_gemm_dtype_config(
          _FP16{}, _FP16{}, _FP16{}, _FP16{}))>(b, s, h, ffn_size, nexperts, topk);
    } else if (dtype == "BF16" or dtype == "bf16") {
      bytedance::flux::run_moe_ag_scatter<decltype(make_gemm_dtype_config(
          _BF16{}, _BF16{}, _BF16{}, _BF16{}))>(b, s, h, ffn_size, nexperts, topk);
    } else {
      FLUX_CHECK(false) << "unsupported dtype: " << dtype;
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
  }
}
