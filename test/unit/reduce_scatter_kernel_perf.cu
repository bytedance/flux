//===- reduce_scatter_kernel_perf.cu ---------------------------- C++ ---===//
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

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <fstream>
#include <map>
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
#include "cuda_utils.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"
#include "cute/util/debug.hpp"
#include "cute/util/print.hpp"
#include "cutlass/coord.h"
#include "cutlass/detail/helper_macros.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/flux.h"
#include "flux/utils.h"
#include "gemm_rs/reduce_scatter_barrier_struct.hpp"
#include "utils.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/device_kernel.h"
#include "gemm_rs/reduce_scatter_kernel.hpp"

namespace bf = bytedance::flux;

constexpr int kTileM = 128, kTileN = 64;
using VecType = uint4;

DEFINE_int32(m, 1024, "");
DEFINE_int32(n, 12288, "");
DEFINE_int32(num_blocks, 8, "");
DEFINE_int32(num_threads, 1024, "");
DEFINE_int32(warmup_iters, 5, "");
DEFINE_int32(iters, 10, "");
DEFINE_int32(ngpus, 8, "");
DEFINE_int32(sub_world_size, 4, "");
DEFINE_int32(sleep_ns, 0, "");
DEFINE_bool(run_local, true, "");
DEFINE_bool(run_remote, true, "");
DEFINE_bool(run_wait, true, "");
DEFINE_bool(run_copy, true, "");
DEFINE_bool(flatten, true, "");
DEFINE_int32(rank_mask, 0xff, "");
DEFINE_bool(push, false, "");
DEFINE_bool(verify, true, "");
DEFINE_bool(1d_ring, false, "");
DEFINE_bool(use_gemmk, true, "");
DEFINE_bool(use_barrier_queue, true, "");
DEFINE_bool(use_cudaMemcpyAsync, false, "");
DEFINE_bool(per_tile_flags, true, "");
DEFINE_int32(n_split, 1, "");
DEFINE_bool(use_cpu_buffer, false, "");

template <typename T>
constexpr static bool kIsFp16 = std::is_same_v<T, half> || std::is_same_v<T, cutlass::half_t>;
template <typename T>
constexpr static bool kIsBf16 =
    std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, cutlass::bfloat16_t>;

template <typename T>
using ToVecType =
    std::conditional_t<kIsFp16<T>, __half2, std::conditional_t<kIsBf16<T>, __nv_bfloat162, void>>;

struct Stat {
  float avg;
  float err;
  float min_value;
  float max_value;
};

Stat
average(const std::vector<float> &values) {
  float min_value = std::numeric_limits<float>::max(),
        max_value = std::numeric_limits<float>::min(), total = 0, total_err = 0;
  for (auto v : values) {
    total += v;
    min_value = std::min(v, min_value);
    max_value = std::max(v, max_value);
  }
  float avg = total / values.size();
  for (auto v : values) {
    total_err += (v - avg) * (v - avg);
  }
  return {avg, sqrtf(total_err / values.size()), min_value, max_value};
}

struct KernelConfig {
  int gemmk : 8;
  int push : 8;
  int use_1d_ring : 8;
  int flatten_tile : 8;
};
bool
operator<(KernelConfig const &lhs, KernelConfig const &rhs) {
  union {
    KernelConfig config;
    int value;
  } lhs_u = {lhs}, rhs_u = {rhs};
  return lhs_u.value < rhs_u.value;
}

CUTLASS_GLOBAL void
reset_signals_per_tiles(int *ptr, int tiled_mn) {
  bytedance::flux::PerTileFlagsWrapper flags(ptr, tiled_mn);
  for (int i = threadIdx.x; i < tiled_mn; i += blockDim.x) {
    *flags.epilogue_ptr(i) = 1;
    *flags.reduce_ptr(i) = 0;
    *flags.reduce_sub_node_ptr(i) = 0;
  }
}

CUTLASS_GLOBAL void
reset_signals_per_segment(int *ptr, int nsegments) {
  bytedance::flux::PerRankFlagsWrapper flags(ptr, nsegments);
  for (int i = threadIdx.x; i < nsegments; i += blockDim.x) {
    *flags.gemm_done_ptr(i) = 1;
    *flags.buffer_ready_ptr(i) = 0;
    *flags.copy_done_ptr(i) = 0;
    *flags.counter_ptr(i) = 0;
    *flags.remote_copy_done_ptr(i) = 0;
  }
}

template <typename T>
CUTLASS_GLOBAL void
memset_kernel(T *ptr, int len, T value) {
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    ptr[i] = value;
  }
}

template <typename T = half, typename VecType = uint4>
class ReduceScatterOp {
 public:
  ReduceScatterOp(size_t m, size_t n, int ngpus)
      : elems_(m * n),
        m_(m),
        n_(n),
        ngpus_(ngpus),
        gemm_outputs(ngpus),
        reduce_buffers(ngpus),
        flags(ngpus) {
    int tiled_m = (m / ngpus_ + kTileM - 1) / kTileM * ngpus_;
    int tiled_n = (n + kTileN - 1) / kTileN;
    tiled_mn_ = tiled_m * tiled_n;
    int nsignals = tiled_m * tiled_n * (sizeof(bf::PerTileFlags) / sizeof(int));

    auto gemm_size = cutlass::make_Coord((int)m, (int)n);
    streams_.reserve(ngpus);
    timers_.reserve(ngpus);
    for (int i = 0; i < ngpus_; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      streams_.emplace_back(i);
      timers_.emplace_back(i);

      auto &gemm_output = gemm_outputs[i];
      gemm_output.resize(gemm_size);  // allocate here
      gemm_output.sync_device();
      T value = i >= 4 ? (1 << (i - 2)) : 1 << i;
      memset_kernel<<<8, 1024, 0, streams_.back()>>>(gemm_output.device_data(), m * n, value);
      CUDA_CHECK(cudaGetLastError());

      auto &reduce_buffer = reduce_buffers[i];
      reduce_buffer.resize(gemm_size);
      reduce_buffer.sync_device();
      CUDA_CHECK(
          cudaMemsetAsync(reduce_buffer.device_data(), 0, m * n * sizeof(T), streams_.back()));

      if (FLAGS_use_cpu_buffer) {
        pinhost_vectors.emplace_back(gemm_size.product());
      }
      if (FLAGS_1d_ring && FLAGS_use_cpu_buffer &&
          (((i % FLAGS_sub_world_size == 0 && !FLAGS_push)) ||
           ((i + 1) % FLAGS_sub_world_size == 0 && FLAGS_push))) {
        void *dptr = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&dptr, pinhost_vectors.back().ptr(), 0));
        reduce_buffer_dptrs.emplace_back((T *)dptr);
      } else {
        reduce_buffer_dptrs.emplace_back(reduce_buffer.device_data());
      }

      auto &flag_tensor = flags[i];
      flag_tensor.resize(cutlass::make_Coord(nsignals, 1));
      flag_tensor.sync_device();
    }
  }

  void
  reset_signals_all_async() {
    for (int i = 0; i < ngpus_; i++) {
      CUDA_CHECK(cudaSetDevice(i));
      auto &stream = streams_[i];
      if (FLAGS_per_tile_flags)
        reset_signals_per_tiles<<<1, 1024, 0, stream>>>((int *)flags[i].device_data(), tiled_mn_);
      else
        reset_signals_per_segment<<<1, 1024, 0, stream>>>((int *)flags[i].device_data(), ngpus_);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  template <bool kGemmk, bool kPushMode, bool kUse1DRing, bool kFlattenTile>
  void
  run_async(int devid) {
    using gemmk_kernel = std::conditional_t<
        kUse1DRing,
        std::conditional_t<
            kPushMode,
            bf::ReduceScatterRing1dPushGemmk<T, kTileM, kTileN, kFlattenTile>,
            bf::ReduceScatterRing1dPullGemmk<T, kTileM, kTileN, kFlattenTile>>,
        std::conditional_t<
            kPushMode,
            bf::ReduceScatterRing2dPushGemmk<T, kTileM, kTileN, kFlattenTile>,
            bf::ReduceScatterRing2dPullGemmk<T, kTileM, kTileN, kFlattenTile>>>;
    if constexpr (!kGemmk) {
      static_assert(kUse1DRing == false);
      static_assert(kPushMode == false);
    }
    using no_gemmk_kernel = bf::ReduceScatterRing2dPull<T, kTileM, kTileN, kFlattenTile>;
    using kernel = std::conditional_t<kGemmk, gemmk_kernel, no_gemmk_kernel>;
    static std::pair<int, int> attr =
        bf::ReduceScatterOp<T, kTileM, kTileN, kFlattenTile>::get_func_attr(
            (void *)cutlass::Kernel2<kernel>);
    auto [num_threads, shmsize] = attr;
    if (devid == 0)
      FLUX_LOG_FIRST_N(INFO, 1) << "num_threads:" << num_threads
                                << " with shared memory size: " << shmsize << "\n";
    bf::ReduceScatterParams params{
        .rank = devid,
        .world_size = ngpus_,
        .nnodes = 1,
        .m = (int)m_,
        .n = (int)n_,
        .num_blocks = FLAGS_num_blocks,
        .sleep_ns = FLAGS_sleep_ns,
#ifdef FLUX_DEBUG_RS
        .run_local = FLAGS_run_local,
        .run_remote = FLAGS_run_remote,
        .do_copy = FLAGS_run_copy,
        .do_wait = FLAGS_run_wait,
#endif
        .use_barrier_queue = FLAGS_use_barrier_queue,
        .use_gemmk = FLAGS_use_gemmk,
        .per_tile_flags = FLAGS_per_tile_flags,
        .use_cudaMemcpyAsync = FLAGS_use_cudaMemcpyAsync,
        .n_split = FLAGS_n_split,
        .sub_world_size = 4,
        .opaque = nullptr,
        .use_1d_ring = false,
        .use_p2p_read = false,
        .args_workspace = nullptr,
    };

    for (int i = 0; i < ngpus_; i++) {
      params.scatter_ptr_aux[i] = gemm_outputs[i].device_data();
      params.reduce_ptr[i] = reduce_buffer_dptrs[i];
      params.barrier_ptr[i] = flags[i].device_data();
    }

    static bf::LaunchProp prop_intra_numa =
        bf::GetLaunchProp((void *)bf::run_per_segment_kernel<T>);
    static bf::LaunchProp prop_intra_node =
        bf::GetLaunchProp((void *)bf::run_per_segment_kernel_tp8<T>);

    CUDA_CHECK(cudaSetDevice(devid));
    auto &stream = streams_[devid];
    if (FLAGS_per_tile_flags) {
      cutlass::Kernel2<kernel><<<params.num_blocks, num_threads, 0, stream>>>(params);
    } else {
      if (FLAGS_use_cudaMemcpyAsync) {
        FLUX_LOG_FIRST_N(INFO, 1) << "use_cudaMemcpyAsync";
        bf::run_per_segment_with_cudaMemcpyAsync<T>(params, stream);
      } else {
        bf::LaunchProp &prop = kUse1DRing ? prop_intra_numa : prop_intra_node;
        void *args[1] = {(void *)&params};
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            kUse1DRing ? (void *)bf::run_per_segment_kernel<T>
                       : (void *)bf::run_per_segment_kernel_tp8<T>,
            dim3(std::min(prop.max_num_blocks, params.num_blocks)),
            dim3(prop.max_threads),
            &args[0],
            0,
            stream));
      }
    }
    CUDA_CHECK(cudaGetLastError());
    // stream.sync();
    if (devid == 0)
      std::cerr << "reduce_scatter done\n";
  }

  void
  run_ngpus_async() {
    static std::map<KernelConfig, std::function<void(int)>> run_async_funcs = {
        {KernelConfig{true, true, true, true},
         [this](int devid) { run_async<true, true, true, true>(devid); }},
        {KernelConfig{true, true, true, false},
         [this](int devid) { run_async<true, true, true, false>(devid); }},
        {KernelConfig{true, true, false, true},
         [this](int devid) { run_async<true, true, false, true>(devid); }},
        {KernelConfig{true, true, false, false},
         [this](int devid) { run_async<true, true, false, false>(devid); }},

        {KernelConfig{true, false, true, true},
         [this](int devid) { run_async<true, false, true, true>(devid); }},
        {KernelConfig{true, false, true, false},
         [this](int devid) { run_async<true, false, true, false>(devid); }},
        {KernelConfig{true, false, false, true},
         [this](int devid) { run_async<true, false, false, true>(devid); }},
        {KernelConfig{true, false, false, false},
         [this](int devid) { run_async<true, false, false, false>(devid); }},
        // not supported now for non-gemmk with push
        // {KernelConfig{false, true, true, true},
        //  [this](int devid) { run_async<false, true, true, true>(devid); }},
        // {KernelConfig{false, true, true, false},
        //  [this](int devid) { run_async<false, true, true, false>(devid); }},
        // {KernelConfig{false, true, false, true},
        //  [this](int devid) { run_async<false, true, false, true>(devid); }},
        // {KernelConfig{false, true, false, false},
        //  [this](int devid) { run_async<false, true, false, false>(devid); }},

        // not supported now for non-gemmk with use_1d_ring
        // {KernelConfig{false, false, true, true},
        //  [this](int devid) { run_async<false, false, true, true>(devid); }},
        // {KernelConfig{false, false, true, false},
        //  [this](int devid) { run_async<false, false, true, false>(devid); }},
        {KernelConfig{false, false, false, true},
         [this](int devid) { run_async<false, false, false, true>(devid); }},
        {KernelConfig{false, false, false, false},
         [this](int devid) { run_async<false, false, false, false>(devid); }},
    };
    KernelConfig config{FLAGS_use_gemmk, FLAGS_push, FLAGS_1d_ring, FLAGS_flatten};
    FLUX_CHECK(run_async_funcs.find(config) != run_async_funcs.end());
    for (int i = 0; i < ngpus_; i++) {
      if ((1 << i) & FLAGS_rank_mask) {
        run_async_funcs[config](i);
      }
    }
  }

  void
  sync_all() {
    for (int i = 0; i < ngpus_; i++) {
      streams_[i].sync();
    }
  }

  int
  get_target_value(int i) {
    return i >= 4 ? (1 << (i - 2)) : 1 << i;
  }

  int
  get_reduce_target_value() {
    int sum = 0;
    for (int i = 0; i < ngpus_; i++) {
      sum += get_target_value(i);
    }
    return sum;
  }

  void
  verify() {
    bool ok = true;
    for (int i = 0; i < ngpus_; i++) {
      auto &buffer = gemm_outputs[i];
      buffer.sync_host();
      int m_per_rank = m_ / ngpus_;
      int count = 0;
      for (int m = m_per_rank * i; m < m_per_rank * (i + 1); m++) {
        for (int n = 0; n < n_; n++) {
          float value = buffer.at({m, n});
          int target = get_reduce_target_value();
          if (value != target) {
            fprintf(stderr, "GPU %d [%d, %d] %f != %d\n", i, m, n, value, target);
            if (count++ >= 5) {
              std::cerr << "too much errors. skip checking\n";
              goto fail;
            }
          }
        }
      }
      if (count) {
      fail:
        ok = false;
      }
    }
    if (ok)
      return;
    fprintf(stderr, "dump gemm outputs and reduce outputs\n");
    // dumps all or dumps none
    for (int i = 0; i < ngpus_; i++) {
      std::fstream fout_gemm(
          "output_" + std::to_string(i) + ".data", std::ios::out | std::ios::binary);
      fout_gemm.write((const char *)gemm_outputs[i].host_data(), m_ * n_ * sizeof(T));

      reduce_buffers[i].sync_host();
      std::fstream fout_reduce(
          "reduce_" + std::to_string(i) + ".data", std::ios::out | std::ios::binary);
      fout_reduce.write((const char *)reduce_buffers[i].host_data(), m_ * n_ * sizeof(T));
    }
  }

  void
  perf() {
    auto run_iters = [&](int iters) -> std::vector<float> {
      std::vector<float> durations(this->ngpus_, 0);
      for (int i = 0; i < iters; i++) {
        reset_signals_all_async();
        sync_all();
        for (int i = 0; i < ngpus_; i++)
          timers_[i].Start(streams_[i]);
        run_ngpus_async();
        for (int i = 0; i < ngpus_; i++)
          timers_[i].Stop();
        sync_all();
        for (int i = 0; i < ngpus_; i++)
          durations[i] += timers_[i].GetEclapsedTime();
      }
      return durations;
    };

    if (FLAGS_verify) {
      run_iters(1);
      sync_all();
      verify();
      std::cerr << "verify done\n";
    }

    run_iters(FLAGS_warmup_iters);
    printf("warmup done\n");
    auto duration_ms = run_iters(FLAGS_iters);
    auto [duration_avg, duration_stderr, duration_min, duration_max] = average(duration_ms);

    double bytes_mb =
        FLAGS_m * FLAGS_n * sizeof(T) / 1e6 * FLAGS_iters * (FLAGS_ngpus - 1) / FLAGS_ngpus;
    printf(
        "%0.2f/%0.2lf ms/iter for %d iters, BW:avg/min/max %0.2lf %0.2lf %0.2lfGB/s\n",
        duration_avg / FLAGS_iters,
        duration_stderr / FLAGS_iters,
        FLAGS_iters,
        bytes_mb / duration_avg,
        bytes_mb / duration_max,
        bytes_mb / duration_min);
  }

 private:
  size_t m_, n_, elems_;
  int tiled_mn_;
  int ngpus_;
  std::vector<cutlass::HostTensor<T, cutlass::layout::RowMajor>> gemm_outputs, reduce_buffers;
  std::vector<PinHostVector<T>> pinhost_vectors;
  std::vector<T *> reduce_buffer_dptrs;
  std::vector<cutlass::HostTensor<int, cutlass::layout::RowMajor>> flags;
  std::vector<CudaStream> streams_;
  std::vector<CudaEventTimer> timers_;
};

int
main(int argc, char **argv) {
  init_flags(&argc, &argv, true);

  for (int i = 0; i < FLAGS_ngpus; i++) {
    CUDA_CHECK(cudaSetDevice(i));
    for (int j = 0; j < FLAGS_ngpus; j++) {
      if (i == j)
        continue;
      CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
    }
  }

  ReduceScatterOp<half, uint4> op(FLAGS_m, FLAGS_n, FLAGS_ngpus);
  op.perf();

  return 0;
}
