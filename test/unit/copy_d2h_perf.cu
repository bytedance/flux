//===- copy_d2h_perf.cu ------------------------------------------- C++ ---===//
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

// test GPU -> host memory w/o cross numa or interleaved
#include <numa.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <memory>
#include "cuda_utils.hpp"
#include "numa_helper.hpp"
#include "utils.hpp"
#include "copy_perf_utils.hpp"
#include "flux/cuda/nvml_stub.h"
using bytedance::flux::nvml_stub;

DEFINE_int32(warmup_iters, 5, "");
DEFINE_int32(iters, 10, "");
DEFINE_int64(elems, 12288 * 4096, "");
DEFINE_int32(num_blocks, 8, "");
DEFINE_int32(num_threads, 1024, "");
DEFINE_bool(copy_h2d, true, "true for copy from host to device");
DEFINE_bool(copy_d2h, true, "true for copy from device to host");
DEFINE_bool(verify, true, "");
DEFINE_bool(copy_by_grid, true, "");
DEFINE_int32(mem_numa_node, -1, "-1 for interleaved");
DEFINE_int32(run_numa_node, -1, "-1 for no bind cpu. 0 for bind to cpu0");

DEFINE_string(copy_h2d_gpus, "0", "split by ,");
DEFINE_string(copy_d2h_gpus, "0", "split by ,");
DEFINE_bool(numa_affinity, false, "");

int
get_gpu_node_id(int gpu_device_id) {
  if (FLAGS_numa_affinity == false)
    return -1;
  bytedance::flux::ensure_nvml_init();
  nvmlDevice_t device;
  auto status = nvml_stub().nvmlDeviceGetHandleByIndex(gpu_device_id, &device);
  unsigned long nodeset;
  NVML_CHECK(nvmlDeviceGetMemoryAffinity(device, 1, &nodeset, NVML_AFFINITY_SCOPE_NODE));
  return __builtin_popcount(nodeset);
}

template <typename T>
class Bench {
 public:
  using VecType = uint4;

  Bench(int elems, int gpu_device_id, bool copy_d2h, bool use_cudaMemcpy)
      : gpu_device_id_(gpu_device_id),
        elems_(elems),
        copy_d2h_(copy_d2h),
        use_cudaMemcpy_(use_cudaMemcpy),
#ifdef USE_NUMA
        host_tensor_(elems, get_gpu_node_id(gpu_device_id), true),
#else
        host_tensor_(elems),
#endif
        device_tensor_(gpu_device_id, elems, 0),
        stream_(gpu_device_id),
        timer_(gpu_device_id) {
    reset();
  }

  void
  reset() {
    if (copy_d2h_) {
      ScopedDevice _(gpu_device_id_);
      CUDA_CHECK(cudaMemset(device_tensor_.ptr(), 0x03, elems_ * sizeof(T)));
    } else {
      std::memset(host_tensor_.ptr(), 0x03, elems_ * sizeof(T));
    }
  }

  void
  copy_async() {
    void *src_ptr = copy_d2h_ ? device_tensor_.ptr() : host_tensor_.ptr();
    void *dst_ptr = copy_d2h_ ? host_tensor_.ptr() : device_tensor_.ptr();
    ScopedDevice _(gpu_device_id_);
    if (use_cudaMemcpy_) {
      CUDA_CHECK(
          cudaMemcpyAsync(dst_ptr, src_ptr, elems_ * sizeof(T), cudaMemcpyDefault, stream_));
    } else {
      run_copy_kernel<T, VecType><<<FLAGS_num_blocks, FLAGS_num_threads, 0, stream_>>>(
          (T *)dst_ptr, (T *)src_ptr, elems_);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  void
  start_record() {
    timer_.Start(stream_);
  }
  void
  stop_record() {
    timer_.Stop();
  }
  void
  sync() {
    stream_.sync();
  }
  float
  get_duration() {
    return timer_.GetEclapsedTime();
  }

  void
  verify_or_die() {
    // ddst is all 01
    std::vector<T> tmp = device_tensor_.cpu();
    T *src = copy_d2h_ ? tmp.data() : host_tensor_.ptr();
    T *dst = copy_d2h_ ? host_tensor_.ptr() : tmp.data();
    for (int i = 0; i < elems_; i++) {
      FLUX_CHECK_EQ(float(src[i]), float(dst[i])) << "not aligned at " << i;
    }
  }

 private:
  int gpu_device_id_;
  int elems_;
  bool copy_d2h_;
  bool use_cudaMemcpy_;
#ifdef USE_NUMA
  NumaVector<T> host_tensor_;
#else
  PinHostVector<T> host_tensor_;
#endif
  DeviceVector<T> device_tensor_;
  CudaStream stream_;
  CudaEventTimer timer_;
};

std::vector<int>
to_list(const std::string &str) {
  auto values = split(str, ",");
  std::vector<int> ret;
  ret.reserve(values.size());
  for (auto &v : values) {
    ret.push_back(std::stoi(v));
  }
  return ret;
}

std::pair<float, float>
average(const std::vector<float> &values) {
  float total = 0, total_err = 0;
  for (auto v : values) {
    total += v;
  }
  float avg = total / values.size();
  for (auto v : values) {
    total_err += (v - avg) * (v - avg);
  }
  return {avg, sqrt(total_err / values.size())};
}

template <typename T>
std::vector<T>
concate(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  std::vector<T> ret(lhs.size() + rhs.size());
  std::copy(lhs.begin(), lhs.end(), ret.begin());
  std::copy(rhs.begin(), rhs.end(), ret.begin() + lhs.size());
  return ret;
}

template <typename T>
void
run_copyd2h(size_t elems, bool use_cudaMemcpy) {
  std::vector<int> gpus_d2h = to_list(FLAGS_copy_d2h_gpus);
  std::vector<int> gpus_h2d = to_list(FLAGS_copy_h2d_gpus);
  std::vector<std::shared_ptr<Bench<T>>> bench_d2h;
  for (auto gpu_id : gpus_d2h) {
    bench_d2h.emplace_back(std::make_shared<Bench<T>>(elems, gpu_id, true, use_cudaMemcpy));
  }
  std::vector<std::shared_ptr<Bench<T>>> bench_h2d;
  for (auto gpu_id : gpus_h2d) {
    bench_h2d.emplace_back(std::make_shared<Bench<T>>(elems, gpu_id, false, use_cudaMemcpy));
  }

  auto run_bench = [&](const std::string &prefix,
                       std::vector<std::shared_ptr<Bench<T>>> &benches) {
    // perf d2h
    if (FLAGS_verify) {
      for (auto &bench : benches) {
        bench->reset();
      }
      for (auto &bench : benches) {
        bench->copy_async();
      }
      for (auto &bench : benches) {
        bench->sync();
      }
      for (auto &bench : benches) {
        bench->verify_or_die();
      }
      fprintf(stderr, "verify done\n");
    }

    for (int i = 0; i < FLAGS_warmup_iters; i++) {
      for (auto &bench : benches) {
        bench->copy_async();
      }
      for (auto &bench : benches) {
        bench->sync();
      }
    }
    fprintf(stderr, "warmup done\n");

    float duration_total_ms = 0;
    std::vector<std::vector<float>> duration_ms_list(benches.size());
    for (int i = 0; i < FLAGS_iters; i++) {
      for (auto &bench : benches) {
        bench->start_record();
        bench->copy_async();
        bench->stop_record();
      }
      int j = 0;
      for (auto &bench : benches) {
        bench->sync();
        float duration_ms = bench->get_duration();
        duration_total_ms += duration_ms;
        duration_ms_list[j].push_back(duration_ms);
        j++;
      }
    }

    float avg_ms = duration_total_ms / benches.size() / FLAGS_iters;
    fprintf(
        stderr,
        "%s Time eclapsed: %0.2fms, bandwidth: %0.2lfGB/s\n",
        prefix.c_str(),
        avg_ms,
        double(sizeof(T) * elems) / 1e6 / avg_ms);
    for (int i = 0; i < benches.size(); i++) {
      auto [avg_ms, stdev_ms] = average(duration_ms_list[i]);
      fprintf(
          stderr,
          "[%d] %s Time eclapsed: %0.2fms stdev %0.3f, bandwidth: %0.2lfGB/s\n",
          i,
          prefix.c_str(),
          avg_ms,
          stdev_ms,
          double(sizeof(T) * elems) / 1e6 / avg_ms);
    }
  };

  run_bench("Host to Device", bench_h2d);
  run_bench("Device to Host", bench_d2h);
  auto bench_all = concate(bench_h2d, bench_d2h);
  run_bench("Bidirectional Host<->Device", bench_all);
}

int
main(int argc, char **argv) {
  init_flags(&argc, &argv, true);

  if ((FLAGS_run_numa_node > 0 || FLAGS_mem_numa_node > 0) && numa_available() < 0) {
    fprintf(stderr, "numa support not available\n");
  }
  int max_numa_nodes = numa_num_configured_nodes();
  printf("total numa nodes: %d\n", max_numa_nodes);
  if (FLAGS_run_numa_node >= max_numa_nodes || FLAGS_mem_numa_node >= max_numa_nodes) {
    fprintf(
        stderr,
        "invalid numa node set: cpu %d, memory %d >= %d\n",
        FLAGS_run_numa_node,
        FLAGS_mem_numa_node,
        max_numa_nodes);
  }

  // seems not working at all. NCCL don't set NUMA
  ScopedNumaRunBind _numa_cpubind(FLAGS_run_numa_node);
  ScopedNumaMembind _numa_membind(FLAGS_mem_numa_node);

  fprintf(stderr, "runing Device<->Host copy perf with cudaMemcpy\n");
  run_copyd2h<half>(FLAGS_elems, true);

  fprintf(stderr, "runing Device<->Host copy perf with CUDA Cores\n");
  run_copyd2h<half>(FLAGS_elems, false);
  return 0;
}
