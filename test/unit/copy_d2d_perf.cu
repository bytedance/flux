//===- copy_d2d_perf.cu ------------------------------------------- C++ ---===//
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

// copy between device 0 and 1
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "flux/cuda/cuda_common.h"
#include "cuda_utils.hpp"
#include "utils.hpp"
#include "copy_perf_utils.hpp"

DEFINE_int32(m, 1024, "");
DEFINE_int32(n, 12288, "");
DEFINE_int32(num_blocks, 8, "");
DEFINE_int32(num_threads, 1024, "");
DEFINE_int32(warmup_iters, 5, "");
DEFINE_int32(iters, 10, "");
DEFINE_bool(read, true, "copy is always 0->1, so 0 for push, 1 for pull");
DEFINE_bool(bidirectional, false, "");
DEFINE_bool(verify, true, "");
DEFINE_string(
    links,
    "0,1",
    " 0,1 means GPU 0 will do read/write on GPU 1 depend on FLAGS_read."
    "multiple links are supported by seperated by `;` such as `0,1;0-2`; "
    "FLAGS_links=`0,1` && FLAGS_bidirectional=true equals to FLAGS_links=`0,1;1,0`");

constexpr int kMemcpy = 0;
constexpr int kReduce = 1;
constexpr int kReduceOffset = 1;
std::array<std::string, 2> kExpNames = {"memcpy", "reduce"};

void
enable_p2p_access(const std::vector<std::pair<int, int>> &gpu_pairs) {
  for (const auto &gpu_pair : gpu_pairs) {
    CUDA_CHECK(cudaSetDevice(gpu_pair.first));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(gpu_pair.second, 0));
  }
}

template <typename T = half, typename VecType = uint4>
class CopyPerf {
 public:
  CopyPerf(size_t elems, int src, int dst, bool read_not_write)
      : elems_(elems),
        src_gpu_(src),
        dst_gpu_(dst),
        read_not_write_(read_not_write),
        src_tensor_(src, elems, 0),  // dsrc -> ddst is 0 -> 1
        dst_tensor_(dst, elems, 0),
        stream_(src) {
    T *src_ptr = src_tensor_.cpu_ptr();
    T *dst_ptr = dst_tensor_.cpu_ptr();
    for (size_t i = 0; i < elems; i++) {
      src_ptr[i] = (T)(float)(i % 1024);
      dst_ptr[i] = (T)(float)kReduceOffset;
    }
    src_tensor_.sync_device();
    dst_tensor_.sync_device();
  }

  void
  run_reduce() {
    ScopedDevice _(src_gpu_);
    T *src_ptr = read_not_write_ ? dst_tensor_.ptr() : src_tensor_.ptr();
    T *dst_ptr = read_not_write_ ? src_tensor_.ptr() : dst_tensor_.ptr();
    run_reduce_kernel<<<FLAGS_num_blocks, FLAGS_num_threads, 0, stream_>>>(
        src_ptr, dst_ptr, elems_);
    CUDA_CHECK(cudaGetLastError());
  }

  bool
  verify_reduce() {
    // verify
    printf("check if matches... ");
    if (read_not_write_) {
      // reduce to src_tensor, so we keep src_tensor.cpu() (src) to dst_tensor.cpu()
      // and sync src_tensor back as target
      // target: src_tensor.cpu()
      // before reduce: dst_tensor.cpu()
      std::memcpy(dst_tensor_.cpu_ptr(), src_tensor_.cpu_ptr(), elems_ * sizeof(T));
    } else {
      // reduce to dst_tensor. so dst_tensor.cpu() as target
      // target: dst_tensor.cpu()
      // before reduce: src_tensor.cpu()
      dst_tensor_.sync_host();
    }
    src_tensor_.sync_host();
    T *src_ptr = read_not_write_ ? dst_tensor_.cpu_ptr() : src_tensor_.cpu_ptr();
    T *dst_ptr = read_not_write_ ? src_tensor_.cpu_ptr() : dst_tensor_.cpu_ptr();
    for (size_t i = 0; i < elems_; i++) {
      float src_value = (float)src_ptr[i], dst_value = (float)dst_ptr[i];
      if (std::fabs(src_value + kReduceOffset - dst_value) > 5e-1) {
        fprintf(stderr, "not matches for %ld: %f + 1 vs %f!\n", i, src_value, dst_value);
        return false;
      }
    }
    printf("matches...");
    return true;
  }

  cudaStream_t
  stream() {
    return stream_;
  }

  void
  run_memcpy(bool use_cudamemcpy) {
    ScopedDevice _(src_gpu_);
    T *src_ptr = read_not_write_ ? dst_tensor_.ptr() : src_tensor_.ptr();
    T *dst_ptr = read_not_write_ ? src_tensor_.ptr() : dst_tensor_.ptr();
    if (use_cudamemcpy) {
      CUDA_CHECK(cudaMemcpyAsync(
          dst_ptr, src_ptr, elems_ * sizeof(T), cudaMemcpyDeviceToDevice, stream_));
    } else {
      run_copy_kernel<T, VecType>
          <<<FLAGS_num_blocks, FLAGS_num_threads, 0, stream_>>>(src_ptr, dst_ptr, elems_);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  bool
  verify_memcpy() {
    // check
    printf("check if matches... ");
    src_tensor_.sync_host();
    dst_tensor_.sync_host();
    T *src_ptr = src_tensor_.cpu_ptr();
    T *dst_ptr = dst_tensor_.cpu_ptr();
    for (size_t i = 0; i < elems_; i++) {
      float src_value = (float)src_ptr[i], dst_value = (float)dst_ptr[i];
      if (std::fabs(src_value - dst_value) > 5e-5) {
        fprintf(stderr, "not matches for %ld: %f vs %f!\n", i, src_value, dst_value);
        return false;
      }
    }
    printf("matches... ");
    return true;
  }

 private:
  size_t elems_;
  int src_gpu_, dst_gpu_;
  bool read_not_write_;
  DeviceVector<T> src_tensor_;
  DeviceVector<T> dst_tensor_;
  CudaStream stream_;
};

// run exp
// @param op[in]: 0 for memcpy, 1 for reduce
template <typename T = half, typename VecType = uint4>
void
run_exp(int op, bool use_cudamemcpy) {
  printf("==== %s ====\n", kExpNames[op].c_str());
  if (use_cudamemcpy) {
    printf("copy with cudaMemcpy\n");
  } else {
    printf("by CUDA Core T=%s VecType=%s\n", type_name<T>().c_str(), type_name<VecType>().c_str());
  }
  size_t elems = FLAGS_m * FLAGS_n;
  const auto &links = parse_links(FLAGS_links, FLAGS_bidirectional);
  size_t link_size = links.size();
  std::vector<CopyPerf<half, VecType>> perfs;
  std::vector<cudaStream_t> streams;
  std::vector<CudaEventTimer> timers;
  perfs.reserve(link_size);
  timers.reserve(link_size);
  streams.reserve(link_size);
  for (const auto &[src_gpu, dst_gpu] : links) {
    perfs.emplace_back(elems, src_gpu, dst_gpu, FLAGS_read);
    streams.emplace_back(perfs.back().stream());
    timers.emplace_back(src_gpu);
  }
  auto run_all = [&]() {
    for (auto &perf : perfs) {
      if (op == kMemcpy) {
        perf.run_memcpy(use_cudamemcpy);
      } else {
        perf.run_reduce();
      }
    }
  };
  auto start_timer = [&]() {
    for (int i = 0; i < link_size; i++) {
      auto &timer = timers[i];
      auto &stream = streams[i];
      timer.Start(stream);
    }
  };
  auto stop_timer = [&]() {
    for (auto &timer : timers) {
      timer.Stop();
    }
  };
  auto get_avg_time = [&]() {
    float duration_sum = 0;
    for (auto &timer : timers) {
      duration_sum += timer.GetEclapsedTime();
    }
    return duration_sum / link_size;
  };
  auto run_iters = [&](int iters) {
    start_timer();
    for (int i = 0; i < iters; i++) {
      run_all();
    }
    stop_timer();
    return get_avg_time();
  };
  if (FLAGS_verify) {
    run_iters(1);
    for (auto &perf : perfs) {
      bool matches = op == kMemcpy ? perf.verify_memcpy() : perf.verify_reduce();
      if (!matches) {
        exit(-1);
      }
    }
  }

  run_iters(FLAGS_warmup_iters);
  printf("warmup done\n");
  float duration_ms = run_iters(FLAGS_iters);
  double bw_gb = FLAGS_m / double(duration_ms) * FLAGS_n * sizeof(T) / 1e6 * FLAGS_iters;
  printf(
      "eclapsed time: %0.2fms/iter for %d iters, BW: %0.2lfGB/s\n",
      duration_ms / FLAGS_iters,
      FLAGS_iters,
      bw_gb);
}

int
main(int argc, char **argv) {
  init_flags(&argc, &argv, true);

  const auto &links = parse_links(FLAGS_links, FLAGS_bidirectional, true);
  enable_p2p_access(links);

  run_exp<half, half2>(kReduce, false);
  run_exp<half, uint4>(kMemcpy, false);
  run_exp<half, uint>(kMemcpy, false);
  run_exp<half, uint4>(kMemcpy, true);

  return 0;
}
