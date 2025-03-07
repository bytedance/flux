//===- copy_d2d_cute_perf.cu -------------------------------------- C++ ---===//
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
#include <cmath>
#include <cstdio>
#include <vector>
#include "cuda_utils.hpp"
#include "flux/flux.h"
#include "flux/utils.h"
#include "utils.hpp"
#include "copy_perf_utils.hpp"

using cute::get;

DEFINE_int32(m, 1024, "");
DEFINE_int32(n, 12288, "");
DEFINE_int32(n_stride, 0, "");
DEFINE_int32(num_blocks, 8, "");
DEFINE_int32(num_threads, 1024, "");
DEFINE_int32(warmup_iters, 5, "");
DEFINE_int32(iters, 10, "");
DEFINE_int32(run_devid, 0, "copy is always 0->1, so 0 for push, 1 for pull");
DEFINE_bool(bidirectional, false, "");
DEFINE_bool(verify, true, "");
DEFINE_bool(strided, false, "");

template <typename T = half, typename VecType = uint4>
class CopyPerf {
 public:
  CopyPerf(size_t m, size_t n, size_t n_stride, int run_devid)
      : elems_(m * std::max(n, n_stride)),
        m_(m),
        n_(n),
        n_stride_(n_stride),
        run_devid_(run_devid),
        run_devid2_(run_devid == 0 ? 1 : 0),
        hsrc_(elems_, 0),
        hdst_origin_(elems_, 0),
        dsrc_(0, elems_, 0),   // dsrc -> ddst is 0 -> 1
        dsrc2_(1, elems_, 0),  // dsrc2 -> ddst2 is 1 -> 0
        ddst_(1, elems_, 0),
        ddst2_(0, elems_, 0),
        stream_(run_devid_),
        stream2_(run_devid2_),
        timer_(run_devid) {
    FLUX_CHECK(n_stride_ >= n_) << " n_stride >= n expected: got n_stride " << n_stride_
                                << " vs n " << n_;
    for (size_t i = 0; i < elems_; i++) {
      hsrc_[i] = (T)(float)(i % 1024);
      hdst_origin_[i] = (T)(float)(i * i % 1024);
    }
    CUDA_CHECK(cudaMemcpy(dsrc_, hsrc_.data(), elems_ * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ddst_, hdst_origin_.data(), elems_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void
  run_memcpy(bool copy_per_tile, bool use_cudamemcpy) {
    auto func = [&](const T *src, T *dst, cudaStream_t stream) {
      if (use_cudamemcpy) {
        if (n_ == n_stride_) {
          FLUX_LOG_FIRST_N(INFO, 1) << "cudaMemcpyAsync\n";
          CUDA_CHECK(
              cudaMemcpyAsync(dst, src, elems_ * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        } else {
          FLUX_LOG_FIRST_N(INFO, 1) << "cudaMemcpy2DAsync\n";
          CUDA_CHECK(cudaMemcpy2DAsync(
              dst,
              n_stride_ * sizeof(T),
              src,
              n_ * sizeof(T),
              n_ * sizeof(T),
              m_,
              cudaMemcpyDeviceToDevice,
              stream));
        }
      } else {
        if (copy_per_tile) {
          run_copy_per_tile_cute(
              src, dst, m_, n_, n_stride_, FLAGS_num_blocks, FLAGS_num_threads, stream);
        } else {
          if (n_ == n_stride_) {
            run_copy_continous_cute(
                src, dst, m_, n_, n_stride_, FLAGS_num_blocks, FLAGS_num_threads, stream);
          } else {
            fprintf(stderr, "cannot copy continous with strided\n");
          }
        }
        CUDA_CHECK(cudaGetLastError());
      }
    };
    auto run_iters = [&](int iters) {
      timer_.Start(stream_);
      for (int i = 0; i < iters; i++) {
        ScopedDevice _(run_devid_);
        func(dsrc_, ddst_, stream_);
        if (FLAGS_bidirectional) {
          ScopedDevice _(run_devid2_);
          func(dsrc2_, ddst2_, stream2_);
        }
      }
      timer_.Stop();
      stream_.sync();
      if (FLAGS_bidirectional) {
        stream2_.sync();
      }
      return timer_.GetEclapsedTime();
    };

    if (FLAGS_verify) {
      run_iters(1);
      std::vector<T> hdst(elems_);
      CUDA_CHECK(cudaMemcpy(hdst.data(), ddst_, elems_ * sizeof(T), cudaMemcpyDeviceToHost));

      // check
      printf("check if matches... ");
      for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
          int index = i * n_stride_ + j;
          float src_val = (float)hsrc_[index];
          float dst_val = (float)hdst[index];
          if (std::fabs(src_val - dst_val) > 5e-1) {
            std::cerr << "not matches at " << i << "," << j << ": " << src_val << " vs " << dst_val
                      << "\n";
            exit(-1);
          }
        }
      }
      printf("matches... ");
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

 private:
  size_t elems_;
  size_t m_, n_, n_stride_;
  int run_devid_, run_devid2_;
  std::vector<T> hsrc_;
  std::vector<T> hdst_origin_;

  DeviceVector<T> dsrc_, dsrc2_;
  DeviceVector<T> ddst_, ddst2_;

  CudaStream stream_, stream2_;
  CudaEventTimer timer_;
};

template <typename T = half, typename VecType = uint4>
void
run_memcpy(bool copy_per_tile, bool use_cudamemcpy) {
  if (use_cudamemcpy) {
    printf("copy with cudaMemcpy\n");
  } else {
    printf(
        "copy by CUDA Core T=%s VecType=%s copy_per_tile=%d\n",
        type_name<T>().c_str(),
        type_name<VecType>().c_str(),
        copy_per_tile);
  }
  CopyPerf<half, VecType> op(
      FLAGS_m, FLAGS_n, FLAGS_strided ? FLAGS_n_stride : FLAGS_n, FLAGS_run_devid);
  op.run_memcpy(copy_per_tile, use_cudamemcpy);
}

int
main(int argc, char **argv) {
  init_flags(&argc, &argv, true);

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

  run_memcpy<half, uint4>(true, false);
  if (!(FLAGS_strided && FLAGS_n_stride != FLAGS_n)) {
    run_memcpy<half, uint4>(false, false);
    run_memcpy<half, uint>(false, false);
  }
  run_memcpy<half, uint4>(true, true);

  return 0;
}
