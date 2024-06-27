//===- cuda_common.h ---------------------------------------------- C++ ---===//
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

#pragma once

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/float8.h"
#include "flux/flux.h"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include <type_traits>

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                            \
  do {                                                                   \
    cutlass::Status error = status;                                      \
    FLUX_CHECK(error == cutlass::Status::kSuccess)                       \
        << "Got cutlass error: " << cutlassGetStatusString(error) << "(" \
        << static_cast<int>(error) << ") at: " << #status << "\n";       \
  } while (0)

#define CUTLASS_CHECK_RTN(x)                   \
  do {                                         \
    cutlass::Status status = (x);              \
    if (status != cutlass::Status::kSuccess) { \
      return status;                           \
    }                                          \
  } while (0)

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                                   \
  do {                                                                                       \
    cudaError_t error = status;                                                              \
    FLUX_CHECK(error == cudaSuccess) << "Got bad cuda status: " << cudaGetErrorString(error) \
                                     << "(" << error << ") at: " << #status << "\n";         \
  } while (0)

#define NCCL_CHECK(status)                                                                   \
  do {                                                                                       \
    ncclResult_t res = status;                                                               \
    FLUX_CHECK(res == ncclSuccess) << "NCCL failure: " << ncclGetErrorString(res) << " / "   \
                                   << ncclGetLastError(NULL) << " at : " << #status << "\n"; \
  } while (0)

#ifndef CUDA_MEM_ALIGN
#define CUDA_MEM_ALIGN(x) ((x + 31) / 32 * 32)
#endif  // CUDA_MEM_ALIGN

namespace bytedance {
namespace flux {

template <DataTypeEnum E>
auto
to_cuda_dtype(cute::C<E> dtype) {
  if constexpr (dtype == _E4M3{}) {
    return make_declval<__nv_fp8_e4m3>();
  } else if constexpr (dtype == _E5M2{}) {
    return make_declval<__nv_fp8_e5m2>();
  } else if constexpr (dtype == _BF16{}) {
    return make_declval<__nv_bfloat16>();
  } else if constexpr (dtype == _FP16{}) {
    return make_declval<__half>();
  } else if constexpr (dtype == _FP32{}) {
    return make_declval<float>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported dtype!");
  }
}

template <DataTypeEnum E>
auto
to_cutlass_element(cute::C<E> dtype) {
  if constexpr (dtype == _FP32{}) {
    return make_declval<float>();
  } else if constexpr (dtype == _BF16{}) {
    return make_declval<cutlass::bfloat16_t>();
  } else if constexpr (dtype == _FP16{}) {
    return make_declval<cutlass::half_t>();
  } else if constexpr (dtype == _E4M3{}) {
    return make_declval<cutlass::float_e4m3_t>();
  } else if constexpr (dtype == _E5M2{}) {
    return make_declval<cutlass::float_e5m2_t>();
  } else if constexpr (dtype == _Void{}) {
    return make_declval<void>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported dtype!");
  }
}

template <ArchEnum E>
auto
to_cutlass_archtag(cute::C<E> arch) {
  if constexpr (arch == _Sm80{}) {
    return make_declval<cutlass::arch::Sm80>();
  } else if constexpr (arch == _Sm89{}) {
    return make_declval<cutlass::arch::Sm89>();
  } else if constexpr (arch == _Sm90{}) {
    return make_declval<cutlass::arch::Sm90>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported arch!");
  }
}

template <GemmLayoutEnum E>
auto
to_cutlass_layout_a(cute::C<E> layout) {
  if constexpr (layout == _RCR{} or layout == _RRR{} or layout == _RCC{}) {
    return make_declval<cutlass::layout::RowMajor>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported A layout!");
  }
}

template <GemmLayoutEnum E>
constexpr auto
to_cutlass_layout_b(cute::C<E> layout) {
  if constexpr (layout == _RCR{} or layout == _RCC{}) {
    return make_declval<cutlass::layout::ColumnMajor>();
  } else if constexpr (layout == _RRR{}) {
    return make_declval<cutlass::layout::RowMajor>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported B layout!");
  }
}

template <GemmLayoutEnum E>
auto
to_cutlass_layout_c(cute::C<E> layout) {
  if constexpr (layout == _RCR{} or layout == _RRR{}) {
    return make_declval<cutlass::layout::RowMajor>();
  } else if constexpr (layout == _RCC{}) {
    return make_declval<cutlass::layout::ColumnMajor>();
  } else {
    static_assert(cutlass::detail::dependent_false<cute::C<E>>, "unsupported C layout!");
  }
}

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU
 * stream
 */
struct GpuTimer {
  cudaStream_t _stream_id{};
  cudaEvent_t _start;
  cudaEvent_t _stop;

  /// Constructor
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&_start));
    CUDA_CHECK(cudaEventCreate(&_stop));
  }

  /// Destructor
  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(_start));
    CUDA_CHECK(cudaEventDestroy(_stop));
  }

  /// Start the timer for a given stream (defaults to the default stream)
  void
  start(cudaStream_t stream_id = nullptr) {
    _stream_id = stream_id;
    CUDA_CHECK(cudaEventRecord(_start, _stream_id));
  }

  /// Stop the timer
  void
  stop() {
    CUDA_CHECK(cudaEventRecord(_stop, _stream_id));
  }

  /// Return the elapsed time (in milliseconds)
  float
  elapsed_millis() {
    float elapsed = 0.0;
    CUDA_CHECK(cudaEventSynchronize(_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
    return elapsed;
  }
};

// exit if error
void ensure_nvml_init();

// exit if error
const char *get_gpu_device_name(int devid);
// exit if error
unsigned get_pcie_gen(int devid);
// exit if error
int get_sm_count();

}  // namespace flux
}  // namespace bytedance
