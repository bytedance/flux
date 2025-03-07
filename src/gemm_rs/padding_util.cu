//===- padding_util.cu ---------------------------------------------- C++ ---===//
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

#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "gemm_rs/padding_util.hpp"

namespace bytedance {
namespace flux {

template <class ElementInput, class ElementScale, bool kHasScale>
__global__ void
pad_m_to_TPxTile_kernel(
    ElementInput const *input,
    ElementInput *output,
    ElementScale const *scale,
    ElementScale *padded_scale,
    int M_per_rank,
    int M_padded_per_rank,
    int N,
    int tp_size) {
  using namespace cute;
  int rank = blockIdx.y;
  if (rank >= tp_size) {
    return;
  }
  int num_threads = blockDim.x * gridDim.x;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  Tensor mInput = make_tensor(input, make_layout(make_shape(N, M_per_rank, tp_size)));
  Tensor mOutput = make_tensor(output, make_layout(make_shape(N, M_padded_per_rank, tp_size)));
  Tensor gInput = mInput(_, _, rank);    // (N,M_per_rank)
  Tensor gOutput = mOutput(_, _, rank);  // (N,M_padded_per_rank)
  constexpr int Alignment = 128 / cute::sizeof_bits_v<ElementInput>;

  using VecType = uint128_t;
  Tensor gInput_vec = recast<VecType>(gInput);
  Tensor gOutput_vec = recast<VecType>(gOutput);
  for (int i = thread_id; i < size(gInput_vec); i += num_threads) {
    gOutput_vec(i) = gInput_vec(i);
  }

  if constexpr (kHasScale) {
    // scale'shape: (M_per_rank * tp_size, 1)
    Tensor mScale = make_tensor(scale, make_layout(make_shape(M_per_rank, tp_size)));
    Tensor mPaddedSacle =
        make_tensor(padded_scale, make_layout(make_shape(M_padded_per_rank, tp_size)));
    Tensor gScale = mScale(_, rank);              // (M_per_rank)
    Tensor gPaddedSacle = mPaddedSacle(_, rank);  // (M_padded_per_rank)
    for (int i = thread_id; i < size(gScale); i += num_threads) {
      gPaddedSacle(i) = gScale(i);
    }
  }
}

void
pad_m_to_TPxTile(
    void const *input,
    void const *scale,
    void *output,
    void *padded_scale,
    int m_size,
    int n_size,
    int tp_size,
    int tile_size,
    DataTypeEnum input_dtype,
    DataTypeEnum scale_dtype,
    cudaStream_t stream) {
  FLUX_CHECK_DIV(m_size, tp_size);
  int m_per_rank = m_size / tp_size;
  int m_padded_per_rank = pad_to(m_per_rank, tile_size);
  int alignmnet = 16 / sizeof_dtype(input_dtype);
  FLUX_CHECK_DIV(n_size, alignmnet);
  constexpr int NumThr = 256;
  dim3 block_dim(NumThr);
  dim3 grid_dim;
  grid_dim.y = tp_size;

  int exp_blks = cute::ceil_div(m_per_rank * n_size / alignmnet, NumThr);
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  int exp_capacity = 1024 / NumThr;
  grid_dim.x = std::min(exp_blks, exp_capacity * sm_count);

  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}, _S8{}), cute::make_tuple(_FP32{})),
      [&](auto tup) {
        auto [c_input_dtype, c_scale_dtype] = tup;
        return c_input_dtype == input_dtype && c_scale_dtype == scale_dtype;
      },
      [&](auto tup) {
        auto [c_input_dtype, c_scale_dtype] = tup;
        using ElementInput = decltype(to_cutlass_element(c_input_dtype));
        using ElementScale = decltype(to_cutlass_element(c_scale_dtype));
        bool has_scale = (scale != nullptr && padded_scale != nullptr);
        if (has_scale) {
          pad_m_to_TPxTile_kernel<ElementInput, ElementScale, true>
              <<<grid_dim, block_dim, 0, stream>>>(
                  static_cast<ElementInput const *>(input),
                  static_cast<ElementInput *>(output),
                  static_cast<ElementScale const *>(scale),
                  static_cast<ElementScale *>(padded_scale),
                  m_per_rank,
                  m_padded_per_rank,
                  n_size,
                  tp_size);
        } else {
          pad_m_to_TPxTile_kernel<ElementInput, ElementScale, false>
              <<<grid_dim, block_dim, 0, stream>>>(
                  static_cast<ElementInput const *>(input),
                  static_cast<ElementInput *>(output),
                  static_cast<ElementScale const *>(input),
                  static_cast<ElementScale *>(output),
                  m_per_rank,
                  m_padded_per_rank,
                  n_size,
                  tp_size);
        }
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for dtype: " << input_dtype << ", " << scale_dtype;
      });
}

}  // namespace flux
}  // namespace bytedance
