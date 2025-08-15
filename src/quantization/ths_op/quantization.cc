//===- quantization.cc -------------------------------------------- C++ ---===//
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

#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/ths_op/util.h"
#include <c10/cuda/CUDAStream.h>
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <ATen/core/jit_type.h>
#include "quantization/ths_op/quantization.h"
#include "quantization/quantization.hpp"

namespace bytedance {
namespace flux {
namespace ths_op {
using torch::Tensor;

class Quantization::QuantizationImpl {
 private:
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const int32_t num_streams;
  static constexpr size_t BLOCK_TILE_DIM = 128;

  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> events;
  cudaEvent_t ready_event;

 public:
  QuantizationImpl(c10::ScalarType input_dtype, c10::ScalarType output_dtype, int32_t num_streams)
      : input_dtype(input_dtype), output_dtype(output_dtype), num_streams(num_streams) {
    FLUX_CHECK(num_streams > 0);
    this->streams.resize(num_streams);
    this->events.resize(num_streams);
    for (int32_t i = 0; i < num_streams; ++i) {
      CUDA_CHECK(cudaStreamCreateWithPriority(
          &this->streams[i], cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
      CUDA_CHECK(cudaEventCreateWithFlags(&this->events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
  }

  ~QuantizationImpl() {
    for (auto &stream : streams) {
      cudaStreamDestroy(stream);
    }
    for (auto &event : events) {
      cudaEventDestroy(event);
    }
    cudaEventDestroy(this->ready_event);
  }

  std::tuple<
      torch::Tensor,
      torch::Tensor,
      c10::optional<torch::Tensor>,
      c10::optional<torch::Tensor>>
  quantize_vector_blockwise(torch::Tensor input, bool return_transpose, float eps) {
    FLUX_CHECK(input.dim() == 2) << "Input must have 2 dimensions.";

    const long num_rows = input.sizes()[0];
    const long num_cols = input.sizes()[1];

    const long num_blocks_x = (num_cols + BLOCK_TILE_DIM - 1) / BLOCK_TILE_DIM;
    const long num_blocks_y = (num_rows + BLOCK_TILE_DIM - 1) / BLOCK_TILE_DIM;

    const bool full_tile = num_cols % BLOCK_TILE_DIM == 0 && num_rows % BLOCK_TILE_DIM == 0;

    torch::Tensor output = torch::empty({num_rows, num_cols}, input.options().dtype(output_dtype));
    torch::Tensor scale = torch::empty(
        {num_blocks_y, num_blocks_x, BLOCK_TILE_DIM}, input.options().dtype(torch::kFloat32));
    torch::Tensor output_t;
    torch::Tensor scale_t;
    if (return_transpose) {
      output_t = torch::empty({num_cols, num_rows}, input.options().dtype(output_dtype));
      scale_t = torch::empty(
          {num_blocks_x, num_blocks_y, BLOCK_TILE_DIM}, input.options().dtype(torch::kFloat32));
    } else {
      output_t = output;
      scale_t = scale;
    }

    size_t scale_stride_x = 0;
    size_t scale_stride_y = 0;
    size_t scale_k = scale.size(1);

    scale_stride_x = 1;
    scale_stride_y = scale_k;

    size_t scale_t_stride_x = 0;
    size_t scale_t_stride_y = 0;

    if (return_transpose) {
      scale_t_stride_x = 1;
      scale_t_stride_y = scale_t.size(1);
    }

    dim3 grid(num_blocks_x, num_blocks_y, 1);
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();

    bytedance::flux::block_scaled_1d_cast_transpose_impl(
        input.data_ptr(),
        output.data_ptr(),
        output_t.data_ptr(),
        scale.data_ptr(),
        scale_t.data_ptr(),
        num_cols,
        num_rows,
        scale_stride_x,
        scale_stride_y,
        scale_t_stride_x,
        scale_t_stride_y,
        eps,
        grid,
        return_transpose,
        current_stream);

    c10::optional<torch::Tensor> output_t_result;
    c10::optional<torch::Tensor> scale_t_result;
    if (return_transpose) {
      output_t_result = output_t;
      scale_t_result = scale_t;
    } else {
      output_t_result = c10::nullopt;
      scale_t_result = c10::nullopt;
    }
    return std::make_tuple(output, scale, output_t_result, scale_t_result);
  }

  std::tuple<
      torch::Tensor,
      torch::Tensor,
      c10::optional<torch::Tensor>,
      c10::optional<torch::Tensor>>
  quantize_square_blockwise(torch::Tensor input, bool return_transpose, float eps) {
    FLUX_CHECK(input.dim() == 2) << "Input must have 2 dimensions.";

    const long num_cols = input.sizes()[1];
    const long num_rows = input.sizes()[0];

    const long num_blocks_x = (num_cols + BLOCK_TILE_DIM - 1) / BLOCK_TILE_DIM;
    const long num_blocks_y = (num_rows + BLOCK_TILE_DIM - 1) / BLOCK_TILE_DIM;

    const bool full_tile = num_cols % BLOCK_TILE_DIM == 0 && num_rows % BLOCK_TILE_DIM == 0;

    torch::Tensor output = torch::empty({num_rows, num_cols}, input.options().dtype(output_dtype));
    torch::Tensor scale =
        torch::empty({num_blocks_y, num_blocks_x}, input.options().dtype(torch::kFloat32));
    torch::Tensor output_t;
    torch::Tensor scale_t;
    if (return_transpose) {
      output_t = torch::empty({num_cols, num_rows}, input.options().dtype(output_dtype));
      scale_t = torch::empty({num_blocks_x, num_blocks_y}, input.options().dtype(torch::kFloat32));
    } else {
      output_t = output;
      scale_t = scale;
    }

    size_t scale_stride_x = 0;
    size_t scale_stride_y = 0;
    size_t scale_k = scale.size(1);

    scale_stride_x = 1;
    scale_stride_y = scale_k;

    size_t scale_t_stride_x = 0;
    size_t scale_t_stride_y = 0;

    if (return_transpose) {
      scale_t_stride_x = 1;
      scale_t_stride_y = scale_t.size(1);
    }

    dim3 grid(num_blocks_x, num_blocks_y, 1);
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();

    bytedance::flux::block_scaled_cast_transpose_impl(
        input.data_ptr(),
        output.data_ptr(),
        output_t.data_ptr(),
        scale.data_ptr(),
        scale_t.data_ptr(),
        num_cols,
        num_rows,
        scale_stride_x,
        scale_stride_y,
        scale_t_stride_x,
        scale_t_stride_y,
        eps,
        grid,
        return_transpose,
        current_stream);

    c10::optional<torch::Tensor> output_t_result;
    c10::optional<torch::Tensor> scale_t_result;
    if (return_transpose) {
      output_t_result = output_t;
      scale_t_result = scale_t;
    } else {
      output_t_result = c10::nullopt;
      scale_t_result = c10::nullopt;
    }
    return std::make_tuple(output, scale, output_t_result, scale_t_result);
  }

  std::tuple<
      torch::Tensor,
      torch::Tensor,
      c10::optional<torch::Tensor>,
      c10::optional<torch::Tensor>>
  batch_quantize_square_blockwise(torch::Tensor input, bool return_transpose, float eps) {
    FLUX_CHECK(input.dim() == 3) << "Input must have 3 dimensions.";

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    const long num_cols = input.sizes()[2];
    const long num_rows = input.sizes()[1];
    const long num_batch = input.sizes()[0];

    const long num_blocks_x = (num_cols + BLOCK_TILE_DIM - 1) / BLOCK_TILE_DIM;
    const long num_blocks_y = (num_rows + BLOCK_TILE_DIM - 1) / BLOCK_TILE_DIM;

    const bool full_tile = num_cols % BLOCK_TILE_DIM == 0 && num_rows % BLOCK_TILE_DIM == 0;

    torch::Tensor output =
        torch::empty({num_batch, num_rows, num_cols}, input.options().dtype(output_dtype));
    torch::Tensor scale = torch::empty(
        {num_batch, num_blocks_y, num_blocks_x}, input.options().dtype(torch::kFloat32));
    torch::Tensor output_t;
    torch::Tensor scale_t;
    if (return_transpose) {
      output_t =
          torch::empty({num_batch, num_cols, num_rows}, input.options().dtype(output_dtype));
      scale_t = torch::empty(
          {num_batch, num_blocks_x, num_blocks_y}, input.options().dtype(torch::kFloat32));
    } else {
      output_t = output;
      scale_t = scale;
    }

    dim3 grid(num_blocks_x, num_blocks_y, 1);

    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    for (int32_t i = 0; i < num_streams; ++i) {
      CUDA_CHECK(cudaStreamWaitEvent(this->streams[i], this->ready_event));
    }

    for (int64_t e_id = 0; e_id < num_batch; ++e_id) {
      size_t scale_stride_x = 0;
      size_t scale_stride_y = 0;
      size_t scale_k = scale.size(2);

      scale_stride_x = 1;
      scale_stride_y = scale_k;

      size_t scale_t_stride_x = 0;
      size_t scale_t_stride_y = 0;

      if (return_transpose) {
        scale_t_stride_x = 1;
        scale_t_stride_y = scale_t.size(2);
      }

      auto cur_stream = this->streams[e_id % num_streams];
      auto cur_event = this->events[e_id % num_streams];
      bytedance::flux::block_scaled_cast_transpose_impl(
          input[e_id].data_ptr(),
          output[e_id].data_ptr(),
          output_t[e_id].data_ptr(),
          scale[e_id].data_ptr(),
          scale_t[e_id].data_ptr(),
          num_cols,
          num_rows,
          scale_stride_x,
          scale_stride_y,
          scale_t_stride_x,
          scale_t_stride_y,
          eps,
          grid,
          return_transpose,
          cur_stream);
      CUDA_CHECK(cudaEventRecord(cur_event, cur_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, cur_event));
    }

    c10::optional<torch::Tensor> output_t_result;
    c10::optional<torch::Tensor> scale_t_result;
    if (return_transpose) {
      output_t_result = output_t;
      scale_t_result = scale_t;
    } else {
      output_t_result = c10::nullopt;
      scale_t_result = c10::nullopt;
    }
    return std::make_tuple(output, scale, output_t_result, scale_t_result);
  }
};

Quantization::Quantization(
    c10::ScalarType input_dtype, c10::ScalarType output_dtype, int32_t num_streams)
    : impl(new QuantizationImpl(input_dtype, output_dtype, num_streams)) {}
Quantization::~Quantization() { delete impl; }

std::
    tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
    Quantization::quantize_vector_blockwise(
        torch::Tensor input, bool return_transpose, float eps) {
  FLUX_CHECK(impl != nullptr) << "Quantization is not initialized";
  return impl->quantize_vector_blockwise(input, return_transpose, eps);
}

std::
    tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
    Quantization::quantize_square_blockwise(
        torch::Tensor input, bool return_transpose, float eps) {
  FLUX_CHECK(impl != nullptr) << "Quantization is not initialized";
  return impl->quantize_square_blockwise(input, return_transpose, eps);
}

std::
    tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>>
    Quantization::batch_quantize_square_blockwise(
        torch::Tensor input, bool return_transpose, float eps) {
  FLUX_CHECK(impl != nullptr) << "Quantization is not initialized";
  return impl->batch_quantize_square_blockwise(input, return_transpose, eps);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
