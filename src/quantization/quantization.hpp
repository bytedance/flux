//===- quantization.hpp ------------------------------------------- C++ ---===//
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

namespace bytedance {
namespace flux {

void block_scaled_1d_cast_transpose_impl(
    void *input,
    void *output,
    void *output_t,
    void *scale_inv,
    void *scale_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon,
    dim3 grid,
    const bool return_transpose,
    cudaStream_t stream);

void block_scaled_cast_transpose_impl(
    void *input,
    void *output,
    void *output_t,
    void *scale_inv,
    void *scale_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon,
    dim3 grid,
    const bool return_transpose,
    cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
