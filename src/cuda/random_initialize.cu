//===- random_initialize.cu --------------------------------------- C++ ---===//
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

#include "cute/container/tuple.hpp"
#include "cutlass/numeric_conversion.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "cutlass/util/reference/device/tensor_fill.h"

namespace bytedance::flux {

void
uniform_initialize(
    DataTypeEnum dtype,
    void *ptr,
    size_t capacity,
    uint64_t seed,
    double min,
    double max,
    void *stream = nullptr) {
  tuple_return_if(
      cute::make_tuple(_FP16{}, _BF16{}, _E4M3{}, _E5M2{}),
      [dtype](auto c_dtype) { return dtype == c_dtype; },
      [&](auto c_dtype) {
        auto cu_stream = reinterpret_cast<cudaStream_t>(stream);
        using Element = decltype(to_cutlass_element(c_dtype));
        cutlass::NumericConverter<Element, double> converter;
        cutlass::reference::device::BlockFillRandomUniform(
            static_cast<Element *>(ptr),
            capacity,
            seed,
            converter(max),
            converter(min),
            /*bits=*/-1,
            /*stream=*/cu_stream);
      },
      [dtype]() { FLUX_CHECK(false) << "unsupported dtype: " << dtype; });
}

}  // namespace bytedance::flux
