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

#include "quantization/ths_op/quantization.h"

#include "comm_none/ths_op/gemm_only.h"
#include "flux/ths_op/ths_pybind.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using QuantizationCls = TorchClassWrapper<Quantization>;

static int _ [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("quantization", [](py::module &m) {
    py::class_<Quantization>(m, "Quantization")
        .def(
            py::init([](torch::ScalarType input_dtype,
                        torch::ScalarType output_dtype,
                        int32_t num_streams) {
              return new Quantization(input_dtype, output_dtype, num_streams);
            }),
            py::arg("input_dtype"),
            py::arg("output_dtype"),
            py::arg("num_streams") = 2)
        .def(
            "quantize_vector_blockwise",
            &Quantization::quantize_vector_blockwise,
            py::arg("input"),
            py::arg("return_tranpose") = true,
            py::arg("eps") = 0.0)
        .def(
            "quantize_square_blockwise",
            &Quantization::quantize_square_blockwise,
            py::arg("input"),
            py::arg("return_tranpose") = true,
            py::arg("eps") = 0.0)
        .def(
            "batch_quantize_square_blockwise",
            &Quantization::batch_quantize_square_blockwise,
            py::arg("input"),
            py::arg("return_tranpose") = true,
            py::arg("eps") = 0.0);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
