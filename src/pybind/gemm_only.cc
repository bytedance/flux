//===- gemm_only.cc ----------------------------------------------- C++ ---===//
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

#include "comm_none/ths_op/blockscale_gemm.h"
#include "comm_none/ths_op/gemm_only.h"
#include "flux/ths_op/ths_pybind.h"
#include "comm_none/ths_op/gemm_grouped_v2.h"
#include "comm_none/ths_op/gemm_grouped_v3.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using GemmOnlyCls = TorchClassWrapper<GemmOnly>;
using BlockScaleGemmCls = TorchClassWrapper<BlockScaleGemm>;
using GemmGroupedV2Cls = TorchClassWrapper<GemmGroupedV2>;
using GemmGroupedV3Cls = TorchClassWrapper<GemmGroupedV3>;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_only", [](py::module &m) {
    py::class_<GemmOnlyCls>(m, "GemmOnly")
        .def(
            py::init([](torch::ScalarType input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        bool use_fp8_gemm) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new GemmOnlyCls(input_dtype, output_dtype, transpose_weight, use_fp8_gemm);
            }),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false,
            py::arg("use_fp8_gemm") = false)
        .def(
            "forward",
            &GemmOnlyCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output_buf") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "profiling",
            &GemmOnlyCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output_buf") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("prof_ctx") = nullptr);

    py::class_<BlockScaleGemmCls>(m, "BlockScaleGemm")
        .def(
            py::init([](torch::ScalarType input_dtype,
                        py::object py_output_dtype,
                        int32_t num_streams) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new BlockScaleGemmCls(input_dtype, output_dtype, num_streams);
            }),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("num_streams") = 2)
        .def(
            "forward",
            &BlockScaleGemmCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none())
        .def(
            "forward_multistream",
            &BlockScaleGemmCls::forward_multistream,
            py::arg("input"),
            py::arg("input_splits"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none())
        .def(
            "reference",
            &BlockScaleGemmCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none())
        .def(
            "profiling",
            &BlockScaleGemmCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("prof_ctx") = nullptr);

    py::class_<GemmGroupedV2Cls>(m, "GemmGroupedV2")
        .def(py::init<torch::Tensor, int64_t, at::ScalarType, at::ScalarType>())
        .def(
            "forward",
            &GemmGroupedV2Cls::forward,
            "GemmGroupedV2Cls::forward",
            py::arg("input"),
            py::arg("splits_cpu"),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0);

    py::class_<GemmGroupedV3Cls>(m, "GemmGroupedV3")
        .def(py::init([](torch::Tensor weight, int64_t num_experts) {
          return new GemmGroupedV3Cls(weight, num_experts);
        }))
        .def("forward", &GemmGroupedV3Cls::forward, py::arg("input"), py::arg("splits_cpu"))
        .def(
            "profiling",
            &GemmGroupedV3Cls::profiling,
            py::arg("input"),
            py::arg("splits_cpu"),
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
