//===- gemm_a2a_transpose.cc -------------------------------------- C++ ---===//
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

#include "gemm_a2a_transpose/ths_op/gemm_all2all_transpose.h"
#include "flux/ths_op/ths_pybind.h"
#include "flux/ths_op/flux_shm.h"
namespace bytedance::flux::ths_op {

using GemmAllToAllTransposeOpCls = TorchClassWrapper<GemmAllToAllTransposeOp>;

namespace py = pybind11;

static int _register_gemm_a2a_transpose_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_all2all_transpose", [](py::module &m) {
    py::class_<GemmAllToAllTransposeOpCls>(m, "GemmAllToAllTranspose")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int32_t nnodes,
                        int32_t bs,
                        int32_t seq,
                        int32_t hidden_dim,
                        int32_t head_dim,
                        int32_t n_dim,
                        torch::ScalarType input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        int32_t gqa,
                        PreAttnAllToAllCommOp comm_op) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);

              return new GemmAllToAllTransposeOpCls(
                  std::make_shared<C10dProcessGroup>("", tp_group),
                  nnodes,
                  bs,
                  seq,
                  hidden_dim,
                  head_dim,
                  n_dim,
                  input_dtype,
                  output_dtype,
                  transpose_weight,
                  gqa,
                  comm_op);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("bs"),
            py::arg("seq"),
            py::arg("hidden_dim"),
            py::arg("head_dim"),
            py::arg("n_dim"),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false,
            py::arg("gqa") = 0,
            py::arg("comm_op") = PreAttnAllToAllCommOp::A2ATranspose)
        .def(
            "forward",
            &GemmAllToAllTransposeOpCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0)
        .def(
            "profiling",
            &GemmAllToAllTransposeOpCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();

}  // namespace bytedance::flux::ths_op