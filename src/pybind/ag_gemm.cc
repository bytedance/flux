//===- ag_gemm.cc ------------------------------------------------- C++ ---===//
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

#include "ag_gemm/ths_op/all_gather_gemm_op.h"
#include "ag_gemm/ths_op/all_gather_gemm_op_crossnode.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"
namespace bytedance::flux::ths_op {

using AllGatherGemmOpCls = TorchClassWrapper<AllGatherGemmOp>;
using AllGatherGemmOpCrossNodeCls = TorchClassWrapper<AllGatherGemmOpCrossNode>;

namespace py = pybind11;

static int _ [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("all_gather_gemm_kernel", [](py::module &m) {
    py::class_<AllGatherGemmOpCls>(m, "AGKernel")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int32_t nnodes,
                        int32_t full_m,
                        int32_t n_dim,
                        int32_t k_dim,
                        py::object py_input_dtype,
                        py::object py_output_dtype,
                        bool use_pdl) {
              auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new AllGatherGemmOpCls(
                  std::make_shared<C10dProcessGroup>("", tp_group),
                  nnodes,
                  full_m,
                  n_dim,
                  k_dim,
                  input_dtype,
                  output_dtype,
                  use_pdl);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("full_m"),
            py::arg("n_dim"),
            py::arg("k_dim"),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("use_pdl") = false)
        .def(
            "forward",
            &AllGatherGemmOpCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("transpose_weight") = false,
            py::arg("all_gather_option") = AllGatherOptionWithOptional(),
            py::arg("gathered_input") = py::none())
        .def(
            "gemm_only",
            &AllGatherGemmOpCls::gemm_only,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("transpose_weight") = false)
        .def(
            "profiling",
            &AllGatherGemmOpCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("transpose_weight") = false,
            py::arg("all_gather_option") = AllGatherOptionWithOptional(),
            py::arg("gathered_input") = py::none(),
            py::arg("prof_ctx") = nullptr);

    py::class_<AllGatherGemmOpCrossNodeCls>(m, "AGKernelCrossNode")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        c10::intrusive_ptr<c10d::ProcessGroup> intra_node_group,
                        int32_t nnodes,
                        torch::Tensor output_buffer,
                        int32_t full_m,
                        int32_t n_dim,
                        int32_t k_dim,
                        c10::ScalarType input_dtype,
                        bool transpose_weight = true,
                        bool local_copy = false,
                        c10::optional<AGRingMode> ring_mode_ = c10::nullopt) {
              return new AllGatherGemmOpCrossNodeCls(
                  std::make_shared<C10dProcessGroup>("", tp_group),
                  std::make_shared<C10dProcessGroup>("", intra_node_group),
                  nnodes,
                  output_buffer,
                  full_m,
                  n_dim,
                  k_dim,
                  input_dtype,
                  transpose_weight,
                  local_copy,
                  ring_mode_);
            }),
            py::arg("tp_group"),
            py::arg("intra_node_group"),
            py::arg("nnodes"),
            py::arg("output_buffer"),
            py::arg("full_m"),
            py::arg("n_dim"),
            py::arg("k_dim"),
            py::arg("input_dtype"),
            py::arg("transpose_weight") = true,
            py::arg("local_copy") = false,
            py::arg("ring_mode") = py::none())
        .def("reset_signals", &AllGatherGemmOpCrossNodeCls::reset_signals)
        .def("copy_local", &AllGatherGemmOpCrossNodeCls::copy_local)
        .def("gemm_only", &AllGatherGemmOpCrossNodeCls::gemm_only)
        .def("forward", &AllGatherGemmOpCrossNodeCls::forward);
  });

  return 0;
}();

}  // namespace bytedance::flux::ths_op
