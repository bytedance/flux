//===- gemm_grouped_v3_ag_scatter.cc ------------------------------ C++ ---===//
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

#include "flux/ths_op/ths_pybind.h"
#include "moe_ag_scatter/ths_op/gemm_grouped_v3_ag_scatter.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using GemmGroupedV3AGScatterOpCls = TorchClassWrapper<GemmGroupedV3AGScatterOp>;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_grouped_v3_ag_scatter", [](py::module &m) {
    py::class_<GemmGroupedV3AGScatterOpCls>(m, "GemmGroupedV3AGScatter")
        .def(py::init<DistEnvTPWithEP, MoeArguments>(), py::arg("tp_env"), py::arg("moe_args"))
        .def("clear_buffers", &GemmGroupedV3AGScatterOpCls::clear_buffers)
        .def(
            "forward",
            &GemmGroupedV3AGScatterOpCls::forward,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0)
        .def(
            "forward_multiple_weights",
            &GemmGroupedV3AGScatterOpCls::forward_multiple_weights,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0)
        .def(
            "profiling",
            &GemmGroupedV3AGScatterOpCls::profiling,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
