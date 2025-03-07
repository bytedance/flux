//===- gemm_grouped_v2_ag_scatter.cc ------------------------------ C++ ---===//
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

#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"
#include "moe_ag_scatter/ths_op/gemm_grouped_v2_ag_scatter.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using GemmGroupedV2AGScatterOpCls = TorchClassWrapper<GemmGroupedV2AGScatterOp>;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_grouped_v2_ag_scatter", [](py::module &m) {
    m.def("prepare_moe_ag_scatter_args", &prepare_moe_ag_scatter_args);
    py::class_<GemmGroupedV2AGScatterOpCls>(m, "GemmGroupedV2AGScatterOp")
        .def(
            py::init([](DistEnvTPWithEP &tp_env, MoeArguments &moe_args) {
              return new GemmGroupedV2AGScatterOpCls(
                  std::make_shared<C10dProcessGroup>("", tp_env.tp_group),
                  tp_env.ep_size,
                  moe_args.max_ntokens,
                  moe_args.ffn_hidden,
                  moe_args.hidden,
                  moe_args.nexperts,
                  moe_args.topk,
                  moe_args.input_dtype,
                  moe_args.output_dtype);
            }),
            py::arg("tp_env"),
            py::arg("moe_args"))
        .def("clear_buffers", &GemmGroupedV2AGScatterOpCls::clear_buffers)
        .def(
            "forward",
            &GemmGroupedV2AGScatterOpCls::forward,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("ag_option") = AllGatherOptionWithOptional())
        .def(
            "forward_triton_aot",
            &GemmGroupedV2AGScatterOpCls::forward_triton_aot,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("ag_option") = AllGatherOptionWithOptional())
        .def(
            "forward_multiple_weights",
            &GemmGroupedV2AGScatterOpCls::forward_multiple_weights,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("ag_option") = AllGatherOptionWithOptional())
        .def(
            "profiling",
            &GemmGroupedV2AGScatterOpCls::profiling,
            py::arg("inputs_shard"),
            py::arg("weights"),
            py::arg("splits_gpu"),
            py::arg("scatter_index"),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("outputs_buf") = py::none(),
            py::arg("allgather_output") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("ag_option") = AllGatherOptionWithOptional(),
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
