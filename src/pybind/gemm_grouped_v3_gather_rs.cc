//===- gemm_grouped_v3_gather_rs.cc ------------------------------- C++ ---===//
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
#include "moe_gather_rs/ths_op/gemm_grouped_v3_gather_rs.h"
#include "moe_gather_rs/ths_op/moe_utils.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using GemmGroupedV3GatherRSOpCls = TorchClassWrapper<GemmGroupedV3GatherRS>;
using TransportOpCls = TorchClassWrapper<TransportOp>;
using All2AllOpCls = TorchClassWrapper<All2AllOp>;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_grouped_v3_gather_rs", [](py::module &m) {
#ifdef FLUX_SHM_USE_NVSHMEM
    m.def("topk_scatter_reduce", &topk_scatter_reduce);
#endif
    py::class_<GemmGroupedV3GatherRSOpCls>(m, "GemmGroupedV3GatherRS")
        .def(
            py::init([](int64_t total_num_experts,
                        int64_t max_m,
                        int64_t n_dim,
                        int64_t topk,
                        int64_t rank,
                        int64_t world_size,
                        int64_t tp_world_size,
                        int64_t ep_world_size,
                        int64_t max_input_groups) {
              return new GemmGroupedV3GatherRSOpCls(
                  std::move(total_num_experts),
                  std::move(max_m),
                  std::move(n_dim),
                  std::move(topk),
                  std::move(rank),
                  std::move(world_size),
                  std::move(tp_world_size),
                  std::move(ep_world_size),
                  std::move(max_input_groups));
            }),
            py::arg("total_num_experts"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("topk"),
            py::arg("rank"),
            py::arg("world_size"),
            py::arg("tp_world_size"),
            py::arg("ep_world_size"),
            py::arg("max_input_groups") = 1)
        .def(
            "forward_gather_rs",
            &GemmGroupedV3GatherRSOpCls::forward_gather_rs,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits_cpu"),
            py::arg("routing_idx"),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false)
        .def(
            "forward_gather_rs_no_zerobuffer",
            &GemmGroupedV3GatherRSOpCls::forward_gather_rs_no_zerobuffer,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits_cpu"),
            py::arg("routing_idx"),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false)
        .def("ep_zero_buffer", &GemmGroupedV3GatherRSOpCls::ep_zero_buffer)
        .def(
            "profiling",
            &GemmGroupedV3GatherRSOpCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits_cpu"),
            py::arg("routing_idx"),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false,
            py::arg("prof_ctx") = nullptr)
        .def(
            "forward_gather_rs_multiple",
            &GemmGroupedV3GatherRSOpCls::forward_gather_rs_multiple,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits_cpu"),
            py::arg("routing_idx"),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = true,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false);

    py::class_<TransportOpCls>(m, "TransportOp")
        .def(py::init<int64_t, int64_t, torch::Tensor>())
        .def("copy_by_sm", &TransportOpCls::copy_by_sm)
        .def("copy_by_ce", &TransportOpCls::copy_by_ce);

    py::class_<All2AllOpCls>(m, "All2AllOp")
        .def(py::init<int64_t, int64_t, torch::Tensor>())
        .def("forward", &All2AllOpCls::forward);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
