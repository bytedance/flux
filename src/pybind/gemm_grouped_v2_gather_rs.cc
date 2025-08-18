//===- gemm_grouped_v2_gather_rs.cc ------------------------------- C++ ---===//
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

#include "c10/util/intrusive_ptr.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"
#include "moe_gather_rs/ths_op/gemm_grouped_v2_gather_rs.h"
#include <c10/core/ScalarType.h>
#include <vector>
#include "moe_gather_rs/moe_utils.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using TopkReduceScatterOpCls = TorchClassWrapper<TopkReduceScatterOp>;
using GemmGroupedV2GatherRSOpCls = TorchClassWrapper<GemmGroupedV2GatherRSOp>;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_grouped_v2_gather_rs", [](py::module &m) {
    m.def(
        "calc_moe_triton_blocked_gather_a",
        [](torch::Tensor splits,
           int ep_start,
           int ep_nexperts,
           int block_size_m,
           torch::Tensor gather_a_index,
           torch::Tensor expert_index,
           intptr_t stream) {
          CHECK_INPUT(splits, at::ScalarType::Int);
          CHECK_INPUT(gather_a_index, at::ScalarType::Int);
          CHECK_INPUT(expert_index, at::ScalarType::Int);
          bytedance::flux::calc_moe_triton_blocked_gather_a(
              splits.data_ptr<int>(),
              ep_start,
              ep_nexperts,
              block_size_m,
              gather_a_index.data_ptr<int>(),
              expert_index.data_ptr<int>(),
              ep_nexperts,
              1024,
              (cudaStream_t)stream);
        },
        py::arg("splits"),
        py::arg("ep_start"),
        py::arg("ep_nexperts"),
        py::arg("block_size_m"),
        py::arg("gather_a_index"),
        py::arg("expert_index"),
        py::arg("stream") = nullptr);
    py::class_<GemmGroupedV2GatherRSOpCls>(m, "GemmGroupedV2GatherRSOp")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int64_t total_num_experts,
                        int64_t max_m,
                        int64_t n_dim,
                        int64_t topk,
                        at::ScalarType output_dtype,
                        int64_t tp_world_size,
                        int64_t ep_world_size,
                        int64_t max_input_groups,
                        int64_t n_split,
                        bool do_all_reduce,
                        bool use_read_mode) {
              return new GemmGroupedV2GatherRSOpCls(
                  std::make_shared<C10dProcessGroup>("", tp_group),
                  total_num_experts,
                  max_m,
                  n_dim,
                  topk,
                  output_dtype,
                  tp_world_size,
                  ep_world_size,
                  max_input_groups,
                  n_split,
                  do_all_reduce,
                  use_read_mode);
            }),
            py::arg("tp_group"),
            py::arg("total_num_experts"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("topk"),
            py::arg("output_dtype"),
            py::arg("tp_world_size"),
            py::arg("ep_world_size"),
            py::arg("max_input_groups") = 1,
            py::arg("n_split") = 4,
            py::arg("do_all_reduce") = false,
            py::arg("use_read_mode") = false)
        .def(
            "forward_gather_rs",
            &GemmGroupedV2GatherRSOpCls::forward_gather_rs,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits"),
            py::arg("scatter_idx"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false)
        .def(
            "forward_gather_rs_triton_aot",
            &GemmGroupedV2GatherRSOpCls::forward_gather_rs_triton_aot,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits"),
            py::arg("scatter_idx"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false)
        .def(
            "profiling",
            &GemmGroupedV2GatherRSOpCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits"),
            py::arg("scatter_idx"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false,
            py::arg("prof_ctx") = nullptr)
        .def(
            "forward_gather_rs_multiple",
            &GemmGroupedV2GatherRSOpCls::forward_gather_rs_multiple,
            py::arg("input"),
            py::arg("weight"),
            py::arg("splits"),
            py::arg("scatter_idx"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_vec_scale") = py::none(),
            py::arg("fast_accum") = true,
            py::arg("sm_margin") = 0,
            py::arg("with_stream_sync") = false);

    py::class_<TopkReduceScatterOpCls>(m, "TopkReduceScatterOp")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int max_m,
                        int n_dim,
                        int topk,
                        at::ScalarType output_dtype,
                        int ep_nexperts,
                        int ep_world_size,
                        std::vector<torch::Tensor> barriers,
                        int n_split,
                        bool do_all_reduce,
                        bool use_read_mode) {
              return new TopkReduceScatterOpCls(
                  std::make_shared<C10dProcessGroup>("", tp_group),
                  max_m,
                  n_dim,
                  topk,
                  output_dtype,
                  ep_nexperts,
                  ep_world_size,
                  barriers,
                  n_split,
                  do_all_reduce,
                  use_read_mode);
            }),
            py::arg("tp_group"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("topk"),
            py::arg("output_dtype"),
            py::arg("num_experts"),
            py::arg("ep_world_size"),
            py::arg("barriers"),
            py::arg("n_split") = 4,
            py::arg("do_all_reduce") = false,
            py::arg("use_read_mode") = false)
        .def(
            "run",
            &TopkReduceScatterOpCls::run,
            py::arg("gemm_outs"),
            py::arg("output"),
            py::arg("ep_start"),
            py::arg("ep_nexperts"),
            py::arg("splits"),
            py::arg("routing_idx"),
            py::arg("output_vec_scales"),
            py::arg("num_thread_blocks"),
            py::arg("cp_stream"))
        .def("reset_buffer", &TopkReduceScatterOpCls::reset_buffer);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
