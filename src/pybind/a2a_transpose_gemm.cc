//===- a2a_transpose_gemm.cc -------------------------------------- C++ ---===//
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

#include "a2a_transpose_gemm/ths_op/all_to_all_transpose_gemm_kernel.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"
namespace bytedance::flux::ths_op {

using AllToAllTransposeGemmOpCls = TorchClassWrapper<AllToAllTransposeGemmOp>;

namespace py = pybind11;

static int _ [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one(
      "all_to_all_transpose_gemm_kernel", [](py::module &m) {
        py::class_<AllToAllTransposeGemmOpCls>(m, "AllToAllTransposeGemm")
            .def(
                py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                            int32_t nnodes,
                            int32_t sp_size,
                            int32_t bs,
                            int32_t num_head,
                            int32_t seq,
                            int32_t head_dim,
                            py::object py_input_dtype,
                            py::object py_output_dtype,
                            int32_t max_num_comm_buf,
                            bool a2a_only) {
                  auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
                  auto output_dtype =
                      py_output_dtype.is(py::none())
                          ? input_dtype
                          : torch::python::detail::py_object_to_dtype(py_output_dtype);
                  return new AllToAllTransposeGemmOpCls(
                      std::make_shared<C10dProcessGroup>("", tp_group),
                      nnodes,
                      sp_size,
                      bs,
                      num_head,
                      seq,
                      head_dim,
                      max_num_comm_buf,
                      input_dtype,
                      output_dtype,
                      a2a_only);
                }),
                py::arg("tp_group"),
                py::arg("nnodes"),
                py::arg("sp_size"),
                py::arg("bs"),
                py::arg("num_head"),
                py::arg("seq"),
                py::arg("head_dim"),
                py::arg("input_dtype"),
                py::arg("output_dtype") = py::none(),
                py::arg("max_num_comm_buf") = 1,
                py::arg("a2a_only") = false)
            .def(
                "forward",
                &AllToAllTransposeGemmOpCls::forward,
                py::arg("input"),
                py::arg("weight"),
                py::arg("seq_lens_cpu") = py::none(),
                py::arg("bias") = py::none(),
                py::arg("output") = py::none(),
                py::arg("input_scale") = py::none(),
                py::arg("weight_scale") = py::none(),
                py::arg("output_scale") = py::none(),
                py::arg("fast_accum") = false,
                py::arg("transpose_weight") = false,
                py::arg("all_to_all_option") = AllToAllOptionWithOptional(),
                py::arg("a2a_transpose_output") = py::none(),
                py::arg("num_comm_sm") = -1,
                py::arg("sm_margin") = 0)
            .def(
                "post_attn_a2a",
                &AllToAllTransposeGemmOpCls::post_attn_a2a,
                py::arg("input"),
                py::arg("seq_lens_cpu") = py::none(),
                py::arg("all_to_all_option") = AllToAllOptionWithOptional(),
                py::arg("num_comm_sm") = -1)
            .def(
                "post_attn_a2a_no_cpy",
                &AllToAllTransposeGemmOpCls::post_attn_a2a_no_cpy,
                py::arg("input"),
                py::arg("seq_lens_cpu") = py::none(),
                py::arg("all_to_all_option") = AllToAllOptionWithOptional(),
                py::arg("num_comm_sm") = -1,
                py::arg("comm_buf_idx") = 0)
            .def("sp_group_barrier_all", &AllToAllTransposeGemmOpCls::sp_group_barrier_all)
            .def(
                "gemm_only",
                &AllToAllTransposeGemmOpCls::gemm_only,
                py::arg("input"),
                py::arg("weight"),
                py::arg("bias") = py::none(),
                py::arg("output") = py::none(),
                py::arg("input_scale") = py::none(),
                py::arg("weight_scale") = py::none(),
                py::arg("output_scale") = py::none(),
                py::arg("fast_accum") = false,
                py::arg("transpose_weight") = false,
                py::arg("num_comm_sm") = -1,
                py::arg("sm_margin") = 0)
            .def(
                "profiling",
                &AllToAllTransposeGemmOpCls::profiling,
                py::arg("input"),
                py::arg("weight"),
                py::arg("bias") = py::none(),
                py::arg("output") = py::none(),
                py::arg("input_scale") = py::none(),
                py::arg("weight_scale") = py::none(),
                py::arg("output_scale") = py::none(),
                py::arg("fast_accum") = false,
                py::arg("transpose_weight") = false,
                py::arg("all_to_all_option") = AllToAllOptionWithOptional(),
                py::arg("a2a_transpose_output") = py::none(),
                py::arg("num_comm_sm") = -1,
                py::arg("sm_margin") = 0,
                py::arg("prof_ctx") = nullptr);
      });
  return 0;
}();

}  // namespace bytedance::flux::ths_op
