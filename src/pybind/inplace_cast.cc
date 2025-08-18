//===- inplace_cast.cc -------------------------------------------- C++ ---===//
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

#include "inplace_cast/ths_op/inplace_cast.h"

#include "comm_none/ths_op/gemm_only.h"
#include "flux/ths_op/ths_pybind.h"
#include "inplace_cast/ths_op/helper_ops.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;
using InplaceCastCls = TorchClassWrapper<InplaceCast>;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("inplace_cast", [](py::module &m) {
    py::class_<InplaceCast>(m, "InplaceCast")
        .def(
            py::init([](int32_t data_size) { return new InplaceCast(data_size); }),
            py::arg("data_size"))
        .def("from_fp32_to_bf16", &InplaceCast::from_fp32_to_bf16, py::arg("input"));
    m.def("inplace_cast_fp32_to_bf16", &inplace_cast_fp32_to_bf16);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
