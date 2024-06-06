//===- module.cpp ------------------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "functions.cpp"  // NOTE: to share code
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/python.h>

PYBIND11_MODULE(pynvshmem, m) {
  m.def("init_with_c10d_pg", &init_with_c10d_pg);
#define NVSHMEMI_TYPENAME_P_PYBIND(TYPENAME, TYPE) m.def(#TYPENAME "_p", &TYPENAME##_p);
  NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_TYPENAME_P_PYBIND)
#undef NVSHMEMI_TYPENAME_P_PYBIND

  m.def("quiet", &quiet);
  m.def("barrier_on_stream", &barrier_on_stream);
  m.def("create_tensor", [](const std::vector<int64_t> shape, py::object dtype) {
    auto cast_dtype = torch::python::detail::py_object_to_dtype(dtype);
    return create_tensor(shape, cast_dtype);
  });
}
