//===- ths_pybind.h ----------------------------------------------- C++ ---===//
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

#include <torch/extension.h>
#include "flux/ths_op/ths_op.h"
#include <torch/python.h>
#include <torch/csrc/utils/pybind.h>

#define FLUX_TORCH_EXTENSION_NAME flux_ths_pybind

#if (TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR < 4) || TORCH_VERSION_MAJOR < 2
#include <pybind11/pybind11.h>
namespace pybind11 {
namespace detail {

template <>
struct type_caster<at::ScalarType> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));

  // PYBIND11_TYPE_CASTER defines a member field called value. at::ScalarType
  // cannot be default-initialized, we provide this constructor to explicitly
  // initialize that field. The value doesn't matter as it will be overwritten
  // after a successful call to load.
  type_caster() : value(at::kFloat) {}

  bool
  load(handle src, bool) {
    PyObject *obj = src.ptr();
    if (THPDtype_Check(obj)) {
      value = reinterpret_cast<THPDtype *>(obj)->scalar_type;
      return true;
    }
    return false;
  }

  static handle
  cast(const at::ScalarType &src, return_value_policy /* policy */, handle /* parent */) {
    return Py_NewRef(torch::getTHPDtype(src));
  }
};

}  // namespace detail
}  // namespace pybind11
#endif

namespace py = pybind11;

namespace bytedance::flux::ths_op {
// Registry of functions that register
// functions into module
class ThsOpsInitRegistry {
 public:
  using OpInitFunc = std::function<void(py::module &)>;
  static ThsOpsInitRegistry &instance();
  void register_one(std::string name, OpInitFunc &&func);
  void initialize_all(py::module &m) const;

 private:
  std::map<std::string, OpInitFunc> registry_;
  mutable std::mutex register_mutex_;

  ThsOpsInitRegistry() {}
  ThsOpsInitRegistry(const ThsOpsInitRegistry &) = delete;
  ThsOpsInitRegistry &operator=(const ThsOpsInitRegistry &) = delete;
};

template <typename T>
struct TorchClassWrapper : public torch::CustomClassHolder, T {
 public:
  using T::T;
};

}  // namespace bytedance::flux::ths_op
