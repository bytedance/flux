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
