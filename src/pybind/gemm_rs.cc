//===- gemm_rs.cc ------------------------------------------------- C++ ---===//
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
#include "gemm_rs/ths_op/gemm_reduce_scatter.h"
#include "gemm_rs/ths_op/helper_ops.h"
#include "flux/ths_op/ths_pybind.h"
#include "gemm_rs/tile_scheduler/threadblock_swizzle_segment_util.hpp"

namespace bytedance::flux::ths_op {
using GemmRSOpCls = TorchClassWrapper<GemmRS>;
namespace py = pybind11;

static int _register_gemm_rs_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_reduce_scatter", [](py::module &m) {
    py::class_<GemmRSOpCls>(m, "GemmRS")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int32_t nnodes,
                        int32_t max_m,
                        int32_t n_dim,
                        torch::ScalarType input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        bool fuse_reduction,
                        bool ring_reduction) {
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);

              return new GemmRSOpCls(
                  std::make_shared<C10dProcessGroup>("", tp_group),
                  nnodes,
                  max_m,
                  n_dim,
                  input_dtype,
                  output_dtype,
                  transpose_weight,
                  fuse_reduction,
                  ring_reduction);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false,
            py::arg("fuse_reduction") = false,
            py::arg("ring_reduction") = false)
        .def("zero_buffers", &GemmRSOpCls::zero_buffers)
        .def(
            "forward",
            &GemmRSOpCls::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("reduce_scatter_option") = ReduceScatterOptionWithOptional())
        .def(
            "forward_barrier",
            &GemmRSOpCls::forward_barrier,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "forward_reduce_scatter",
            &GemmRSOpCls::forward_reduce_scatter,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none())
        .def(
            "profiling",
            &GemmRSOpCls::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("prof_ctx") = nullptr,
            py::arg("reduce_scatter_option") = ReduceScatterOptionWithOptional());

    py::class_<SegmentInfo>(m, "SegmentInfo")
        .def_readonly("segment_origin", &SegmentInfo::segment_origin)
        .def_readonly("size", &SegmentInfo::size)
        .def_readonly("tile_m_start_origin", &SegmentInfo::tile_m_start_origin)
        .def_readonly("tile_m_start_new", &SegmentInfo::tile_m_start_new);

    m.def(
        "get_gemm_rs_threadblock_segments_info",
        [](std::vector<int> problem_shape,  // (M, N)
           std::vector<int> tiled_shape,
           int rank,
           int world_size,
           int sub_world_size,
           int nnodes,
           bool use_2d_ring,
           bool per_tile_flags) {
          FLUX_CHECK(problem_shape.size() == 2 || problem_shape.size() == 3);
          FLUX_CHECK(tiled_shape.size() == 2 || tiled_shape.size() == 3);
          ThreadBlockSwizzleSegmentUtils helper(
              cutlass::gemm::GemmCoord(problem_shape[0], problem_shape[1], 1),
              cutlass::gemm::GemmCoord(tiled_shape[0], tiled_shape[1], 1),
              rank,
              world_size,
              sub_world_size,
              nnodes,
              use_2d_ring,
              per_tile_flags);
          return helper.get_segments_info();
        },
        py::arg("problem_shape"),
        py::arg("tiled_shape"),
        py::arg("rank"),
        py::arg("world_size"),
        py::arg("sub_world_size"),
        py::arg("nnodes"),
        py::arg("use_2d_ring"),
        py::arg("per_tile_flags"));
    m.def(
        "calc_gemm_rs_threadblock_segments_info",
        [](torch::Tensor segments,
           std::vector<int> problem_shape,  // (M, N)
           std::vector<int> tiled_shape,    // (TILED_M, TILED_N)
           int rank,
           int world_size,
           int sub_world_size,
           int nnodes,
           bool use_2d_ring,
           bool per_tile_flags) {
          FLUX_CHECK(problem_shape.size() == 2 || problem_shape.size() == 3);
          FLUX_CHECK(tiled_shape.size() == 2 || tiled_shape.size() == 3);
          ThreadBlockSwizzleSegmentUtils helper(
              cutlass::gemm::GemmCoord(problem_shape[0], problem_shape[1], 1),
              cutlass::gemm::GemmCoord(tiled_shape[0], tiled_shape[1], 1),
              rank,
              world_size,
              sub_world_size,
              nnodes,
              use_2d_ring,
              per_tile_flags);
          return helper.calc_segments_info((SegmentInfo *)segments.data_ptr());
        },
        py::arg("segments"),
        py::arg("problem_shape"),
        py::arg("tiled_shape"),
        py::arg("rank"),
        py::arg("world_size"),
        py::arg("sub_world_size"),
        py::arg("nnodes"),
        py::arg("use_2d_ring"),
        py::arg("per_tile_flags"));
    m.def("bsr_reduce", &bsr_reduce);
    m.def("pad_m_to_TPxTile", &pad_m_to_TPxTile);
  });
  return 0;
}();
}  // namespace bytedance::flux::ths_op
