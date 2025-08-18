//===- flux_coll_op.cc ------------------------------------------ C++ ---===//
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

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "coll/ths_op/all2all_op.h"
#include "coll/ths_op/all2all_single_2d.h"
#include "coll/ths_op/all_gather_op.h"
#include "coll/ths_op/dis_scatter_backward.h"
#include "coll/ths_op/dis_scatter_forward.h"
#include "coll/ths_op/isendrecv.h"
#include "flux/ths_op/flux_shm.h"
#include "flux/ths_op/ths_pybind.h"

namespace bytedance::flux::ths_op {

namespace py = pybind11;

using AllGatherOpCls = ths_op::TorchClassWrapper<AllGatherOp>;

static int _ [[maybe_unused]] = []() {
  ths_op::ThsOpsInitRegistry::instance().register_one("coll_op", [](py::module &m) {
    py::class_<AllGatherOpCls>(m, "AllGatherOp")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                        int nnodes,
                        size_t max_m,
                        size_t k,
                        at::ScalarType input_dtype) {
              return new AllGatherOpCls(
                  std::make_shared<C10dProcessGroup>("", tp_group), nnodes, max_m, k, input_dtype);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("max_m_dim"),
            py::arg("k_dim"),
            py::arg("input_dtype"))
        .def(
            "run",
            [](AllGatherOpCls &self,
               torch::Tensor input,
               torch::optional<torch::Tensor> input_scale,
               const AllGatherOptionWithOptional &opt,
               intptr_t stream) {
              return self.run_with_optional_options(
                  input, input_scale, opt, reinterpret_cast<cudaStream_t>(stream));
            },
            py::arg("input"),
            py::arg("input_scale") = py::none(),
            py::arg("all_gather_option") = AllGatherOptionWithOptional(),
            py::arg("stream") = nullptr)
        .def("local_input_buffer", &AllGatherOpCls::local_input_buffer)
        .def("local_barrier_buffer", &AllGatherOpCls::local_barrier_buffer)
        .def("local_input_scale_buffer", &AllGatherOpCls::local_input_scale_buffer);
  });
  return 0;
}();
#ifdef FLUX_SHM_USE_NVSHMEM
using DisScatterForwardCls = ths_op::TorchClassWrapper<DisScatterForward>;

static int reg_dis_scatter_forward [[maybe_unused]] = []() {
  ths_op::ThsOpsInitRegistry::instance().register_one("dis_scatter_forward", [](py::module &m) {
    py::class_<DisScatterForwardCls>(m, "DisScatterForward")
        .def(
            py::init([](int64_t total_num_experts,
                        int64_t max_m,
                        int64_t n_dim,
                        int64_t topk,
                        int64_t rank,
                        int64_t tp_world_size,
                        int64_t ep_world_size,
                        int64_t local_world_size,
                        float moe_capacity_ratio,
                        int64_t duplicate_comm_buffer,
                        int64_t team) {
              return new DisScatterForwardCls(
                  total_num_experts,
                  max_m,
                  n_dim,
                  topk,
                  rank,
                  tp_world_size,
                  ep_world_size,
                  local_world_size,
                  moe_capacity_ratio,
                  duplicate_comm_buffer,
                  team);
            }),
            py::arg("total_num_experts"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("topk"),
            py::arg("rank"),
            py::arg("tp_world_size"),
            py::arg("ep_world_size"),
            py::arg("local_world_size"),
            py::arg("moe_capacity_ratio") = 1.3,
            py::arg("duplicate_comm_buffer") = 1,
            py::arg("team") = 0)
        .def(
            "forward",
            &DisScatterForwardCls::forward,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits_cpu"),
            py::arg("ep_token_counts"),
            py::arg("sm_margin"),
            py::arg("copy_to_local_tensor") = true)
        .def(
            "forward_gpu",
            &DisScatterForwardCls::forward_gpu,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits"),
            py::arg("ep_token_counts"),
            py::arg("output_tensor"),
            py::arg("sm_margin"),
            py::arg("copy_to_local_tensor") = true)
        .def(
            "forward_no_cpy",
            &DisScatterForwardCls::forward_no_cpy,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits_cpu"),
            py::arg("ep_token_counts"),
            py::arg("sm_margin"),
            py::arg("comm_buffer_id") = 0)
        .def(
            "forward_gpu_no_cpy",
            &DisScatterForwardCls::forward_gpu_no_cpy,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits"),
            py::arg("ep_token_counts"),
            py::arg("output_tensor"),
            py::arg("sm_margin"),
            py::arg("comm_buffer_id") = 0)
        .def(
            "copy_from_output_comm_buffer",
            &DisScatterForwardCls::copy_from_output_comm_buffer,
            py::arg("output"),
            py::arg("comm_buffer_id") = 0)
        .def(
            "copy_to_input_comm_buffer",
            &DisScatterForwardCls::copy_to_input_comm_buffer,
            py::arg("input"),
            py::arg("comm_buffer_id") = 0)
        .def(
            "get_input_comm_buffer",
            &DisScatterForwardCls::get_input_comm_buffer,
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("comm_buffer_id") = 0)
        .def("ep_barrier_all", &DisScatterForwardCls::ep_barrier_all)
        .def(
            "pre_comm_index",
            &DisScatterForwardCls::pre_comm_index,
            py::arg("ep_token_counts_cpu"),
            py::arg("cur_topk_indices"),
            py::arg("cur_topk_values"),
            py::arg("sm_margin"))
        .def(
            "pre_comm_index_gpu",
            &DisScatterForwardCls::pre_comm_index_gpu,
            py::arg("ep_token_counts_gpu"),
            py::arg("cur_topk_indices"),
            py::arg("cur_topk_values"),
            py::arg("ag_topk_indices"),  // output
            py::arg("ag_topk_values"),   // output
            py::arg("sm_margin"))
        .def(
            "profiling",
            &DisScatterForwardCls::profiling,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits_cpu"),
            py::arg("ep_token_counts"),
            py::arg("sm_margin"),
            py::arg("opt_ctx") = nullptr);
  });
  return 0;
}();

using DisScatterBackwardCls = ths_op::TorchClassWrapper<DisScatterBackward>;

static int reg_dis_scatter_backward [[maybe_unused]] = []() {
  ths_op::ThsOpsInitRegistry::instance().register_one("dis_scatter_backward", [](py::module &m) {
    py::class_<DisScatterBackwardCls>(m, "DisScatterBackward")
        .def(
            py::init([](int64_t total_num_experts,
                        int64_t max_m,
                        int64_t n_dim,
                        int64_t topk,
                        int64_t rank,
                        int64_t tp_world_size,
                        int64_t ep_world_size,
                        int64_t local_world_size,
                        float moe_capacity_ratio,
                        int64_t team) {
              return new DisScatterBackwardCls(
                  total_num_experts,
                  max_m,
                  n_dim,
                  topk,
                  rank,
                  tp_world_size,
                  ep_world_size,
                  local_world_size,
                  moe_capacity_ratio,
                  team);
            }),
            py::arg("total_num_experts"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("topk"),
            py::arg("rank"),
            py::arg("tp_world_size"),
            py::arg("ep_world_size"),
            py::arg("local_world_size"),
            py::arg("moe_capacity_ratio") = 1.3,
            py::arg("team") = 0)
        .def(
            "forward",
            &DisScatterBackwardCls::forward,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits_cpu"),
            py::arg("ep_token_counts"),
            py::arg("sm_margin"),
            py::arg("insert_extra_barrier") = false)
        .def(
            "forward_gpu",
            &DisScatterBackwardCls::forward_gpu,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits"),
            py::arg("ep_token_counts"),
            py::arg("output"),
            py::arg("sm_margin"),
            py::arg("insert_extra_barrier") = false)
        .def(
            "forward_no_cpy",
            &DisScatterBackwardCls::forward_no_cpy,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits_cpu"),
            py::arg("ep_token_counts"),
            py::arg("sm_margin"),
            py::arg("insert_extra_barrier") = false)
        .def(
            "forward_gpu_no_cpy",
            &DisScatterBackwardCls::forward_gpu_no_cpy,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits"),
            py::arg("ep_token_counts"),
            py::arg("output"),
            py::arg("sm_margin"),
            py::arg("insert_extra_barrier") = false)
        .def(
            "copy_to_input_comm_buffer",
            &DisScatterBackwardCls::copy_to_input_comm_buffer,
            py::arg("input"))
        .def("ep_barrier_all", &DisScatterBackwardCls::ep_barrier_all)
        .def("local_barrier_all", &DisScatterBackwardCls::local_barrier_all)
        .def(
            "get_input_comm_buffer",
            &DisScatterBackwardCls::get_input_comm_buffer,
            py::arg("shape"),
            py::arg("data_type"))
        .def(
            "profiling",
            &DisScatterBackwardCls::profiling,
            py::arg("input"),
            py::arg("ag_exp_indices"),
            py::arg("ag_scatter_idx"),
            py::arg("splits_cpu"),
            py::arg("ep_token_counts"),
            py::arg("sm_margin"),
            py::arg("opt_ctx") = nullptr);
  });
  return 0;
}();

using All2AllInferenceCls = ths_op::TorchClassWrapper<All2AllInference>;

static int reg_all2all_inference [[maybe_unused]] = []() {
  ths_op::ThsOpsInitRegistry::instance().register_one("all2all_inference", [](py::module &m) {
    py::class_<All2AllInferenceCls>(m, "All2AllInference")
        .def(
            py::init([](int64_t max_m,
                        int64_t n_dim,
                        int64_t rank,
                        int64_t total_num_experts,
                        int64_t world_size,
                        int64_t local_world_size,
                        int64_t max_element_size) {
              return new All2AllInferenceCls(
                  std::move(max_m),
                  std::move(n_dim),
                  std::move(rank),
                  std::move(total_num_experts),
                  std::move(world_size),
                  std::move(local_world_size),
                  std::move(max_element_size));
            }),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("rank"),
            py::arg("total_num_experts"),
            py::arg("world_size"),
            py::arg("local_world_size"),
            py::arg("max_element_size"))
        .def(
            "forward",
            &All2AllInferenceCls::forward,
            py::arg("input_size"),
            py::arg("input_split_cumsum"),
            py::arg("element_size"),
            py::arg("with_scale"))
        .def(
            "get_input_buffer",
            &All2AllInferenceCls::get_input_buffer,
            py::arg("input_shap"),
            py::arg("element_size"),
            py::arg("with_scale"));
  });
  return 0;
}();

using All2AllSinlgeCls = ths_op::TorchClassWrapper<All2AllSingle>;

static int reg_all2all_single [[maybe_unused]] = []() {
  ths_op::ThsOpsInitRegistry::instance().register_one("all2all_single", [](py::module &m) {
    py::class_<All2AllSinlgeCls>(m, "All2AllSingle")
        .def(
            py::init([](c10::intrusive_ptr<c10d::ProcessGroup> pg,
                        int64_t max_split,
                        int64_t n_dim,
                        int64_t local_world_size,
                        at::ScalarType input_dtype,
                        int64_t ep_team) {
              return new All2AllSinlgeCls(
                  std::make_shared<C10dProcessGroup>("", pg),
                  max_split,
                  n_dim,
                  local_world_size,
                  input_dtype,
                  ep_team);
            }),
            py::arg("pg"),
            py::arg("max_split"),
            py::arg("n_dim"),
            py::arg("local_world_size"),
            py::arg("input_dtype"),
            py::arg("ep_team") = 0)
        .def(
            "forward",
            &All2AllSinlgeCls::forward,
            py::arg("input"),
            py::arg("output"),
            py::arg("input_splits"),
            py::arg("output_splits"),
            py::arg("num_comm_sm"));
  });
  return 0;
}();

using AsyncSendRecvCls = ths_op::TorchClassWrapper<AsyncSendRecv>;

static int reg_async_send_recv_single [[maybe_unused]] = []() {
  ths_op::ThsOpsInitRegistry::instance().register_one("async_send_recv", [](py::module &m) {
    py::class_<AsyncSendRecvCls>(m, "AsyncSendRecv")
        .def(
            py::init([](int64_t max_m,
                        int64_t n_dim,
                        int64_t rank,
                        int64_t world_size,
                        at::ScalarType input_dtype,
                        int64_t duplicate) {
              return new AsyncSendRecvCls(max_m, n_dim, rank, world_size, input_dtype, duplicate);
            }),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("rank"),
            py::arg("world_size"),
            py::arg("input_dtype"),
            py::arg("duplicate") = 1)
        .def("get_comm_buffer", &AsyncSendRecv::get_comm_buffer, py::arg("comm_buff_id"))
        .def(
            "read_comm_buffer",
            &AsyncSendRecv::read_comm_buffer,
            py::arg("tgt_rank"),
            py::arg("src_comm_buff_id"),
            py::arg("tgt_comm_buff_id"))
        .def(
            "write_comm_buffer",
            &AsyncSendRecv::write_comm_buffer,
            py::arg("tgt_rank"),
            py::arg("src_comm_buff_id"),
            py::arg("tgt_comm_buff_id"))
        .def(
            "set_signal",
            &AsyncSendRecv::set_signal,
            py::arg("tgt_rank"),
            py::arg("comm_buff_id"),
            py::arg("value"))
        .def("reset_signal", &AsyncSendRecv::reset_signal, py::arg("comm_buff_id"))
        .def(
            "wait_signal_eq",
            &AsyncSendRecv::wait_signal_eq,
            py::arg("comm_buff_id"),
            py::arg("value"));
  });
  return 0;
}();

#endif

}  // namespace bytedance::flux::ths_op
