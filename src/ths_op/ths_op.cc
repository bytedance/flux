//===- ths_op.cc -------------------------------------------------- C++ ---===//
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

#include "flux/ths_op/ths_op.h"
#include "c10/cuda/CUDAFunctions.h"
#include "flux/flux.h"
#include "flux/cuda/cuda_common.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <ATen/core/jit_type.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "flux/runtime_config.h"

namespace bytedance::flux::ths_op {

DataTypeEnum
from_torch_dtype(at::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case at::ScalarType::Half: {
      return _FP16{};
    }; break;
    case at::ScalarType::BFloat16: {
      return _BF16{};
    }; break;
    case at::ScalarType::Float8_e4m3fn: {
      return _E4M3{};
    }; break;
    case at::ScalarType::Float8_e5m2: {
      return _E5M2{};
    }; break;
    default:
      throw std::runtime_error(
          std::string("unsupported torch_dtype:") + at::toString(torch_dtype));
  }
  return DataTypeEnum{};
}

bool
is_fp8_torch_dtype(at::ScalarType torch_dtype) {
  return (torch_dtype == at::ScalarType::Float8_e4m3fn) ||
         (torch_dtype == at::ScalarType::Float8_e5m2);
}

size_t
torch_dtype_size(at::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16: return 2;
    case at::ScalarType::Float8_e4m3fn:
    case at::ScalarType::Float8_e5m2: return 1;
    default:
      throw std::runtime_error(
          std::string("unsupported torch_dtype:") + at::toString(torch_dtype));
  }
  return 0;
}
void CUDART_CB
closeIpcMemHandleCallback(cudaStream_t stream, cudaError_t status, void *devPtr) {
  cudaIpcCloseMemHandle(devPtr);
}

void CUDART_CB
releaseCpuDataCallback(cudaStream_t stream, cudaError_t status, void *data_ptr) {
  auto ptr_to_release = reinterpret_cast<void **>(data_ptr);
  delete[] ptr_to_release;
}

class AppendCloseIpcMemHandleCallbackRAII {
 public:
  AppendCloseIpcMemHandleCallbackRAII(cudaStream_t stream, void *dev_ptr)
      : stream(stream), dev_ptr(dev_ptr) {}
  ~AppendCloseIpcMemHandleCallbackRAII() {
    cudaStreamAddCallback(stream, closeIpcMemHandleCallback, dev_ptr, 0);
  }

 private:
  cudaStream_t stream;
  void *dev_ptr;
};

PyTuningRecord::PyTuningRecord(
    UnifiedGemmMeta meta, RuntimeConfig rt_conf, UnifiedGemmHParams best_hparams)
    : meta(std::move(meta)), rt_conf(std::move(rt_conf)), best_hparams(std::move(best_hparams)) {}

void
load_tuning_record(PyTuningRecord const &record) {
  TuningConfigRegistry::instance().add(record.meta, record.rt_conf, record.best_hparams);
}

ProfilingContext::ProfilingContext(std::string name) : codegen(std::move(name)), counter(0) {}

TuningConfigGenerator const &
ProfilingContext::get_codegen() const {
  return this->codegen;
}

std::string
ProfilingContext::get_code() const {
  return this->get_codegen().str();
}

std::string
ProfilingContext::to_string_topk(TopHParams const &top_hparams, int topk) const {
  std::ostringstream ss;
  int top_idx = 0;
  for (auto iter = top_hparams.begin(); top_idx < topk && iter != top_hparams.end(); ++iter) {
    ss << " * TopK=" << (++top_idx);
    ss << std::setprecision(3) << " (" << iter->first.first << " ms): " << iter->second;
    if (top_idx < topk) {
      ss << "\n";
    }
  }
  return std::move(ss).str();
}

std::vector<std::string>
ProfilingContext::get_all_prof_results() const {
  std::vector<std::string> ret;
  for (auto const &par : prof_results) {
    std::ostringstream ss;
    auto [meta, rt_conf] = par.first;
    const auto &top_hparams = par.second;
    ss << meta << "\n" << rt_conf << "\n";
    ss << to_string_topk(top_hparams, kReturnTopK);
    ret.emplace_back(std::move(ss).str());
  }
  return ret;
}

std::vector<PyTuningRecord>
ProfilingContext::get_all_records() const {
  std::vector<PyTuningRecord> rets;
  for (auto const &par : prof_results) {
    std::ostringstream ss;
    auto [meta, rt_conf] = par.first;
    const auto &top_hparams = par.second;
    rets.emplace_back(meta, rt_conf, top_hparams.begin()->second);
  }
  return rets;
}

std::string
ProfilingContext::get_latest_prof_result() const {
  FLUX_CHECK(latest_key_ptr != nullptr) << "no latest prof results found";
  auto key = *latest_key_ptr;
  auto iter = prof_results.find(key);
  FLUX_CHECK(iter != prof_results.end())
      << "key not found: (" << key.first << ", " << key.second << ")";
  std::ostringstream ss;
  ss << key.first << "\n" << key.second << "\n";
  ss << to_string_topk(iter->second, kReturnTopK);
  return std::move(ss).str();
}

PyTuningRecord
ProfilingContext::get_latest_record() const {
  FLUX_CHECK(latest_key_ptr != nullptr) << "no latest prof results found";
  auto key = *latest_key_ptr;
  auto iter = prof_results.find(key);
  FLUX_CHECK(iter != prof_results.end())
      << "key not found: (" << key.first << ", " << key.second << ")";
  auto [meta, rt_conf] = key;
  auto const &top_hparams = iter->second;
  return PyTuningRecord(meta, rt_conf, top_hparams.begin()->second);
}

void
ProfilingContext::add(
    UnifiedGemmMeta const &meta,
    RuntimeConfig const &rt_conf,
    UnifiedGemmHParams hparams,
    float elapsed_ms) {
  auto key = std::make_pair(meta, rt_conf);
  if (prof_results.count(key) == 0) {
    prof_results[key] = TopHParams();
  }
  prof_results[key].emplace(std::make_pair(elapsed_ms, counter++), std::move(hparams));
}

UnifiedGemmHParams
ProfilingContext::record_best(UnifiedGemmMeta const &meta, RuntimeConfig const &rt_conf) {
  auto key = std::make_pair(meta, rt_conf);
  latest_key_ptr = std::make_unique<decltype(key)>(key);

  auto iter = prof_results.find(key);
  FLUX_CHECK(iter != prof_results.end()) << "no prof results found for" << meta << ", " << rt_conf;
  auto const &top_hparams = iter->second;
  auto best_hparams = top_hparams.begin()->second;
  codegen.add(meta, rt_conf, best_hparams);
  return best_hparams;
}

// inline torch::Tensor
// create_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
//   check_nvshmem_init();
//   auto option_gpu =
//       at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
//   auto size = torch::elementSize(dtype) *
//               std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
//   return at::from_blob(
//       nvshmem_malloc(size), shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);
// }

/**
 * @return vector<torch::Tensor> of size local_world_size (NOTE: not world_size)
 */
// std::vector<torch::Tensor>
// create_tensor_list(const std::vector<int64_t> &shape, c10::ScalarType dtype) {
//   check_nvshmem_init();
//   auto option_gpu =
//       at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
//   auto size = torch::elementSize(dtype) *
//               std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
//   void *ptr = nvshmem_malloc(size);
//   std::vector<torch::Tensor> tensors;
//   for (int i = 0; i < nvshmem_n_pes(); i++) {
//     if (i == nvshmem_my_pe()) {  // release only local
//       tensors.push_back(
//           at::from_blob(ptr, shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu));
//     } else {
//       auto *rptr = nvshmem_ptr(ptr, i);
//       if (rptr) {
//         tensors.push_back(at::from_blob(nvshmem_ptr(ptr, i), shape, option_gpu));
//       }
//     }
//   }
//   return tensors;
// }

ThsOpsInitRegistry &
ThsOpsInitRegistry::instance() {
  static ThsOpsInitRegistry inst;
  return inst;
}

void
ThsOpsInitRegistry::register_one(std::string name, OpInitFunc &&func) {
  std::lock_guard<std::mutex> guard(register_mutex_);
  registry_.emplace(std::move(name), std::move(func));
}

void
ThsOpsInitRegistry::initialize_all(py::module &m) const {
  std::lock_guard<std::mutex> guard(register_mutex_);
  for (auto const &par : registry_) {
    auto [name, func] = par;
    func(m);
  }
}

DistEnvTP::DistEnvTP(c10::intrusive_ptr<c10d::ProcessGroup> tp_group, int nnodes)
    : DistEnv(tp_group->getRank(), tp_group->getSize(), nnodes), tp_group(tp_group) {}

std::string
DistEnvTP::toString() const {
  std::stringstream ss;
  ss << "DistEnvTP(";
  ss << "rank=" << rank;
  ss << "world_size=" << world_size;
  ss << "nnodes=" << nnodes;
  ss << ")";
  return std::move(ss).str();
}

DistEnvTPWithEP::DistEnvTPWithEP(
    c10::intrusive_ptr<c10d::ProcessGroup> tp_group_,
    int nnodes,
    c10::intrusive_ptr<c10d::ProcessGroup> ep_group)
    : DistEnv(tp_group_->getRank(), tp_group_->getSize(), nnodes),
      tp_group(tp_group_),
      ep_group(ep_group),
      ep_rank(ep_group != nullptr ? ep_group->getRank() : 0),
      ep_size(ep_group != nullptr ? ep_group->getSize() : 1),
      ffn_tp_size(world_size / ep_size),
      ffn_tp_rank(rank % ffn_tp_size) {
  FLUX_CHECK(world_size % ep_size == 0) << world_size << " % " << ep_size << " != 0";
}

std::string
DistEnvTPWithEP::toString() const {
  std::stringstream ss;
  ss << "DistEnvTPWithEP(";
  ss << "rank=" << rank;
  ss << ",world_size=" << world_size;
  ss << ",nnodes=" << nnodes;
  ss << ",ep_rank=" << ep_rank;
  ss << ",ep_size=" << ep_size;
  ss << ")";
  return std::move(ss).str();
}

MoeArguments::MoeArguments(
    int32_t max_ntokens,
    int32_t hidden,
    int32_t ffn_hidden,
    int32_t nexperts,
    int32_t topk,
    c10::ScalarType input_dtype,
    c10::ScalarType output_dtype)
    : max_ntokens(max_ntokens),
      hidden(hidden),
      ffn_hidden(ffn_hidden),
      nexperts(nexperts),
      topk(topk),
      input_dtype(input_dtype),
      output_dtype(output_dtype) {}

void
init_profiling_context(py::module &m) {
  py::class_<ProfilingContext, c10::intrusive_ptr<ProfilingContext>>(m, "ProfilingContext")
      .def(py::init<std::string>())
      .def("get_code", &ProfilingContext::get_code)
      .def("get_all_prof_results", &ProfilingContext::get_all_prof_results)
      .def("get_latest_prof_result", &ProfilingContext::get_latest_prof_result)
      .def("get_latest_record", &ProfilingContext::get_latest_record)
      .def("get_all_records", &ProfilingContext::get_all_records);
}

void
init_tuning_record(py::module &m) {
  py::class_<PyTuningRecord, c10::intrusive_ptr<PyTuningRecord>>(m, "TuningRecord");
}

void
init_dist_env_tp(py::module &m) {
  py::class_<DistEnvTP>(m, "DistEnvTP")
      .def(
          py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group, int32_t nnodes) {
            return new DistEnvTP(tp_group, nnodes);
          }),
          py::arg("tp_group"),
          py::arg("nnodes"))
      .def("__repr__", &DistEnvTP::toString);
}

void
init_dist_env_tp_with_ep(py::module &m) {
  py::class_<DistEnvTPWithEP>(m, "DistEnvTPWithEP")
      .def(
          py::init([](c10::intrusive_ptr<c10d::ProcessGroup> tp_group,
                      int32_t nnodes,
                      c10::intrusive_ptr<c10d::ProcessGroup> ep_group) {
            return new DistEnvTPWithEP(tp_group, nnodes, ep_group);
          }),
          py::arg("tp_group"),
          py::arg("nnodes"),
          py::arg("ep_group") = py::none())
      .def("__repr__", &DistEnvTPWithEP::toString);
}

void
init_moe_arguments(py::module &m) {
  py::class_<MoeArguments, c10::intrusive_ptr<MoeArguments>>(m, "MoeArguments")
      .def(
          py::init([](int32_t max_ntokens,
                      int32_t hidden,
                      int32_t ffn_hidden,
                      int32_t nexperts,
                      int32_t topk,
                      py::object py_input_dtype,
                      py::object py_output_dtype) {
            auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
            auto output_dtype = py_output_dtype.is(py::none())
                                    ? input_dtype
                                    : torch::python::detail::py_object_to_dtype(py_output_dtype);
            return new MoeArguments(
                max_ntokens, hidden, ffn_hidden, nexperts, topk, input_dtype, output_dtype);
          }),
          py::arg("max_ntokens"),
          py::arg("hidden"),
          py::arg("ffn_hidden"),
          py::arg("nexperts"),
          py::arg("topk"),
          py::arg("input_dtype"),
          py::arg("output_dtype") = py::none());
}

void
lazy_init_buffer_tensor(torch::Tensor *tensor, int64_t buffer_size) {
  if (buffer_size <= 0 || tensor == nullptr) {
    return;
  }

  buffer_size = (buffer_size + 127) / 128 * 128;

  if (!tensor->defined() || buffer_size > tensor->numel()) {
    auto options =
        torch::TensorOptions().dtype(c10::ScalarType::Byte).device(torch::Device(torch::kCUDA));
    *tensor = torch::empty({buffer_size}, options);
  }
}

PYBIND11_MODULE(FLUX_TORCH_EXTENSION_NAME, m) {
  m.def("bitwise_check", &bitwise_check);
  m.def("uniform_initialize", &uniform_initialize);
  m.def("load_tuning_record", &load_tuning_record);
  m.def("init_flux_shm", &init_flux_shm);
  m.def(
      "flux_barrier_all_on_stream",
      &pyflux_barrier_all_on_stream,
      py::arg("stream"),
      py::arg("sync_buffers") = py::none(),
      py::arg("rank") = py::none());
  init_tuning_record(m);
  init_profiling_context(m);
  init_dist_env_tp(m);
  init_dist_env_tp_with_ep(m);
  init_moe_arguments(m);

  // Initialize ops in registry
  ThsOpsInitRegistry::instance().initialize_all(m);
}

}  // namespace bytedance::flux::ths_op
