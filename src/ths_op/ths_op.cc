//===- ths_op.cc -------------------------------------------------- C++ ---===//
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

#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/moe_utils.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"
#include <ATen/Context.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>

namespace bytedance::flux::ths_op {

namespace {
bool
get_deterministic() {
  return
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 2
      at::globalContext().deterministicFillUninitializedMemory();
#else
      at::globalContext().deterministicAlgorithms();
#endif
}
void
set_deterministic(bool use_deterministic_algorithms, bool deterministic_algorithms_warn_only) {
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 2
  at::globalContext().setDeterministicFillUninitializedMemory(use_deterministic_algorithms);
#else
  at::globalContext().setDeterministicAlgorithms(
      use_deterministic_algorithms, deterministic_algorithms_warn_only);
#endif
}
}  // namespace

class TorchDeterministicGuard::TorchDeterministicGuardImpl {
 public:
  TorchDeterministicGuardImpl(bool use_deterministic_algorithms)
      : use_deterministic_algorithms_old_(get_deterministic()),
        deterministic_algorithms_warn_only_old_(
            at::globalContext().deterministicAlgorithmsWarnOnly()) {
    set_deterministic(use_deterministic_algorithms, use_deterministic_algorithms_old_);
  }

  ~TorchDeterministicGuardImpl() { exit(); }

  void
  exit() {
    if (exited_)
      return;
    set_deterministic(
        deterministic_algorithms_warn_only_old_, deterministic_algorithms_warn_only_old_);
  }

 private:
  bool use_deterministic_algorithms_old_;
  bool deterministic_algorithms_warn_only_old_;
  bool exited_ = false;
};

TorchDeterministicGuard::TorchDeterministicGuard(bool use_deterministic_algorithms)
    : impl_(new TorchDeterministicGuardImpl(use_deterministic_algorithms)) {}

TorchDeterministicGuard::~TorchDeterministicGuard() { delete impl_; }

void
TorchDeterministicGuard::exit() {
  impl_->exit();
}

DataTypeEnum
from_torch_dtype(at::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case at::ScalarType::Float: {
      return _FP32{};
    }; break;
    case at::ScalarType::Int: {
      return _S32{};
    }; break;
    case at::ScalarType::Char: {
      return _S8{};
    }; break;
    case at::ScalarType::Half: {
      return _FP16{};
    }; break;
    case at::ScalarType::BFloat16: {
      return _BF16{};
    }; break;
#if TORCH_SUPPOER_FP8
    case at::ScalarType::Float8_e4m3fn: {
      return _E4M3{};
    }; break;
    case at::ScalarType::Float8_e5m2: {
      return _E5M2{};
    }; break;
#endif
    default:
      throw std::runtime_error(
          std::string("unsupported torch_dtype:") + at::toString(torch_dtype));
  }
  return DataTypeEnum{};
}

at::ScalarType
to_torch_dtype(DataTypeEnum dtype) {
  switch (dtype) {
    case _FP32{}: {
      return at::ScalarType::Float;
    }; break;
    case _S32{}: {
      return at::ScalarType::Int;
    }; break;
    case _S8{}: {
      return at::ScalarType::Char;
    }; break;
    case _FP16{}: {
      return at::ScalarType::Half;
    }; break;
    case _BF16{}: {
      return at::ScalarType::BFloat16;
    }; break;
#if TORCH_SUPPOER_FP8
    case _E4M3{}: {
      return at::ScalarType::Float8_e4m3fn;
    }; break;
    case _E5M2{}: {
      return at::ScalarType::Float8_e5m2;
    }; break;
#endif
    default:
      throw std::runtime_error(
          std::string("unsupported dtype: ") + std::string(enum_to_string(dtype)));
  }
  return at::ScalarType::Undefined;
}

bool
is_s8_torch_dtype(at::ScalarType torch_dtype) {
  return torch_dtype == at::ScalarType::Char;
}

bool
is_fp8_torch_dtype(at::ScalarType torch_dtype) {
#if TORCH_SUPPOER_FP8
  return c10::isFloat8Type(torch_dtype);
#else
  return false;
#endif
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
  FLUX_CHECK_DIV(world_size, ep_size);
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

void
copy_tensor_with_kernel_async(const torch::Tensor src, torch::Tensor dst, cudaStream_t stream) {
  FLUX_CHECK_EQ(src.scalar_type(), dst.scalar_type());
  FLUX_CHECK_EQ(src.sizes(), dst.sizes());
  FLUX_CHECK_EQ(src.numel(), dst.numel());
  FLUX_CHECK(src.is_cuda() || src.is_pinned());
  FLUX_CHECK(dst.is_cuda() || dst.is_pinned());
  copy_continous_aligned(dst.data_ptr(), src.data_ptr(), src.nbytes(), 1, 1024, stream);
}

}  // namespace bytedance::flux::ths_op
