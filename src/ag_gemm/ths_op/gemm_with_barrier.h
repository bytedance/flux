//===- gemm_with_barrier.h ---------------------------------------- C++ ---===//
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
#pragma once
#include <torch/all.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "coll/ths_op/all_gather_types.h"
#include "flux/gemm_hparams.h"
#include "flux/runtime_config.h"
#include "flux/op_registry.h"

namespace bytedance::flux {

class GemmWithBarirer {
 private:
  // used to ThreadblockSwizzle logic.
  int nnodes;
  int world_size;
  int rank;
  // cutlass gemm workspace buffer
  torch::Tensor gemm_buffer;
  OpRegistry::OpPtr cutlass_op;

 private:
  void lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size);

 public:
  GemmWithBarirer(int rank, int world_size, int32_t nnodes);
  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      torch::Tensor barrier,
      bool fast_accum,
      int32_t *producer_signal,
      bool transpose_weight);

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      torch::Tensor barrier,
      bool fast_accum,
      bool transpose_weight,
      c10::optional<UnifiedGemmHParams> const &hparams,
      int32_t *producer_signal,  // this signal only be set in the producer kernel(communication
                                 // kernel for input, e.g. AllGather).
      cudaStream_t stream);

  torch::Tensor initialize(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      torch::Tensor barrier,
      bool fast_accum,
      bool transpose_weight,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream);

  void run(cudaStream_t stream, bool launch_with_pdl);
};

using AGGemmMeta = GemmMeta<
    GemmDTypeConfig<
        DataTypeEnum,
        DataTypeEnum,
        DataTypeEnum,
        DataTypeEnum,
        DataTypeEnum,
        DataTypeEnum>,
    ArchEnum,
    cute::C<bytedance::flux::CommOpEnum::AGKernel>,
    GemmLayoutEnum,
    ImplEnum,
    std::variant<None, GemmV2Meta<bool>, GemmV3Meta<bool, bool>>,
    None>;

AGGemmMeta get_gemm_meta(
    at::ScalarType input_torch_dtype,
    at::ScalarType output_torch_dtype,
    bool transpose_weight,
    bool has_bias,
    bool fast_accum = false);

RuntimeConfig get_rt_config(
    int world_size, int nnodes, int m, int n, int k, AGRingMode ring_mode = AGRingMode::All2All);

}  // namespace bytedance::flux
