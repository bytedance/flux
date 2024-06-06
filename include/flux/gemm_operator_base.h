//===- gemm_operator_base.h --------------------------------------- C++ ---===//
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

#pragma once
#include <any>
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include <stdexcept>
#include <type_traits>

namespace bytedance::flux {

// Simple Wrapper for CUTLASS's `GemmKernel` with `GemmMetaT` and `GemmHParamsT` as part
// of its type.
//
// The purpose of this wrapper is to prevent a bug that will cause "invalid argument error".
//
// When & How is the bug triggered?
// When the same type of cutlass::device_kernel<Operator> (with GemmKernel as the Operator)
// are compiled by different compilation units.
// This could happend because we seperate combinations of <GemmMeta,GemmHParams>
// into multiple .cu files and combinations in different .cu files yield to the same
// GemmKernel. In such cases only the combinations in one .cu file can run,
// others will fail with "invalid argument error" on kernel launching.
//
// How does this wrapper solve the bug?
// By making sure different combinations of <GemmMeta,GemmHParams> yields to different
// types of cutlass::device_kernel<Operator>, with Operator to be FluxGemmKernel
template <class GemmMetaT, class GemmHParamsT, class GemmKernel>
struct FluxGemmKernel : public GemmKernel {
  using GemmKernel::GemmKernel;
};

// Base type for the space of (meta,hparams) of ops.
// Providing the default implementation for splitting the space.
template <class Derived>
struct OpSpaceBase {
  static constexpr auto
  all_gemm_metas() {
    return Derived::AllGemmMeta;
  }

  static constexpr auto
  all_gemm_hparams() {
    return Derived::AllGemmHParams;
  }

  template <int SplitIdx, int NSplits, int ArchFilter = 0>
  static constexpr auto
  split_slice_meta() {
    return tuple_split_slice<SplitIdx, NSplits>(tuple_filter(all_gemm_metas(), [](auto meta) {
      return ArchFilter == 0 or meta.arch() == ArchEnum(ArchFilter);
    }));
  }

  template <int SplitIdx, int NSplits, int ArchFilter = 0>
  static constexpr auto
  enumerate_split_meta_hparams_pairs() {
    auto meta_split = split_slice_meta<SplitIdx, NSplits, ArchFilter>();
    return tuple_unpack_cat(tuple_transform(meta_split, [](auto meta) {
      return tuple_enumerate(
          make_space_meta_hparams_pair(cute::make_tuple(meta), all_gemm_hparams()));
    }));
  }
};

template <class T>
inline constexpr bool is_flux_op_space_v = std::is_base_of_v<OpSpaceBase<T>, T>;

struct GemmOperatorBase {
 public:
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmOperatorBase)

  virtual ~GemmOperatorBase() = default;
  virtual void run(std::any const &args, void *workspace = nullptr, void *stream = nullptr) = 0;

  // for device allocation required by Gemm::Arguments
  virtual size_t
  get_args_workspace_size(std::any const &args) const {
    return 0;
  }

  virtual void
  initialize_args_workspace(
      std::any const &args, void *args_workspace = nullptr, void *stream = nullptr) const {
    // noop
  }

  // total workspace: args_workspace + gemm workspace
  virtual size_t
  get_workspace_size(std::any const &args) const {
    return 0;
  }

  virtual size_t
  get_barrier_workspace_size(std::any const &args) const {
    return 0;
  }

  virtual UnifiedGemmHParams
  get_runtime_gemm_hparams() const {
    throw std::logic_error("get_runtime_gemm_hparams not implemented");
  }
};

}  // namespace bytedance::flux
