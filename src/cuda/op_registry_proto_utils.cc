//===- op_registry_proto_utils.cc -------------------------------- C++ ---===//
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
#include "flux/op_registry_proto_utils.h"
#include <fcntl.h>
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/runtime_config.h"
#if defined(WITH_PROTOBUF)
#include "proto/flux.pb.h"
#include "flux/utils.h"
#include <unistd.h>
#include "cute/container/tuple.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace bytedance::flux {
namespace {
bool
load_proto_from_file(const std::string &filename, google::protobuf::Message *proto) {
  using google::protobuf::io::FileInputStream;
  int fd = open(filename.c_str(), O_RDONLY);
  FLUX_CHECK(fd != -1) << "prototxt file not found: " << filename;
  auto *input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}
int
load_from_file(const std::string &filename, std::string &content) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "open file " << filename << " failed" << std::endl;
    return -1;
  }
  content.resize(fin.seekg(0, std::ios::end).tellg());
  fin.seekg(0, std::ios::beg).read(&content[0], content.size());
  fin.close();
  return 0;
}
}  // namespace

auto
ToUnifiedGemmMata(const proto::GemmMeta &config) {
  // NO default value settings now.
  FLUX_CHECK(config.has_dtype());
  const auto &dtype = config.dtype();
  auto impl_spec = [&config]() -> UnifiedImplMeta {
    if (config.has_gemm_v3_meta()) {
      return make_gemm_v3_meta(config.gemm_v3_meta().fast_accum());
    }
    return None{};
  }();

  auto comm_spec = [&config]() -> UnifiedCommMeta {
    if (config.has_reduce_scatter_meta()) {
      return make_reduce_scatter_meta(
          config.reduce_scatter_meta().fuse_reduction(),
          (CommKindEnum)config.reduce_scatter_meta().comm_kind());
    } else if (config.has_gather_rs_meta()) {
      return make_gather_rs_meta((int)config.gather_rs_meta().topk());
    }
    return None{};
  }();
  return make_gemm_meta(
      make_gemm_dtype_config(
          (DataTypeEnum)dtype.a(),
          (DataTypeEnum)dtype.b(),
          (DataTypeEnum)dtype.c(),
          (DataTypeEnum)dtype.d(),
          (DataTypeEnum)dtype.acc()),
      (ArchEnum)config.arch(),
      (CommOpEnum)config.comm_op(),
      (GemmLayoutEnum)config.gemm_layout(),
      (ImplEnum)config.impl(),
      std::move(impl_spec),
      std::move(comm_spec));
}

auto
ToRuntimeConfig(const proto::RuntimeConfig &config) {
  UnifiedCommRuntimeConfig comm_spec = None{};
  if (config.has_all_gather_rt_conf()) {
    auto const &ag_rt_conf = config.all_gather_rt_conf();
    comm_spec = make_all_gather_runtime_config(
        (int)ag_rt_conf.world_size(), (int)ag_rt_conf.nnodes(), (int)ag_rt_conf.ring_mode());
  } else if (config.has_reduce_scatter_rt_conf()) {
    auto const &rs_rt_conf = config.reduce_scatter_rt_conf();
    comm_spec =
        make_reduce_scatter_runtime_config((int)rs_rt_conf.world_size(), (int)rs_rt_conf.nnodes());
  }

  return make_runtime_config(config.m(), config.n(), config.k(), std::move(comm_spec));
}

auto
ToUnifiedGemmHparams(const proto::GemmHparams &hparams) {
  auto impl_spec = [&]() -> UnifiedImplHParams {
    if (hparams.has_gemm_v2_hparams()) {
      const auto &gemm_v2_hparams = hparams.gemm_v2_hparams();
      FLUX_CHECK(gemm_v2_hparams.warp_shape_size() == 3);
      FLUX_CHECK(gemm_v2_hparams.instruction_shape_size() == 3);
      return make_gemm_v2_hparams(
          cute::make_tuple(
              (long)gemm_v2_hparams.warp_shape(0),
              (long)gemm_v2_hparams.warp_shape(1),
              (long)gemm_v2_hparams.warp_shape(2)),
          cute::make_tuple(
              gemm_v2_hparams.instruction_shape(0),
              gemm_v2_hparams.instruction_shape(1),
              gemm_v2_hparams.instruction_shape(2)),
          (GemmStreamkModeEnum)gemm_v2_hparams.streamk_mode());
    }
    if (hparams.has_gemm_v3_hparams()) {
      const auto &gemm_v3_hparams = hparams.gemm_v3_hparams();
      FLUX_CHECK(gemm_v3_hparams.cluster_shape_size() == 3);
      return make_gemm_v3_hparams(
          cute::make_tuple(
              (long)gemm_v3_hparams.cluster_shape(0),
              (long)gemm_v3_hparams.cluster_shape(1),
              (long)gemm_v3_hparams.cluster_shape(2)),
          GemmKernelScheduleEnum(gemm_v3_hparams.kernel_schedule()));
    }
    FLUX_CHECK(false) << " unsupported impl_spec. oneof (GemmV2HParams, GemmV3HParams)";
    return None{};
  }();
  auto comm_spec = [&]() -> UnifiedCommHParams {
    if (hparams.has_gather_rs_hparams()) {
      const auto &gather_rs_hparams = hparams.gather_rs_hparams();
      return make_gather_rs_hparams(
          gather_rs_hparams.gather_rs_ctas(), gather_rs_hparams.n_dim_per_split());
    }
    return None{};
  }();

  FLUX_CHECK(hparams.tile_shape_size() == 3);
  return make_gemm_hparams(
      impl_spec,
      comm_spec,
      cute::make_tuple(hparams.tile_shape(0), hparams.tile_shape(1), hparams.tile_shape(2)),
      (GemmKindEnum)hparams.gemm_kind(),
      hparams.mainloop_stage(),
      (GemmRasterOrderEnum)hparams.raster_order());
}

void
load_tune_config_from_file(TuningConfigRegistry &registry, const std::string &file_name) {
  if (!std::filesystem::exists(file_name)) {
    std::cerr << "FLUX_TUNE_CONFIG_FILE " << file_name << " not exist, skip\n";
    return;
  }

  proto::TuneConfigList proto;
  if (load_proto_from_file(file_name, &proto) == false) {
    std::string content;
    if (load_from_file(file_name, content) != 0) {
      std::cerr << "open FLUX_TUNE_CONFIG_FILE error: " << file_name << "\n";
      return;
    }
    std::cerr << "parse FLUX_TUNE_CONFIG_FILE " << file_name << " failed, skip:\n " << content
              << "\n";
    return;
  }
  for (const auto &config : proto.tune_configs()) {
    FLUX_CHECK(config.has_meta());
    FLUX_CHECK(config.has_rt_conf());
    FLUX_CHECK(config.has_best_hparams());
    auto meta = ToUnifiedGemmMata(config.meta());
    auto rt_conf = ToRuntimeConfig(config.rt_conf());
    auto hparams = ToUnifiedGemmHparams(config.best_hparams());
    auto *hparam_old = registry.get(meta, rt_conf);
#if defined(FLUX_DEBUG)
    if (get_int_from_env("RANK", 0) == 0) {
      if (hparam_old != nullptr) {
        std::cerr << "[warning] (" << meta << " x " << rt_conf << ") already exists."
                  << *hparam_old << " overwrite hparams " << *hparam_old << " with " << hparams
                  << "\n";
      } else {
        std::cout << "add new hparams: (" << meta << " x " << rt_conf << ") -> " << hparams
                  << "\n";
      }
    }
#endif
    registry.add(meta, rt_conf, hparams);
  }
}
}  // namespace bytedance::flux

#else
namespace bytedance::flux {
void
load_tune_config_from_file(TuningConfigRegistry &registry, const std::string &file_name) {
  std::cerr << "add tune config at runtime is not supported, please recompile flux with protobuf "
               "support and try again\n";
}
}  // namespace bytedance::flux

#endif
