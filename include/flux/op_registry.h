//===- op_registry.h ---------------------------------------------- C++ ---===//
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
#include <map>
#include <set>
#include <utility>
#include <shared_mutex>
#include <functional>
#include <sstream>
#include <memory>
#include <mutex>
#include "cute/numeric/integral_constant.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/runtime_config.h"

namespace bytedance::flux {

// get arch of current device
ArchEnum get_arch();

class TuningConfigGenerator {
 private:
  static constexpr auto Indent = "  ";
  std::string name;

  std::string head;
  std::string tail;
  std::ostringstream body_ss;

  std::string
  gen_head() const {
    std::ostringstream ss;
    ss << R"(// clang-format off)" << '\n';
    ss << R"(#include "flux/op_registry.h")" << '\n';
    ss << "namespace bytedance::flux {\n";
    ss << "using namespace cute;\n\n";
    ss << "static int " << name << " = []() {\n";
    ss << Indent << "auto &inst = TuningConfigRegistry::instance();\n";
    return std::move(ss).str();
  }

  std::string
  gen_tail() const {
    std::ostringstream ss;
    ss << Indent << "return 0;\n";
    ss << "}();\n";
    ss << "}\n";
    ss << R"(// clang-format on)";
    return std::move(ss).str();
  }

 public:
  TuningConfigGenerator(std::string name)
      : name(std::move(name)), head(gen_head()), tail(gen_tail()) {}

  template <class... Ts, class... Us>
  void
  add(GemmMeta<Ts...> const &meta,
      RuntimeConfig const &rt_conf,
      GemmHParams<Us...> const &best_hparams) {
    // make GemmMeta and GemmHParams to be unified in generated code
    // to reduce template substitution overhead of compiling
    body_ss << Indent << "inst.add(" << to_make_expr(unify_type(meta)) << ","
            << to_make_expr(rt_conf) << "," << to_make_expr(unify_type(best_hparams)) << ");\n";
  }

  std::string
  str() const {
    return head + body_ss.str() + tail;
  }
};

/// Registry for the tuning configs
class TuningConfigRegistry {
 public:
  template <class... Ts, class... Us>
  void
  add(GemmMeta<Ts...> const &meta,
      RuntimeConfig const &rt_conf,
      GemmHParams<Us...> const &best_hparams) {
    std::unique_lock<std::shared_mutex> lock(register_mutex_);
    auto unified_meta = unify_type(meta);
    auto key = std::make_pair(unified_meta, rt_conf);
    auto unified_hparams = unify_type(best_hparams);
    registry_.emplace(std::move(key), std::move(unified_hparams));
  }

  template <class... Ts>
  UnifiedGemmHParams *
  get(GemmMeta<Ts...> const &meta, RuntimeConfig const &rt_conf) {
    std::shared_lock<std::shared_mutex> lock(register_mutex_);
    auto unified_meta = unify_type(meta);
    auto key = std::make_pair(unified_meta, rt_conf);
    auto iter = registry_.find(key);
    return iter == registry_.end() ? nullptr : &(iter->second);
  }

  static TuningConfigRegistry &instance();

 private:
  // [GemmMeta, RuntimeConfig] -> GemmHParams
  std::map<std::pair<UnifiedGemmMeta, RuntimeConfig>, UnifiedGemmHParams> registry_;
  std::shared_mutex register_mutex_;

  TuningConfigRegistry() {}
  TuningConfigRegistry(const TuningConfigRegistry &) = delete;
  TuningConfigRegistry &operator=(const TuningConfigRegistry &) = delete;
};

class OpRegistry {
 public:
  using OpPtr = std::unique_ptr<GemmOperatorBase>;
  using OpCreator = std::function<OpPtr()>;
  using HParamsFilter = std::function<bool(UnifiedGemmHParams const &)>;
  using Dispatcher = std::function<UnifiedGemmHParams(RuntimeConfig const &)>;

  static OpRegistry &instance();

  // Register a hparams for a specific meta, with hparams_idx are the priority, hparams with
  // smaller hparams_idx will be stored before ones with larger hparams_idx
  template <class... Ts, class... Us>
  void
  register_creator(
      OpCreator &&creator, GemmMeta<Ts...> meta, GemmHParams<Us...> hparams, int hparams_idx = 0) {
    static_assert(cute::is_static_v<GemmMeta<Ts...>>, "requires static GemmMeta");
    static_assert(cute::is_static_v<GemmHParams<Us...>>, "requires static GemmHParams");
    static_assert(hparams.is_materialized(), "requires hparams be materialized");
    std::unique_lock<std::shared_mutex> lock(register_mutex_);
    auto unified_meta = unify_type(meta);

    if (op_registry_.find(unified_meta) == op_registry_.end()) {
      op_registry_[unified_meta] = {};
    }
    auto &meta_reg = op_registry_[unified_meta];

    auto unified_hparams = unify_type(hparams);
    if (meta_reg.find(unified_hparams) != meta_reg.end()) {
      // Allow duplicated hparams
      return;
    }
#if defined(FLUX_DEBUG)
    std::cout << "register creator for meta:[" << meta << "], hparams:[" << hparams
              << "] with hparams_idx:[" << hparams_idx << "]" << std::endl;
#endif  // FLUX_DEBUG
    meta_reg.emplace(unified_hparams, std::move(creator));

    // record all hparams registerd for this meta
    if (gemm_hparams_.find(unified_meta) == gemm_hparams_.end()) {
      gemm_hparams_[unified_meta] = {};
    }
    gemm_hparams_[unified_meta].emplace(hparams_idx, std::move(unified_hparams));
  }

  template <class... Ts>
  void
  register_dispatcher(Dispatcher &&dispatcher, GemmMeta<Ts...> meta) {
    std::unique_lock<std::shared_mutex> lock(register_mutex_);
    auto unified_meta = unify_type(meta);
    FLUX_CHECK(dispatcher_registry_.find(unified_meta) == dispatcher_registry_.end())
        << "duplicated registering dispatcher for meta:" << meta;
    dispatcher_registry_.emplace(unified_meta, std::move(dispatcher));
  }

  template <class... Ts>
  UnifiedGemmHParams
  get_hparams(
      GemmMeta<Ts...> meta,
      RuntimeConfig runtime_config = make_runtime_config(),
      HParamsFilter const &filter = [](UnifiedGemmHParams const &) { return true; }) {
    std::shared_lock<std::shared_mutex> lock(register_mutex_);

    auto unified_meta = unify_type(meta);
    // Try get record from tuning config registery
    auto config_record = TuningConfigRegistry::instance().get(unified_meta, runtime_config);
    if (config_record != nullptr) {
      if (filter(*config_record)) {
#if defined(FLUX_DEBUG)
        std::cout << "get record for (" << unified_meta << " x " << runtime_config
                  << ") from config reg: " << *config_record << std::endl;
#endif
        return *config_record;
      }
    }

    // Fallback to user defined dispatcher if no tuned config matched
    auto dispatcher_iter = dispatcher_registry_.find(unified_meta);
    if (dispatcher_iter != dispatcher_registry_.end()) {
      auto const &meta_reg = op_registry_[unified_meta];
      auto hparams = dispatcher_iter->second(runtime_config);
      if (filter(hparams) and (meta_reg.find(hparams) != meta_reg.end())) {
#if defined(FLUX_DEBUG)
        std::cout << "use dispatcher to get hparams for (" << unified_meta << " x "
                  << runtime_config << "), hparams: " << hparams << std::endl;
#endif
        return hparams;
      }
    }

    // Fallback to the first registered hparams if not dispatcher registered
    auto visit_iter = gemm_hparams_.find(unified_meta);

    FLUX_CHECK(visit_iter != gemm_hparams_.end())
        << "No registered hparams found for meta:" << unified_meta;
    const std::set<std::pair<int, UnifiedGemmHParams>> &registered_hparams = visit_iter->second;
    FLUX_CHECK(not registered_hparams.empty()) << "registered hparams empty for meta:" << meta;
    for (auto const &hparams_pair : registered_hparams) {
      auto const &hparams = hparams_pair.second;
      if (filter(hparams)) {
#if defined(FLUX_DEBUG)
        std::cout << "fallback to registered hparams for (" << unified_meta << " x "
                  << runtime_config << "), hparams: " << hparams << std::endl;
#endif
        return hparams;
      }
    }
    FLUX_CHECK(false) << "no registered hparams satisfying filter found for meta:" << meta;
    return registered_hparams.cbegin()->second;  // this will never reach
  }

  template <class... Ts, class... Us>
  OpPtr
  get_op(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
    static_assert(hparams.is_materialized(), "requires hparams be materialized");
    std::shared_lock<std::shared_mutex> lock(register_mutex_);

    auto unified_meta = unify_type(meta);
    auto meta_iter = op_registry_.find(unified_meta);
    FLUX_CHECK(meta_iter != op_registry_.end()) << "unregistered op for meta:" << meta;
    auto &meta_reg = meta_iter->second;
    auto unified_hparams = unify_type(hparams);
    auto hparams_iter = meta_reg.find(unified_hparams);
    FLUX_CHECK(hparams_iter != meta_reg.end())
        << "unregistered op for meta:" << meta << " x hparams: " << hparams;
    return hparams_iter->second();
  }

  template <class... Ts>
  OpPtr
  get_op(
      GemmMeta<Ts...> meta,
      RuntimeConfig runtime_config = make_runtime_config(),
      HParamsFilter const &filter = [](UnifiedGemmHParams const &) { return true; }) {
    auto hparams = get_hparams(meta, runtime_config, filter);
    return get_op(meta, hparams);
  }

  // Iterate all hparams registered for a meta and call func.
  // This can be useful for tuning.
  template <class... Ts>
  void
  visit_hparams(std::function<void(UnifiedGemmHParams)> &&func, GemmMeta<Ts...> meta) {
    std::shared_lock<std::shared_mutex> lock(register_mutex_);
    auto unified_meta = unify_type(meta);
    auto iter = gemm_hparams_.find(unified_meta);
    FLUX_CHECK(iter != gemm_hparams_.end()) << "no op registered for meta:" << meta;
    for (const auto &hparams_pair : iter->second) {
      auto const &hparams = hparams_pair.second;
      func(hparams);
    }
  }

 private:
  // GemmMeta -> [GemmHParams -> Creator]
  std::map<UnifiedGemmMeta, std::map<UnifiedGemmHParams, OpCreator>> op_registry_;
  // user defined dispacher
  std::map<UnifiedGemmMeta, Dispatcher> dispatcher_registry_;
  // for visit_hparams
  std::map<UnifiedGemmMeta, std::set<std::pair<int, UnifiedGemmHParams>>> gemm_hparams_;
  std::shared_mutex register_mutex_;

  OpRegistry() {}
  OpRegistry(const OpRegistry &) = delete;
  OpRegistry &operator=(const OpRegistry &) = delete;
};
}  // namespace bytedance::flux
