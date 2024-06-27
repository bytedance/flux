//===- op_registry.cu --------------------------------------------- C++ ---===//
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
#include "flux/flux.h"
#include "flux/op_registry.h"
#include <mutex>

namespace bytedance {
namespace flux {

namespace {
std::once_flag init_flag;
ArchEnum arch;

void
init_arch_tag() {
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  int arch_num = major * 10 + minor;
  FLUX_CHECK(arch_num == 80 || arch_num == 89 || arch_num == 90)
      << "unsupported arch: " << arch_num;
  arch = ArchEnum{arch_num};
}
}  // namespace

ArchEnum
get_arch() {
  std::call_once(init_flag, init_arch_tag);
  return arch;
}

TuningConfigRegistry &
TuningConfigRegistry::instance() {
  static TuningConfigRegistry inst;
  char *env = getenv("FLUX_TUNE_CONFIG_FILE");
  if (env != nullptr) {
    static std::once_flag flag;
    std::call_once(flag, load_tune_config_from_file, inst, std::string(env));
  } else {
#if defined(FLUX_DEBUG)
    std::cerr
        << "FLUX_TUNE_CONFIG_FILE not set. no tune config file specified, using default configs\n";
#endif
  }
  return inst;
}

OpRegistry &
OpRegistry::instance() {
  static OpRegistry inst;
  return inst;
}

}  // namespace flux
}  // namespace bytedance
