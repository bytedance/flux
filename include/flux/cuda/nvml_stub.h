//===- nvml.h ----------------------------------------------- C++ ---===//
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

/*!
 * \file nvml.h
 * \brief CUDA stub to avoid direct CUDA driver call
 */
#pragma once

#include <nvml.h>

namespace bytedance::flux {

#define FLUX_FORALL_NVML(_)              \
  _(nvmlDeviceGetCount)                  \
  _(nvmlDeviceGetCudaComputeCapability)  \
  _(nvmlDeviceGetNvLinkRemoteDeviceType) \
  _(nvmlDeviceGetFieldValues)            \
  _(nvmlDeviceGetHandleByIndex)          \
  _(nvmlDeviceGetHandleByPciBusId)       \
  _(nvmlDeviceGetIndex)                  \
  _(nvmlDeviceGetMaxPcieLinkGeneration)  \
  _(nvmlDeviceGetName)                   \
  _(nvmlDeviceGetNvLinkCapability)       \
  _(nvmlDeviceGetNvLinkRemotePciInfo)    \
  _(nvmlDeviceGetNvLinkState)            \
  _(nvmlDeviceGetNvLinkVersion)          \
  _(nvmlDeviceGetP2PStatus)              \
  _(nvmlErrorString)                     \
  _(nvmlInit)                            \
  _(nvmlShutdown)

extern "C" {
typedef struct NVML {
#define CREATE_MEMBER(name) decltype(&(name)) name;
  FLUX_FORALL_NVML(CREATE_MEMBER)
#undef CREATE_MEMBER
} NVML;
}

NVML &nvml_stub();

#define NVML_CHECK(expr)                                                                        \
  do {                                                                                          \
    nvmlReturn_t rtn = expr;                                                                    \
    FLUX_CHECK(rtn == NVML_SUCCESS)                                                             \
        << "Got bad nvml status: " << nvml_stub().nvmlErrorString(rtn) << "(" << rtn << ") at " \
        << #expr << "\n";                                                                       \
  } while (0)

}  // namespace bytedance::flux
