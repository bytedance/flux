
//===- nvml_stub.cc --------------------------------------------- C++ ---===//
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
#include "flux/cuda/nvml_stub.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include <dlfcn.h>
#include <string>

namespace bytedance::flux {
namespace _stubs {
#define _STUB_0(LIB, NAME, RETTYPE)                                                      \
  RETTYPE NAME() {                                                                       \
    auto fn = reinterpret_cast<decltype(&(NAME))>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                       \
    nvml_stub_.NAME = fn;                                                                \
    return fn();                                                                         \
  }

#define _STUB_1(LIB, NAME, RETTYPE, ARG1)                                                \
  RETTYPE NAME(ARG1 a1) {                                                                \
    auto fn = reinterpret_cast<decltype(&(NAME))>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                       \
    nvml_stub_.NAME = fn;                                                                \
    return fn(a1);                                                                       \
  }

#define _STUB_2(LIB, NAME, RETTYPE, ARG1, ARG2)                                          \
  RETTYPE NAME(ARG1 a1, ARG2 a2) {                                                       \
    auto fn = reinterpret_cast<decltype(&(NAME))>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                       \
    nvml_stub_.NAME = fn;                                                                \
    return fn(a1, a2);                                                                   \
  }

#define _STUB_3(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3)                                    \
  RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3) {                                              \
    auto fn = reinterpret_cast<decltype(&(NAME))>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                       \
    nvml_stub_.NAME = fn;                                                                \
    return fn(a1, a2, a3);                                                               \
  }

#define _STUB_4(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3, ARG4)                              \
  RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3, ARG4 a4) {                                     \
    auto fn = reinterpret_cast<decltype(&(NAME))>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                       \
    nvml_stub_.NAME = fn;                                                                \
    return fn(a1, a2, a3, a4);                                                           \
  }

#define NVML_STUB0(NAME) _STUB_0(NVML, NAME, nvmlReturn_t)
#define NVML_STUB1(NAME, A1) _STUB_1(NVML, NAME, nvmlReturn_t, A1)
#define NVML_STUB2(NAME, A1, A2) _STUB_2(NVML, NAME, nvmlReturn_t, A1, A2)
#define NVML_STUB3(NAME, A1, A2, A3) _STUB_3(NVML, NAME, nvmlReturn_t, A1, A2, A3)
#define NVML_STUB4(NAME, A1, A2, A3, A4) _STUB_4(NVML, NAME, nvmlReturn_t, A1, A2, A3, A4)
namespace {
struct DynamicLibrary {
  DynamicLibrary(const char *name);
  void *Symbol(const char *name);
  ~DynamicLibrary();
  DynamicLibrary(const DynamicLibrary &) = delete;
  void operator=(const DynamicLibrary &) = delete;

 private:
  void *handle_ = nullptr;
};

DynamicLibrary::DynamicLibrary(const char *name) {
  handle_ = dlopen(name, RTLD_LOCAL | RTLD_NOW);
  FLUX_CHECK(handle_ != nullptr) << "Error in dlopen(" << name << "): " << dlerror();
}

void *
DynamicLibrary::Symbol(const char *name) {
  if (handle_ == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle_, name);
  FLUX_CHECK(func != nullptr) << "Error in dlsym(" << name << "): " << dlerror();
  return func;
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle_)
    return;
  dlclose(handle_);
}

DynamicLibrary &
GetNVMLLibrary() {
  static DynamicLibrary lib("libnvidia-ml.so.1");
  return lib;
}
}  // namespace

extern NVML nvml_stub_;

NVML_STUB1(nvmlDeviceGetCount, unsigned int *);
NVML_STUB3(nvmlDeviceGetCudaComputeCapability, nvmlDevice_t, int *, int *);
NVML_STUB3(
    nvmlDeviceGetNvLinkRemoteDeviceType, nvmlDevice_t, unsigned int, nvmlIntNvLinkDeviceType_t *);
NVML_STUB3(nvmlDeviceGetFieldValues, nvmlDevice_t, int, nvmlFieldValue_t *);
NVML_STUB2(nvmlDeviceGetHandleByIndex, unsigned int, nvmlDevice_t *);
NVML_STUB2(nvmlDeviceGetHandleByPciBusId, const char *, nvmlDevice_t *);
NVML_STUB2(nvmlDeviceGetIndex, nvmlDevice_t, unsigned int *);
NVML_STUB2(nvmlDeviceGetMaxPcieLinkGeneration, nvmlDevice_t, unsigned int *);
NVML_STUB3(nvmlDeviceGetName, nvmlDevice_t, char *, unsigned int);
NVML_STUB3(nvmlDeviceGetNvLinkState, nvmlDevice_t, unsigned int, nvmlEnableState_t *);
NVML_STUB3(nvmlDeviceGetNvLinkRemotePciInfo, nvmlDevice_t, unsigned int, nvmlPciInfo_t *);
NVML_STUB3(nvmlDeviceGetNvLinkVersion, nvmlDevice_t, unsigned int, unsigned int *);
NVML_STUB4(
    nvmlDeviceGetNvLinkCapability,
    nvmlDevice_t,
    unsigned int,
    nvmlNvLinkCapability_t,
    unsigned int *);
NVML_STUB4(
    nvmlDeviceGetP2PStatus,
    nvmlDevice_t,
    nvmlDevice_t,
    nvmlGpuP2PCapsIndex_t,
    nvmlGpuP2PStatus_t *);
_STUB_1(NVML, nvmlErrorString, const char *, nvmlReturn_t);
NVML_STUB0(nvmlInit);
NVML_STUB0(nvmlShutdown);

NVML nvml_stub_ = {
#define _REFERENCE_MEMBER(name) _stubs::name,
    FLUX_FORALL_NVML(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
};
}  // namespace _stubs

NVML &
nvml_stub() {
  return _stubs::nvml_stub_;
}

}  // namespace bytedance::flux
