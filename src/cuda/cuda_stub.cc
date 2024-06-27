
//===- cuda_stub.cc --------------------------------------------- C++ ---===//
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
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"

#include <dlfcn.h>
#include <string>

namespace bytedance::flux {
namespace _stubs {
#define _STUB_1(LIB, NAME, RETTYPE, ARG1)                                              \
  RETTYPE NAME(ARG1 a1) {                                                              \
    auto fn = reinterpret_cast<decltype(&NAME)>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                     \
    cuda_stub_.NAME = fn;                                                              \
    return fn(a1);                                                                     \
  }

#define _STUB_2(LIB, NAME, RETTYPE, ARG1, ARG2)                                        \
  RETTYPE NAME(ARG1 a1, ARG2 a2) {                                                     \
    auto fn = reinterpret_cast<decltype(&NAME)>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                     \
    cuda_stub_.NAME = fn;                                                              \
    return fn(a1, a2);                                                                 \
  }

#define _STUB_3(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3)                                  \
  RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3) {                                            \
    auto fn = reinterpret_cast<decltype(&NAME)>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                     \
    cuda_stub_.NAME = fn;                                                              \
    return fn(a1, a2, a3);                                                             \
  }

#define _STUB_4(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3, ARG4)                            \
  RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3, ARG4 a4) {                                   \
    auto fn = reinterpret_cast<decltype(&NAME)>(Get##LIB##Library().Symbol(__func__)); \
    FLUX_CHECK(fn) << "Can't get " << #LIB << " symbol " << #NAME;                     \
    cuda_stub_.NAME = fn;                                                              \
    return fn(a1, a2, a3, a4);                                                         \
  }

#define CUDA_STUB1(NAME, A1) _STUB_1(CUDA, NAME, CUresult, A1)
#define CUDA_STUB2(NAME, A1, A2) _STUB_2(CUDA, NAME, CUresult, A1, A2)
#define CUDA_STUB3(NAME, A1, A2, A3) _STUB_3(CUDA, NAME, CUresult, A1, A2, A3)
#define CUDA_STUB4(NAME, A1, A2, A3, A4) _STUB_4(CUDA, NAME, CUresult, A1, A2, A3, A4)

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

  static int cudaDriverVersion;
  CUDA_CHECK(cudaDriverGetVersion(&cudaDriverVersion));
  FLUX_CHECK(cudaDriverVersion >= 11070) << "flux requires cuda driver version>=11.7";
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
GetCUDALibrary() {
  static DynamicLibrary lib("libcuda.so.1");
  return lib;
}

extern CUDA cuda_stub_;
CUDA_STUB2(cuGetErrorName, CUresult, const char **)
CUDA_STUB2(cuGetErrorString, CUresult, const char **);
CUDA_STUB3(cuDeviceGetName, char *, int, CUdevice);
CUDA_STUB4(cuStreamWaitValue32_v2, CUstream, CUdeviceptr, cuuint32_t, unsigned int);
CUDA_STUB4(cuStreamWriteValue32_v2, CUstream, CUdeviceptr, cuuint32_t, unsigned int);
CUDA_STUB4(cuStreamWaitValue64_v2, CUstream, CUdeviceptr, cuuint64_t, unsigned int);
CUDA_STUB4(cuStreamWriteValue64_v2, CUstream, CUdeviceptr, cuuint64_t, unsigned int);
CUDA_STUB4(
    cuStreamBatchMemOp_v2, CUstream, unsigned int, CUstreamBatchMemOpParams *, unsigned int);
CUDA_STUB1(cuCtxGetDevice, CUdevice *);

CUDA cuda_stub_ = {
#define _REFERENCE_MEMBER(name) _stubs::name,
    FLUX_FORALL_CUDA(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
};
}  // namespace _stubs

CUDA &
cuda_stub() {
  return _stubs::cuda_stub_;
}

}  // namespace bytedance::flux
