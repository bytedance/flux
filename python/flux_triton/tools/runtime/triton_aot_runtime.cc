//===- triton_aot_runtime.cc ------------------------------------ C++ ------===//
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
#include <cuda.h>
#include <dlfcn.h>
#include <cstring>
#include <mutex>
#include <map>

#ifdef __cplusplus
#define FLUX_TRITON_EXTERN extern "C" __attribute__((visibility("hidden")))
#else
#define FLUX_TRITON_EXTERN
#endif

#define CHECK(rtn)                                  \
  do {                                              \
    bool rtn_ = (rtn);                              \
    if (!rtn_) {                                    \
      fprintf(                                      \
          stderr,                                   \
          "CHECK error at %s:%d, errno: %d (%s)\n", \
          __FILE__,                                 \
          __LINE__,                                 \
          errno,                                    \
          strerror(errno));                         \
      exit(rtn);                                    \
    }                                               \
  } while (0)

#define CHECK_RTN_RETURN(rtn)                                                \
  do {                                                                       \
    auto rtn_ = (rtn);                                                       \
    if (rtn_ != CUDA_SUCCESS) {                                              \
      fprintf(stderr, "CUDA error %d at %s:%d\n", rtn_, __FILE__, __LINE__); \
      return rtn;                                                            \
    }                                                                        \
  } while (0)

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
  CHECK(handle_ != nullptr);
}

void *
DynamicLibrary::Symbol(const char *name) {
  if (handle_ == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle_, name);
  CHECK(func != nullptr);
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
FLUX_TRITON_EXTERN
CUresult
cuLaunchKernel_stub(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra) {
  using FuncType = decltype(cuLaunchKernel);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuLaunchKernel");
  CHECK(func != nullptr);
  return func(
      f,
      gridDimX,
      gridDimY,
      gridDimZ,
      blockDimX,
      blockDimY,
      blockDimZ,
      sharedMemBytes,
      hStream,
      kernelParams,
      extra);
}

FLUX_TRITON_EXTERN CUresult
cuCtxGetCurrent_stub(CUcontext *pctx) {
  using FuncType = decltype(cuCtxGetCurrent);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuCtxGetCurrent");
  CHECK(func != nullptr);
  return func(pctx);
}

FLUX_TRITON_EXTERN CUresult
cuDeviceGetAttribute_stub(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  using FuncType = decltype(cuDeviceGetAttribute);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuDeviceGetAttribute");
  CHECK(func != nullptr);
  return func(pi, attrib, dev);
}

FLUX_TRITON_EXTERN CUresult
cuGetErrorString_stub(CUresult error, const char **pStr) {
  using FuncType = decltype(cuGetErrorString);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuGetErrorString");
  CHECK(func != nullptr);
  return func(error, pStr);
}

FLUX_TRITON_EXTERN CUresult
cuModuleUnload_stub(CUmodule hmod) {
  using FuncType = decltype(cuModuleUnload);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuModuleUnload");
  CHECK(func != nullptr);
  return func(hmod);
}

FLUX_TRITON_EXTERN CUresult
cuModuleLoadData_stub(CUmodule *module, const void *image) {
  using FuncType = decltype(cuModuleLoadData);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuModuleLoadData");
  CHECK(func != nullptr);
  return func(module, image);
}

FLUX_TRITON_EXTERN CUresult
cuModuleGetFunction_stub(CUfunction *hfunc, CUmodule hmod, const char *name) {
  using FuncType = decltype(cuModuleGetFunction);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuModuleGetFunction");
  CHECK(func != nullptr);
  return func(hfunc, hmod, name);
}

FLUX_TRITON_EXTERN CUresult
cuFuncSetAttribute_stub(CUfunction hfunc, CUfunction_attribute attrib, int value) {
  using FuncType = decltype(cuFuncSetAttribute);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuFuncSetAttribute");
  CHECK(func != nullptr);
  return func(hfunc, attrib, value);
}

FLUX_TRITON_EXTERN CUresult
cuFuncSetCacheConfig_stub(CUfunction hfunc, CUfunc_cache config) {
  using FuncType = decltype(cuFuncSetCacheConfig);
  FuncType *func = (FuncType *)GetCUDALibrary().Symbol("cuFuncSetCacheConfig");
  CHECK(func != nullptr);
  return func(hfunc, config);
}

class CUDAModule {
 public:
  CUDAModule(const void *image) : image_(image) {}
  CUresult
  Load(CUcontext context) {
    std::lock_guard<std::mutex> _(mutex_);
    return LoadImpl(context);
  }

  CUresult
  Unload(CUcontext context) {
    std::lock_guard<std::mutex> _(mutex_);
    if (module_.find(context) != module_.end()) {
      return CUDA_ERROR_UNKNOWN;
    }
    auto rtn = cuModuleUnload_stub(module_[context]);
    module_.erase(context);
    return rtn;
  }

  CUresult
  GetOrLoad(CUcontext context, CUmodule &module) {
    std::lock_guard<std::mutex> _(mutex_);
    auto iter = module_.find(context);
    if (iter == module_.end()) {  // Get from another CUcontext
      CHECK_RTN_RETURN(LoadImpl(context));
      module = module_[context];
    } else {
      module = iter->second;
    }
    return CUDA_SUCCESS;
  }

 private:
  CUresult
  LoadImpl(CUcontext context) {
    auto iter = module_.find(context);
    if (iter != module_.end()) {
      return CUDA_ERROR_UNKNOWN;
    }
    return cuModuleLoadData_stub(&module_[context], image_);
  }

 private:
  const void *image_;
  std::mutex mutex_;
  std::map<const CUcontext, CUmodule> module_;
};

class CUDAFunction {
 public:
  CUDAFunction(CUDAModule *module, const char *name) : module_(module), name_(name) {}
  CUresult
  Load(CUcontext context) {
    std::lock_guard<std::mutex> _(mutex_);
    return LoadImpl(context);
  }

  CUresult
  GetOrLoad(CUcontext context, CUfunction *function) {
    auto iter = func_.find(context);
    if (iter == func_.end()) {  // Get from another CUcontext
      CHECK_RTN_RETURN(LoadImpl(context));
      *function = func_[context];
    } else {
      *function = iter->second;
    }
    return CUDA_SUCCESS;
  }

 private:
  CUresult
  LoadImpl(CUcontext context) {
    CUmodule mod;
    CHECK_RTN_RETURN(module_->GetOrLoad(context, mod));
    auto iter = func_.find(context);
    if (iter != func_.end()) {  // duplicate load
      return CUDA_ERROR_UNKNOWN;
    }
    return cuModuleGetFunction_stub(&func_[context], mod, name_);
  }

 private:
  CUDAModule *module_ = nullptr;
  const char *name_ = nullptr;
  std::mutex mutex_;
  std::map<CUcontext, CUfunction> func_;
};

using CUDAModuleHandle = CUDAModule *;
using CUDAFunctionHandle = CUDAFunction *;

FLUX_TRITON_EXTERN CUresult
CUDAModuleLoadData(CUDAModuleHandle *module, const void *image) {
  CUDAModule *mod = new CUDAModule(image);
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent_stub(&context));
  CHECK_RTN_RETURN(mod->Load(context));
  *module = mod;
  return CUDA_SUCCESS;
}

FLUX_TRITON_EXTERN CUresult
CUDAModuleGetFunction(CUDAFunctionHandle *hfunc, CUDAModuleHandle hmod, const char *name) {
  CUDAFunction *func = new CUDAFunction(hmod, name);
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent_stub(&context));
  CHECK_RTN_RETURN(func->Load(context));
  *hfunc = func;
  return CUDA_SUCCESS;
}

FLUX_TRITON_EXTERN CUresult
CUDAModuleUnload(CUDAModuleHandle hmod) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent_stub(&context));
  CHECK_RTN_RETURN(hmod->Unload(context));
  return CUDA_SUCCESS;
}

FLUX_TRITON_EXTERN CUresult
CUDAFuncSetAttribute(CUDAFunctionHandle func, CUfunction_attribute attrib, int value) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent_stub(&context));
  CUfunction f;
  CHECK_RTN_RETURN(func->GetOrLoad(context, &f));
  return cuFuncSetAttribute_stub(f, attrib, value);
}

FLUX_TRITON_EXTERN CUresult
CUDAFuncSetCacheConfig(CUDAFunctionHandle func, CUfunc_cache config) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent_stub(&context));
  CUfunction f;
  CHECK_RTN_RETURN(func->GetOrLoad(context, &f));
  return cuFuncSetCacheConfig_stub(f, config);
}

FLUX_TRITON_EXTERN CUresult
CUDALaunchKernel(
    CUDAFunctionHandle f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra) {
  CUcontext context;
  CHECK_RTN_RETURN(cuCtxGetCurrent_stub(&context));
  CUfunction func;
  CHECK_RTN_RETURN(f->GetOrLoad(context, &func));
  return cuLaunchKernel_stub(
      func,
      gridDimX,
      gridDimY,
      gridDimZ,
      blockDimX,
      blockDimY,
      blockDimZ,
      sharedMemBytes,
      hStream,
      kernelParams,
      extra);
}
