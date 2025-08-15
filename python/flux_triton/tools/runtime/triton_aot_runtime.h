//===- triton_aot_runtime.h ------------------------------------- C++ ------===//
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
#include <cuda.h>
// CUDA 12.0+ has CUDA context independent module loading. but what about CUDA 11.8
// https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
#ifdef __cplusplus
extern "C" {
#endif

// CUDA driver stubs to avoid direct dependency on libcuda.so
CUresult cuGetErrorString_stub(CUresult error, const char **pStr);
CUresult cuDeviceGetAttribute_stub(int *pi, CUdevice_attribute attrib, CUdevice dev);

// CUDA patch for Multiple CUDA context support: using any CUDA context
typedef struct CUDAModule *CUDAModuleHandle;
typedef struct CUDAFunction *CUDAFunctionHandle;

CUresult CUDAModuleLoadData(CUDAModuleHandle *module, const void *image);

CUresult CUDAModuleUnload(CUDAModuleHandle module);

CUresult CUDAModuleGetFunction(CUDAFunctionHandle *hfunc, CUDAModuleHandle hmod, const char *name);

CUresult CUDALaunchKernel(
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
    void **extra);

CUresult CUDAFuncSetAttribute(CUDAFunctionHandle func, CUfunction_attribute attrib, int value);

CUresult CUDAFuncSetCacheConfig(CUDAFunctionHandle func, CUfunc_cache config);

#ifdef __cplusplus
}
#endif
