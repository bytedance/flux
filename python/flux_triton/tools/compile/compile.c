/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <cuda.h>

#include "triton_aot_runtime.h"

// helpers to check for cuda errors
#define CUDA_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert(CUresult code, const char *file, int line) {{
  if (code != CUDA_SUCCESS) {{
    const char *prefix = "Triton Error [CUDA]: ";
    const char *str;
    cuGetErrorString_stub(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

// globals
#define CUBIN_NAME {kernel_name}_cubin
CUDAModuleHandle {kernel_name}_mod = NULL;
CUDAFunctionHandle {kernel_name}_func = NULL;
unsigned char CUBIN_NAME[{bin_size}] = {{ {bin_data} }};


void unload_{kernel_name}(void) {{
    CUDA_CHECK(CUDAModuleUnload({kernel_name}_mod));
}}

// TODO: some code duplication with `runtime/backend/cuda.c`
void load_{kernel_name}() {{
    int dev = 0;
    void *bin = (void *)&CUBIN_NAME;
    int shared = {shared};
    CUDA_CHECK(CUDAModuleLoadData(&{kernel_name}_mod, bin));
    CUDA_CHECK(CUDAModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{triton_kernel_name}"));
    // set dynamic shared memory if necessary
    int shared_optin;
    CUDA_CHECK(cuDeviceGetAttribute_stub(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev));
    if (shared > 49152 && shared_optin > 49152) {{
      CUDA_CHECK(CUDAFuncSetCacheConfig({kernel_name}_func, CU_FUNC_CACHE_PREFER_SHARED));
      CUDA_CHECK(CUDAFuncSetAttribute({kernel_name}_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin))
    }}
}}

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    void *args[{num_args}] = {{ {arg_pointers} }};
    // TODO: shared memory
    if(gX * gY * gZ > 0)
      return CUDALaunchKernel({kernel_name}_func, gX, gY, gZ, {num_warps} * 32, 1, 1, {shared}, stream, args, NULL);

    fprintf(stderr, "invalid grid size: %d, %d, %d\n", gX, gY, gZ);
    return CUDA_ERROR_INVALID_VALUE;
}}
