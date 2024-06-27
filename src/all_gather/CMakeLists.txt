cmake_minimum_required(VERSION 3.17)

set(OP_REGS "")
flux_add_gemm_op_reg("all_gather/gemm_v2_ag_kernel.hpp" "GemmV2AGKernel" 8 "OP_REGS")
flux_add_gemm_op_reg("all_gather/gemm_v3_ag_kernel.hpp" "GemmV3AGKernel" 16 "OP_REGS")

file(GLOB DISPATCHERS dispatcher/*.cu)
file(GLOB TUNING_CONFIGS tuning_config/*.cu)
set(CU_FILES
  ${OP_REGS}
  ${DISPATCHERS}
  ${TUNING_CONFIGS}
)

set(LIB_NAME "flux_cuda_all_gather")
flux_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS})
list(APPEND FLUX_CUDA_OP_TARGETS ${LIB_NAME})
set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS} PARENT_SCOPE)

if (BUILD_THS)
  if(ENABLE_NVSHMEM)
    # across node ops depend on NVSHMEM
    file(GLOB THS_FILES ths_op/*.cc)
  else()
    file(GLOB THS_FILES ths_op/all_gather_gemm_kernel.cc)
  endif()
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES})
  list(APPEND FLUX_THS_OP_FILES ${THS_FILES})
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES} PARENT_SCOPE)
endif()
