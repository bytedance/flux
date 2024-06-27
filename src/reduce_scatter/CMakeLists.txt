cmake_minimum_required(VERSION 3.17)

set(OP_REGS "")
flux_add_gemm_op_reg("reduce_scatter/gemm_v2_reduce_scatter.hpp" "GemmV2ReduceScatter" 16 "OP_REGS")
flux_add_gemm_op_reg("reduce_scatter/gemm_v3_reduce_scatter.hpp" "GemmV3ReduceScatter" 16 "OP_REGS")

file(GLOB DISPATCHERS dispatcher/*.cu)
file(GLOB TUNING_CONFIGS tuning_config/*.cu)
set(CU_FILES
  bsr_reduce.cu
  ${OP_REGS}
  ${DISPATCHERS}
  ${TUNING_CONFIGS}
)

set(LIB_NAME "flux_cuda_reduce_scatter")
flux_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
target_link_libraries(${LIB_NAME} PUBLIC nccl)
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS})
list(APPEND FLUX_CUDA_OP_TARGETS ${LIB_NAME})
set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS} PARENT_SCOPE)

if (BUILD_THS)
  file(GLOB THS_FILES ths_op/*.cc)
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES})
  list(APPEND FLUX_THS_OP_FILES ${THS_FILES})
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES} PARENT_SCOPE)
endif()

if (BUILD_TEST)
  add_subdirectory(test)
endif()
