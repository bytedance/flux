cmake_minimum_required(VERSION 3.17)

set(OP_REGS "")
flux_add_gemm_op_reg("comm_none/gemm_v2_comm_none.hpp" "GemmV2CommNone" 8 "OP_REGS")
flux_add_gemm_op_reg("comm_none/gemm_v3_comm_none.hpp" "GemmV3CommNone" 8 "OP_REGS")


file(GLOB DISPATCHERS dispatcher/*.cu)
file(GLOB TUNING_CONFIGS tuning_config/*.cu)
set(CU_FILES
  ${OP_REGS}
  ${DISPATCHERS}
  ${TUNING_CONFIGS}
)

set(LIB_NAME "flux_cuda_comm_none")
flux_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
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
