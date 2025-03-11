cmake_minimum_required(VERSION 3.17)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/registers")
execute_process(
  COMMAND ${FLUX_GENERATOR_BINARY_DIR}/gen_gemm_rs "--dir=./registers" "--archs=${CUDAARCHS}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND_ECHO STDOUT
)

file(GLOB OP_REGS ${CMAKE_CURRENT_BINARY_DIR}/registers/*.cu)
file(GLOB TUNING_CONFIGS tuning_config/*.cu)
set(CU_FILES
  bsr_reduce.cu
  padding_util.cu
  ring_reduce.cu
  ${OP_REGS}
  ${TUNING_CONFIGS}
)

set(LIB_NAME "flux_cuda_reduce_scatter")
flux_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS})
list(APPEND FLUX_CUDA_OP_TARGETS ${LIB_NAME})
set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS} PARENT_SCOPE)

if (BUILD_THS)
  file(GLOB THS_FILES ths_op/*.cc)
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES})
  list(APPEND FLUX_THS_OP_FILES ${THS_FILES})
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES} PARENT_SCOPE)
  flux_add_ths_op_target("gemm_rs")
endif()

if (BUILD_TEST)
  add_subdirectory(test)
endif()
