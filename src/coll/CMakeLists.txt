cmake_minimum_required(VERSION 3.17)
set(CU_FILES
	all_gather_impls.cu
	local_copy_and_reset.cu
)

set(LIB_NAME "flux_coll")
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
  flux_add_ths_op_target("flux_coll_op")
endif()
