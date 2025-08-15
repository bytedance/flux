cmake_minimum_required(VERSION 3.17)
set(CU_FILES
	all_gather_impls.cu
	local_copy_and_reset.cu
)
if(ENABLE_NVSHMEM)
  list(APPEND CU_FILES
  dis_scatter_forward_impl.cu
  dis_scatter_backward_impl.cu
  all2all_impl.cu
  all2all_single_2d_impl.cu
  )
endif()

set(LIB_NAME "flux_coll")
flux_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)

set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS})
list(APPEND FLUX_CUDA_OP_TARGETS ${LIB_NAME})
set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS} PARENT_SCOPE)
if (BUILD_THS)
  file(GLOB THS_FILES ths_op/*.cc)
  file(GLOB ALL2ALL_THS_FILES ths_op/dis_scatter*.cc ths_op/all2all_op.cc ths_op/all2all_single_2d.cc)
  list(REMOVE_ITEM THS_FILES ${ALL2ALL_THS_FILES})
  if(ENABLE_NVSHMEM)
    list(APPEND THS_FILES ${ALL2ALL_THS_FILES})
  endif()
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES})
  list(APPEND FLUX_THS_OP_FILES ${THS_FILES})
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES} PARENT_SCOPE)
  flux_add_ths_op_target("flux_coll_op")
endif()
