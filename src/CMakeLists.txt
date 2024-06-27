cmake_minimum_required(VERSION 3.17)

set(GEMM_OP_REG_INP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gemm_op_registry.cu.in)

# append 'flux/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Generated op reg files and append to `GENERATED` file list
function(flux_add_gemm_op_reg IMPL_HEADER IMPL_NAME NSPLITS GENERATED)
  set(GENERATED_REG_DIR "${CMAKE_CURRENT_BINARY_DIR}")

  set(GENERATED_FILES "${${GENERATED}}")
  math(EXPR max_value "${NSPLITS} - 1")
  foreach(SPLIT_IDX RANGE 0 ${max_value})
    foreach(ARCH ${CUDAARCHS})
      set(OUTPUT_FILE "${GENERATED_REG_DIR}/${IMPL_NAME}_sm${ARCH}_${SPLIT_IDX}_${NSPLITS}.cu")
      configure_file(${GEMM_OP_REG_INP_FILE} ${OUTPUT_FILE} @ONLY)
      list(APPEND GENERATED_FILES ${OUTPUT_FILE})
    endforeach()
  endforeach()
  set(${GENERATED} ${GENERATED_FILES} PARENT_SCOPE)
endfunction()

# create an object library for each op subfolder
function(flux_add_op_cu_obj_lib TARGET FILES)
  add_library(${TARGET} OBJECT ${FILES})
  set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart ${CMAKE_THREAD_LIBS_INIT} )
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

set(FLUX_CUDA_OP_TARGETS "") # cu op targets of subdirectory
if (BUILD_THS)
  set(FLUX_THS_OP_FILES "") # ths op source files of subdirectory
endif()

add_subdirectory(comm_none)
add_subdirectory(all_gather)
add_subdirectory(reduce_scatter)
message(STATUS "op_targets: ${FLUX_CUDA_OP_TARGETS}")

# this must be after all op subdirectory added
add_subdirectory(cuda)

if (BUILD_THS)
  add_subdirectory(ths_op)
endif()
