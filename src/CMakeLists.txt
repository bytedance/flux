cmake_minimum_required(VERSION 3.17)

set(GEMM_OP_REG_INP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gemm_op_registry.cu.in)

# append 'flux/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# create an object library for each op subfolder
function(flux_add_op_cu_obj_lib TARGET FILES)
  add_library(${TARGET} OBJECT ${FILES})
  set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart ${CMAKE_THREAD_LIBS_INIT} )
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

set_property(GLOBAL PROPERTY FLUX_THS_TARGETS)
function (flux_add_ths_op_target TARGET)
  get_property(FLUX_THS_TARGETS_LOCAL GLOBAL PROPERTY FLUX_THS_TARGETS)
  set_property(GLOBAL PROPERTY FLUX_THS_TARGETS "${TARGET};${FLUX_THS_TARGETS_LOCAL}")
endfunction()

set(FLUX_CUDA_OP_TARGETS "") # cu op targets of subdirectory
if (BUILD_THS)
  set(FLUX_THS_OP_FILES "") # ths op source files of subdirectory
endif()

include(ExternalProject)
ExternalProject_Add(
  FLUX_GENERATOR
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generator
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""  # Prevents automatic build during build phase
  INSTALL_COMMAND ""  # Prevents automatic install
)
ExternalProject_Get_property(FLUX_GENERATOR SOURCE_DIR)
ExternalProject_Get_property(FLUX_GENERATOR BINARY_DIR)
set(FLUX_GENERATOR_BINARY_DIR ${BINARY_DIR})
set(FLUX_GENERATOR_SOURCE_DIR ${SOURCE_DIR})

execute_process(
  COMMAND ${CMAKE_COMMAND} ${FLUX_GENERATOR_SOURCE_DIR}
  WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND ${CMAKE_COMMAND} --build . --parallel 8
  WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

if (WITH_TRITON_AOT)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import triton; from packaging.version import Version; assert Version(triton.__version__) >= Version('3.0.0')"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(MOE_AG_SCATTER_KERNEL ${PROJECT_SOURCE_DIR}/python/flux_triton/kernels/moe_ag_scatter.py:moe_ag_scatter_grouped_gemm_kernel ${PROJECT_SOURCE_DIR}/python/flux_triton/kernels/moe_gather_rs.py:moe_gather_rs_grouped_gemm_kernel)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/python/flux_triton/tools/compile_aot.py --workspace ${CMAKE_CURRENT_SOURCE_DIR}/triton_aot_generated --kernels ${MOE_AG_SCATTER_KERNEL}  --library flux_triton_aot
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  add_subdirectory(triton_aot_generated)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/triton_aot_generated)
  add_definitions(-DFLUX_WITH_TRITON_AOT)
endif()

add_subdirectory(comm_none)
add_subdirectory(coll)
add_subdirectory(ag_gemm)
add_subdirectory(gemm_rs)
add_subdirectory(gemm_a2a_transpose)
add_subdirectory(a2a_transpose_gemm)
add_subdirectory(inplace_cast)
if(ENABLE_NVSHMEM)
  # MoE OPs will not be open-sourced, so remain MoE OPs to be dependent on NVSHMEM
  add_subdirectory(moe_ag_scatter)
  add_subdirectory(moe_gather_rs)
endif()
message(STATUS "op_targets: ${FLUX_CUDA_OP_TARGETS}")

# this must be after all op subdirectory added
add_subdirectory(cuda)

if (BUILD_THS)
  add_subdirectory(ths_op)
endif()

include(GNUInstallDirs)
install(
  DIRECTORY ./ "${PROJECT_SOURCE_DIR}/include/flux/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flux/
  FILES_MATCHING PATTERN "*/ths_op/*.h")

file(GLOB FLUX_HEADERS ${PROJECT_SOURCE_DIR}/include/flux/*.h)
install(
  FILES ${FLUX_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flux/
)
install(
  FILES ${PROJECT_SOURCE_DIR}/cmake/FluxConfig.cmake
  DESTINATION share/cmake/Flux/
)
