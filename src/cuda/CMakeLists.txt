cmake_minimum_required(VERSION 3.17)

set(LIB_FILES
  op_registry.cu
  bitwise_check.cu
  random_initialize.cu
  utils.cc
  cuda_common.cc
  cuda_stub.cc
  nvml_stub.cc
)

set(_VER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/version.ld)

add_library(flux_cuda SHARED ${LIB_FILES})
set_target_properties(flux_cuda PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  LINK_DEPENDS ${_VER_FILE}
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(flux_cuda
  PUBLIC -lnvshmem_host CUDA::cudart CUDA::cuda_driver ${FLUX_CUDA_OP_TARGETS}
  PRIVATE -lnvshmem_device
)

target_link_options(flux_cuda
  PRIVATE
  $<$<PLATFORM_ID:Linux>:LINKER:--version-script=${_VER_FILE}
                         LINKER:--no-as-needed>
)
