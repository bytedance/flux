cmake_minimum_required(VERSION 3.17)
if (WITH_PROTOBUF)
  add_compile_options("-DWITH_PROTOBUF")
  set(flux_proto_lib flux_proto ${Protobuf_LIBRARIES})
  set(flux_proto_inc ${Protobuf_INCLUDE_DIRS})
else ()
  set(flux_proto_lib)
  set(flux_proto_inc)
endif()

set(LIB_FILES
  op_registry.cu
  op_registry_proto_utils.cc
  cudaipc_barrier_all.cu
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
if(ENABLE_NVSHMEM)
target_link_libraries(flux_cuda
  PUBLIC -lnvshmem_host CUDA::cudart CUDA::cuda_driver ${FLUX_CUDA_OP_TARGETS}
  PRIVATE -lnvshmem_device ${flux_proto_lib}
)
else()
target_link_libraries(flux_cuda
  PUBLIC CUDA::cudart CUDA::cuda_driver ${FLUX_CUDA_OP_TARGETS}
  PRIVATE ${flux_proto_lib}
)
endif()
target_include_directories(flux_cuda PRIVATE ${flux_proto_inc})
target_link_options(flux_cuda
  PRIVATE
  $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs=libflux_proto:libprotobuf
                         LINKER:--version-script=${_VER_FILE}
                         LINKER:--no-as-needed>
)
