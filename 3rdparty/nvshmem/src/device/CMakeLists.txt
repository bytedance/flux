# Allow users to set the CUDA toolkit through the env.
cmake_minimum_required(VERSION 3.19)

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/../.." CACHE PATH "..." FORCE)

if (DEFINED ENV{NVSHMEM_DEVICELIB_CUDA_HOME})
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT $ENV{NVSHMEM_DEVICELIB_CUDA_HOME})
elseif(DEFINED ENV{CUDA_HOME})
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT $ENV{CUDA_HOME})
else()
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT "/usr/local/cuda")
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_UNDEFINED 1)
endif()

if (NOT DEFINED CUDA_ARCHITECTURES)
  set(CUDA_ARCHITECTURES_UNDEFINED 1)
endif()

set(NVSHMEM_DEVICELIB_CUDA_HOME ${NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT} CACHE PATH "path to CUDA installation")

if (NVSHMEM_DEBUG)
	message(STATUS "Setting build type to Debug as requested by the environment.")
  set(CMAKE_BUILD_TYPE "debug" CACHE STRING "Choose the type of build." FORCE)
endif()

if(NOT CUDAToolkit_Root AND NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA_DEVICELIB_HOME: ${NVSHMEM_DEVICELIB_CUDA_HOME}")
  set(CUDAToolkit_Root ${NVSHMEM_DEVICELIB_CUDA_HOME} CACHE PATH "Root of Cuda Toolkit." FORCE)
  set(CMAKE_CUDA_COMPILER "${NVSHMEM_DEVICELIB_CUDA_HOME}/bin/nvcc" CACHE PATH "Root of Cuda Toolkit." FORCE)
endif()

project(
  NVSHMEM_DEVICE
  LANGUAGES CUDA CXX C
  VERSION ${PROJECT_VERSION}
)

find_package(CUDAToolkit)

if(DEFINED CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
  if(NOT DEFINED CUDA_ARCHITECTURES_UNDEFINED)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} CACHE STRING "CUDA ARCHITECTURES" FORCE)
  else()
    if(CUDAToolkit_VERSION_MAJOR LESS 11)
      set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA ARCHITECTURES" FORCE)
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CUDAToolkit_VERSION_MINOR LESS 8)
      set(CMAKE_CUDA_ARCHITECTURES 70 80 CACHE STRING "CUDA ARCHITECTURES" FORCE)
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CUDAToolkit_VERSION_MINOR LESS 7)
      set(CMAKE_CUDA_ARCHITECTURES 70 80 90 CACHE STRING "CUDA ARCHITECTURES" FORCE)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 80 90 100 CACHE STRING "CUDA ARCHITECTURES" FORCE)
    endif()
  endif()
endif()
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  set(NVCC_GENCODE "${NVCC_GENCODE}" "-gencode=arch=compute_${ARCH},code=sm_${ARCH};-gencode=arch=compute_${ARCH},code=compute_${ARCH}")
endforeach()
mark_as_advanced(NVCC_GENCODE CMAKE_CUDA_ARCHITECTURES_UNDEFINED CUDA_ARCHITECTURES_UNDEFINED)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "NVSHMEM_PREFIX: ${NVSHMEM_PREFIX}")
message(STATUS "NVSHMEM_DEBUG: ${NVSHMEM_DEBUG}")

install(
  EXPORT NVSHMEMDeviceTargets
  NAMESPACE nvshmem::
  DESTINATION lib/cmake/nvshmem/
)

add_library(
  nvshmem_device STATIC
)

install(
	TARGETS nvshmem_device
  EXPORT NVSHMEMDeviceTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include
)

set_target_properties(nvshmem_device
                      PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CXX_STANDARD_REQUIRED ON
                      CUDA_STANDARD_REQUIRED ON
                      CXX_STANDARD 11
                      CUDA_STANDARD 11
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../lib"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../lib"
		      VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
		      SOVERSION ${PROJECT_VERSION_MAJOR}
)

macro(nvshmem_library_set_base_config LIBNAME)
  target_compile_definitions(${LIBNAME}
    PRIVATE $<$<CONFIG:Debug>:_NVSHMEM_DEBUG;NVSHMEM_IBGDA_DEBUG>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","x86_64">,NVSHMEM_X86_64,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","ppc64le">,__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS;NVSHMEM_PPC64LE,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","aarch64">,NVSHMEM_AARCH64,>
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
  )

  target_compile_options(${LIBNAME}
    INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVSHMEM_VERBOSE}>>:-Xptxas -v>
    PRIVATE $<IF:$<CONFIG:Debug>,-O0;-g;,-O3>
    $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    $<IF:$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>,-O0;-g;-G;,-O3>
    $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},"x86_64">,-msse,>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  )

  if(NVSHMEM_DEVEL)
    target_compile_options(
      ${LIBNAME}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror
              all-warnings>
              $<$<COMPILE_LANGUAGE:CXX>:-Werror
              -Wall
              -Wextra
              -Wno-unused-function
              -Wno-unused-parameter
              -Wno-missing-field-initializers>
    )
  endif()
endmacro()

set(NVSHMEM_DEVICE_SOURCES
    init/init_device.cu
    launch/collective_launch.cpp
)

target_include_directories(
  nvshmem_device
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
            $<INSTALL_INTERFACE:include>
)

if(NVSHMEM_ENABLE_ALL_DEVICE_INLINING)
  configure_file(../include/non_abi/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_SOURCE_DIR}/../include/non_abi/device/pt-to-pt/transfer_device.cuh COPYONLY)
else()
  set(NVSHMEM_DEVICE_SOURCES
      ${NVSHMEM_DEVICE_SOURCES}
      comm/transfer_device.cu
  )
  configure_file(../include/non_abi/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_SOURCE_DIR}/comm/transfer_device.cu COPYONLY)
endif()

target_link_libraries(nvshmem_device PRIVATE CUDA::cudart_static)

if(NVSHMEM_USE_NCCL)
  target_include_directories(nvshmem_device PRIVATE ${NCCL_INCLUDE})
endif()

nvshmem_library_set_base_config(nvshmem_device)

set_source_files_properties(${NVSHMEM_DEVICE_SOURCES} PROPERTIES COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>)
target_sources(nvshmem_device PRIVATE ${NVSHMEM_DEVICE_SOURCES} ${NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT})
