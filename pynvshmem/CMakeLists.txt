cmake_minimum_required(VERSION 3.19)
project(
  pynvshmem
  LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
link_directories(
  ${CUDA_TOOLKIT_ROOT_DIR}/lib
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs
)

message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
if(CUDA_VERSION VERSION_LESS "11.0")
  message(FATAL_ERROR "requires cuda to be >= 11.0")
elseif(CUDA_VERSION VERSION_LESS "12.0")
  set(CUDAARCHS "80" CACHE STRING "CUDA Architectures")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4")
  set(CUDAARCHS "80;89;90" CACHE STRING "CUDA Architectures")
else()
  set(CUDAARCHS "80;90" CACHE STRING "CUDA Architectures")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDAARCHS})
message(STATUS "CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

set(CUDA_ARCH_FLAGS)
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  list(APPEND CUDA_ARCH_FLAGS "-gencode=arch=compute_${ARCH},code=\\\"sm_${ARCH},compute_${ARCH}\\\"")
endforeach()

set(TORCH_CUDA_ARCH_LIST)
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  if(ARCH STREQUAL "80")
    list(APPEND TORCH_CUDA_ARCH_LIST "8.0")
  elseif(ARCH STREQUAL "89")
    list(APPEND TORCH_CUDA_ARCH_LIST "8.9")
  elseif(ARCH STREQUAL "90")
    list(APPEND TORCH_CUDA_ARCH_LIST "9.0")
  else()
    message(WARNING "Unsupported CUDA arch [${ARCH}] for TORCH_CUDA_ARCH")
  endif()
endforeach()

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(NVSHMEM HINTS ${NVSHMEM_PREFIX}/lib/cmake/nvshmem)
if (NVSHMEM_FOUND)
  add_library(nvshmem ALIAS nvshmem::nvshmem)
else()
  if (EXISTS ${NVSHMEM_HOME})
    message("USING NVSHMEM " ${NVSHMEM_HOME})
  else()
    message(FATAL_ERROR "NO NVSHMEM found. please set NVSHMEM_HOME")
  endif()
  list(APPEND NVSHMEM_INCLUDE_DIRS
    "${NVSHMEM_HOME}/include"
    "${NVSHMEM_HOME}/include/modules/common"
    "${NVSHMEM_HOME}/include/modules/bootstrap")
  add_library(nvshmem IMPORTED INTERFACE)
  target_link_libraries(nvshmem INTERFACE "${NVSHMEM_HOME}/lib/libnvshmem.a" CUDA::cudart_static nvidia-ml)
  target_link_directories(nvshmem INTERFACE "${NVSHMEM_HOME}/lib")
endif()


######## pytorch ########
find_program(PYTHON_EXECUTABLE NAMES python3 python)
execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message("PY:${PYTHONPATH}")
      message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_DIR}/lib")

######## options ########
option(BUILD_EXAMPLE "build examples or not" ON)

add_library(nvshmem_bootstrap_torch SHARED)
target_sources(nvshmem_bootstrap_torch PRIVATE bootstrap/torch/bootstrap_torch.cpp)
target_link_libraries(nvshmem_bootstrap_torch PRIVATE ${TORCH_LIBRARIES})
target_include_directories(
  nvshmem_bootstrap_torch
  PRIVATE ${NVSHMEM_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS}
)
set_target_properties(nvshmem_bootstrap_torch PROPERTIES PREFIX "")

######## pynvshmem ########
set(PYNVSHMEM "pynvshmem")
message(STATUS "PYTHON3 INC: ${Python3_INCLUDE_DIRS}")
message(STATUS "COMMON INC: ${NVSHMEM_INCLUDE_DIRS}")
add_library(${PYNVSHMEM} SHARED src/module.cpp)
set_target_properties(${PYNVSHMEM} PROPERTIES PREFIX "")
set_target_properties(${PYNVSHMEM} PROPERTIES SUFFIX ".${Python3_SOABI}.so")
target_link_libraries(${PYNVSHMEM} "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" nvshmem)
target_compile_options(${PYNVSHMEM} PRIVATE -DPYBIND11_COMPILER_TYPE="_gcc" -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011")
target_include_directories(
  ${PYNVSHMEM}
  PRIVATE ${NVSHMEM_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
)
set_target_properties(${PYNVSHMEM} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
