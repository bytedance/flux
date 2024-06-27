cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FLUX LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

# cmake global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(BUILD_THS ON CACHE INTERNAL "Build Torch op")
set(BUILD_TEST ON CACHE INTERNAL "Build unit tests")
set(ENABLE_NVSHMEM ON CACHE INTERNAL "Use NVSHMEM to transfer data")
set(CUTLASS_TRACE OFF CACHE INTERNAL "Print CUTLASS Host Trace info")
set(FLUX_DEBUG OFF CACHE INTERNAL "Define FLUX_DEBUG")
OPTION(WITH_PROTOBUF "build with protobuf" OFF)
message("PYTHONPATH: ${PYTHONPATH}")
message("NVShmem Support: ${ENABLE_NVSHMEM}")

# find cuda
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
  message(FATAL_ERROR "requires cuda to be >= 11.0")
elseif(CUDAToolkit_VERSION VERSION_LESS "12.0")
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

string(JOIN " " JOINED_CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${JOINED_CUDA_ARCH_FLAGS}")
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wno-psabi")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -DCUTLASS_DEBUG_TRACE_LEVEL=0")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

if(CUTLASS_TRACE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTLASS_DEBUG_TRACE_LEVEL=1")
endif()

if(FLUX_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFLUX_DEBUG")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFLUX_DEBUG")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(BUILD_THS)
  message("Build THS on")
  find_package(Python 3 REQUIRED)
  find_program(PYTHON_EXECUTABLE NAMES python3 python)

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

  execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());
print(sysconfig.get_config_var('EXT_SUFFIX'));"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE _PYTHON_VALUES)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message("PY:${PYTHON_EXECUTABLE}")
      message(FATAL_ERROR "Python config Error.")
  endif()
  string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
  string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
  list(GET _PYTHON_VALUES 0 PY_INCLUDE_DIR)
  list(GET _PYTHON_VALUES 1 PY_SUFFIX)
  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

  execute_process(COMMAND ${PYTHON_EXECUTABLE} "-c"
                  "from torch.utils import cpp_extension; import re; import torch;                                    \
                   version = tuple(int(i) for i in re.match('(\\d+)\\.(\\d+)\\.(\\d+)', torch.__version__).groups()); \
                   args = ([],True,False,False) if version >= (1, 8, 0) else ([],True,False);                         \
                   print(' '.join(cpp_extension._prepare_ldflags(*args)),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_LINK)
  message("-- TORCH_LINK ${TORCH_LINK}")
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "PyTorch link config Error.")
  endif()
endif()

# force use sm90a for cutlass
string(REGEX REPLACE "sm_90" "sm_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX REPLACE "compute_90" "compute_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/library/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/profiler/include
)

set(COMMON_LIB_DIRS "")
list(APPEND COMMON_LIB_DIRS "${CUDAToolkit_LIBRARY_DIR}")
message(ENABLE_NVSHMEM "ENABLE_NVSHMEM is set to: ${ENABLE_NVSHMEM}")
if(ENABLE_NVSHMEM)
  add_definitions(-DFLUX_SHM_USE_NVSHMEM)
  set(NVSHMEM_BUILD_DIR ${PROJECT_SOURCE_DIR}/3rdparty/nvshmem/build)
  message(STATUS "NVSHMEM build dir: ${NVSHMEM_BUILD_DIR}")
  if(NOT EXISTS ${NVSHMEM_BUILD_DIR})
      message(FATAL_ERROR "NVSHMEM not found. Please run ./build_nvshmem.sh first.")
  endif()
  list(APPEND COMMON_HEADER_DIRS "${NVSHMEM_BUILD_DIR}/src/include")
  list(APPEND COMMON_LIB_DIRS "${NVSHMEM_BUILD_DIR}/src/lib")
endif()

# append headers explicitly for .cu files, in order to enable vscode clangd intellisense
foreach(inc_dir ${COMMON_HEADER_DIRS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -I${inc_dir}")
endforeach()
message("final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(src)
