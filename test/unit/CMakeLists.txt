cmake_minimum_required(VERSION 3.17)

link_libraries(CUDA::cudart CUDA::nvml ${CMAKE_THREAD_LIBS_INIT})

include_directories(${COMMON_HEADER_DIRS})
find_package(gflags)
if(False)
# if(gflags_FOUND)
  message("gflags_FOUND: ${gflags_FOUND}")
  include_directories(${gflags_INCLUDE_DIR})
  link_libraries(${gflags_LIBRARIES})
  add_compile_options("-DWITH_GFLAGS")
endif()

find_package(glog)
# if(glog_FOUND)
if (False)
  message("glog_FOUND: ${glog_FOUND}")
  include_directories(${glog_INCLUDE_DIR})
  link_libraries(glog::glog)
  add_compile_options("-DFLUX_USE_GLOG")
endif()

find_package(NUMA)
if(NUMA_FOUND)
  message("NUMA_FOUND: ${NUMA_FOUND}")
  include_directories(${NUMA_INCLUDE_DIR})
  link_libraries(NUMA::NUMA)
  add_compile_options("-DWITH_NUMA")
endif()

add_executable(test_flux_templates test_flux_templates.cc)
target_link_libraries(test_flux_templates PUBLIC ${CMAKE_THREAD_LIBS_INIT})

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
  add_executable(test_tma_copy test_tma_copy.cu)
  target_link_libraries(test_tma_copy PRIVATE CUDA::cuda_driver)
  set_target_properties(test_tma_copy PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(test_vectorized_copy test_vectorized_copy.cu)
set_target_properties(test_vectorized_copy
                      PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(test_cute test_cute.cc)

add_executable(test_cuda_common test_cuda_common.cu)
target_link_libraries(test_cuda_common PUBLIC flux_cuda)

add_executable(test_memory_bound test_memory_bound.cc)
target_link_libraries(test_memory_bound PUBLIC flux_cuda)

add_executable(test_tuning test_tuning.cc)
target_link_libraries(test_tuning PUBLIC flux_cuda)

add_executable(copy_d2d_perf copy_d2d_perf.cu)

add_executable(copy_d2d_cute_perf copy_d2d_cute_perf.cu)
target_link_libraries(copy_d2d_cute_perf PUBLIC NUMA::NUMA flux_cuda)

# using this for reduce_scatter_kernel.hpp to perf ReduceScatterRing[1,2]d[Pull,Push]Gemmk without GEMM
# ./build/bin/reduce_scatter_kernel_perf \
#   -warmup_iters=3 -iters 10  \
#   -m 4096 -n 12288 \  # the shape, dtype is default to FP16
#   -run_wait=true \    # only works when compiles with -DFLUX_DEBUG_RS=1. don't wait tile flags and copy at full speed
#   -num_blocks 8 \     # launch config. using 8 grids and 1024 threads per grid
#   -ngpus 8 \          # number of GPUs. for TP=8, set to 8.
#   -flatten=false \    # should flatten the tile. see implementation from reduce_scatter_kernel.hpp
#   -1d_ring=true \     # the algorithm. use 1d ring or 2d ring.
#   -sleep_ns=0 \       # this is for E2E gemm+reduce_scatter. to avoid reduce_scatter makes GEMM slow, force reduce_scatter to sleep
#   -push               # push means using push mode, that's P2P write mode. default is false.
add_executable(reduce_scatter_kernel_perf reduce_scatter_kernel_perf.cu)
target_link_libraries(reduce_scatter_kernel_perf PUBLIC NUMA::NUMA flux_cuda)
target_include_directories(reduce_scatter_kernel_perf
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_options(reduce_scatter_kernel_perf PUBLIC -g) # -DFLUX_DEBUG=1)

add_executable(test_pingpong_latency test_pingpong_latency.cu)

add_executable(test_globaltimer test_globaltimer.cu)

add_executable(copy_d2h_perf copy_d2h_perf.cu)
target_link_libraries(copy_d2h_perf PUBLIC NUMA::NUMA flux_cuda)
target_compile_options(copy_d2h_perf PUBLIC -g) # -DFLUX_DEBUG=1)

add_executable(
  test_topo_utils test_topo_utils.cc)
target_link_libraries(test_topo_utils PUBLIC ${TORCH_LIBRARIES} flux_cuda flux_cuda_ths_op)
target_include_directories(test_topo_utils PUBLIC ${TORCH_INCLUDE_DIRS})

add_executable(inplace_cast inplace_cast.cu)
target_compile_options(inplace_cast PUBLIC -g) # -DFLUX_DEBUG=1)

set_property(
  TARGET copy_d2h_perf copy_d2d_perf test_globaltimer test_pingpong_latency inplace_cast
  PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
