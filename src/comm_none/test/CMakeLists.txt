cmake_minimum_required(VERSION 3.17)

add_executable(test_gemm_only test_gemm_only.cc)
target_link_libraries(test_gemm_only PUBLIC flux_cuda)

