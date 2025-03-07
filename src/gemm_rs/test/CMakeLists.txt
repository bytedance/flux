cmake_minimum_required(VERSION 3.17)

add_executable(test_gemm_rs test_gemm_rs.cc)
target_link_libraries(test_gemm_rs PUBLIC flux_cuda)
