cmake_minimum_required(VERSION 3.17)

add_executable(test_gemm_only test_gemm_only.cc)
target_link_libraries(test_gemm_only PUBLIC flux_cuda)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.3")
    add_executable(test_grouped_gemm_v3_comm_none test_grouped_gemm_comm_none.cc)
    target_link_libraries(test_grouped_gemm_v3_comm_none PUBLIC flux_cuda)

    add_executable(test_blockscale_gemm_v3_comm_none test_blockscale_gemm_comm_none.cu)
    target_link_libraries(test_blockscale_gemm_v3_comm_none PUBLIC flux_cuda)
endif()

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4")
    add_executable(test_sm89_fp8_gemm_v2_comm_none test_sm89_fp8_gemm_comm_none.cu)
    target_link_libraries(test_sm89_fp8_gemm_v2_comm_none PUBLIC flux_cuda)
endif()
