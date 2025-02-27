cmake_minimum_required(VERSION 3.17)

find_package(MPI)
if (MPI_FOUND)
    add_executable(test_moe_ag_scatter test_moe_ag_scatter.cc)
    target_link_libraries(test_moe_ag_scatter PUBLIC flux_cuda MPI::MPI_CXX)
else()
    message(STATUS "MPI not found, mpi_target will not be built")
endif()

add_executable(test_sort_utils test_sort_utils.cc)
target_link_libraries(test_sort_utils PUBLIC flux_cuda)
target_compile_options(test_sort_utils PRIVATE -g)
