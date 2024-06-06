cmake_minimum_required(VERSION 3.17)

file(GLOB OTHER_THS_FILES *.cc)
set(FLUX_THS_FILES
    ${FLUX_THS_OP_FILES}
    ${OTHER_THS_FILES}
)

message(STATUS "ths_files: ${FLUX_THS_FILES}")

# Write the unchached variable to a file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flux_ths_files.txt" "FLUX_THS_FILES=${FLUX_THS_FILES}\n")
