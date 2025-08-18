# FindFlux
# -------
#
# Finds the Flux library
#
# This will define the following variables:
#
#   FLUX_FOUND        -- True if the system has the Flux library
#   FLUX_INCLUDE_DIRS -- The include directories for flux
#   FLUX_LIBRARIES    -- Libraries to link against
#   FLUX_CXX_FLAGS    -- Additional (required) compiler flags
#
# and the following imported targets:
#
#   flux
macro(append_fluxlib_if_found)
  foreach (_arg ${ARGN})
    find_library(${_arg}_LIBRARY ${_arg} PATHS "${FLUX_INSTALL_PREFIX}/lib")
    if(${_arg}_LIBRARY)
      list(APPEND FLUX_LIBRARIES ${${_arg}_LIBRARY})
    else()
      message(WARNING "library ${${_arg}_LIBRARY} not found.")
    endif()
  endforeach()
endmacro()

include(FindPackageHandleStandardArgs)

if(DEFINED ENV{FLUX_INSTALL_PREFIX})
  set(FLUX_INSTALL_PREFIX $ENV{FLUX_INSTALL_PREFIX})
else()
  # Assume we are in <install-prefix>/share/cmake/Flux/FluxConfig.cmake
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(FLUX_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
endif()

# Include directories.
set(FLUX_INCLUDE_DIRS
  ${FLUX_INSTALL_PREFIX}/include
  ${FLUX_INSTALL_PREFIX}/include/flux
)


# Library dependencies.
append_fluxlib_if_found(flux_cuda)
append_fluxlib_if_found(flux_cuda_ths_op)

# When we build libflux with the old libstdc++ ABI, dependent libraries must too.
# if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
#   set(FLUX_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=@GLIBCXX_USE_CXX11_ABI@")
# endif()

find_library(FLUX_LIBRARY flux_cuda_ths_op PATHS "${FLUX_INSTALL_PREFIX}/lib")
# set_target_properties(flux PROPERTIES
#     INTERFACE_INCLUDE_DIRECTORIES "${FLUX_INCLUDE_DIRS}"
#     CXX_STANDARD 17
# )
# if(FLUX_CXX_FLAGS)
#   set_property(TARGET flux PROPERTY INTERFACE_COMPILE_OPTIONS "${FLUX_CXX_FLAGS}")
# endif()

find_package_handle_standard_args(Flux DEFAULT_MSG FLUX_LIBRARY FLUX_INCLUDE_DIRS)
