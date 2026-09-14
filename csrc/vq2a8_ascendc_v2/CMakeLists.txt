# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build integration only; this license does not relicense the expert sources.
cmake_minimum_required(VERSION 3.16)
project(vq2a8_ascendc_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(RUN_MODE npu CACHE STRING "AscendC execution mode")
set(SOC_VERSION "" CACHE STRING "Exact Ascend950 SoC from the installed CANN platform config")
string(TOLOWER "${SOC_VERSION}" SOC_LOWER)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Build VQ2A8 operator v2 on Linux with CANN and torch_npu installed")
endif()
if(NOT SOC_LOWER MATCHES "^ascend950" OR NOT RUN_MODE STREQUAL "npu")
  message(FATAL_ERROR "Require exact Ascend950 SOC_VERSION and RUN_MODE=npu")
endif()
if(NOT IS_DIRECTORY "${ASCEND_HOME_PATH}" OR NOT IS_DIRECTORY "${TORCH_NPU_PATH}")
  message(FATAL_ERROR "ASCEND_HOME_PATH and TORCH_NPU_PATH must be existing directories")
endif()

set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
if(NOT VQ2A8_ASCENDC_CMAKE)
  foreach(candidate
      tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake
      compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake
      ascendc_devkit/tikcpp/samples/cmake/ascendc.cmake
      aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake
      x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake
      aarch64-linux/asc/cmake/ascendc.cmake
      x86_64-linux/asc/cmake/ascendc.cmake
      tools/ascendc_kernel_cmake/ascendc.cmake
      cmake/asc/ascendc.cmake)
    if(EXISTS "${ASCEND_HOME_PATH}/${candidate}")
      set(VQ2A8_ASCENDC_CMAKE "${ASCEND_HOME_PATH}/${candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT EXISTS "${VQ2A8_ASCENDC_CMAKE}")
  message(FATAL_ERROR "Installed CANN ascendc.cmake not found; use tools/build_vq2a8_ascendc_v2.py for discovery")
endif()
include("${VQ2A8_ASCENDC_CMAKE}")
if(NOT COMMAND ascendc_library)
  message(FATAL_ERROR "Installed CANN CMake does not expose ascendc_library")
endif()

# The installed CANN framework generates AIC/AIV objects, MIX_1_2 metadata,
# binary registration and the host launch stub. Do not guess ld.lld flags or
# link chat-recovered hacl_rt.h declarations into the PyTorch extension.
ascendc_library(vq2a8_ascendc_v2_kernel STATIC kernel.cpp)
ascendc_include_directories(vq2a8_ascendc_v2_kernel PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Torch REQUIRED)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_library(vq2a8_ascendc_v2 SHARED torch_binding.cpp)
target_compile_options(vq2a8_ascendc_v2 PRIVATE ${TORCH_FLAGS})
target_include_directories(vq2a8_ascendc_v2 PRIVATE
  ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_PATH}/include"
  "${ASCEND_HOME_PATH}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_directories(vq2a8_ascendc_v2 PRIVATE
  "${TORCH_NPU_PATH}/lib" "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(vq2a8_ascendc_v2 PRIVATE vq2a8_ascendc_v2_kernel
  ${TORCH_LIBRARIES} torch_npu ascendcl tiling_api platform)
set_target_properties(vq2a8_ascendc_v2 PROPERTIES
  BUILD_RPATH "${TORCH_NPU_PATH}/lib;${ASCEND_HOME_PATH}/lib64"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
