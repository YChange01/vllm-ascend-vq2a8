# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
cmake_minimum_required(VERSION 3.16)
project(vq2a8_ascendc_v3 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(RUN_MODE npu CACHE STRING "AscendC execution mode")
set(SOC_VERSION Ascend950PR_957d CACHE STRING "Exact Ascend950 target")
string(TOLOWER "${SOC_VERSION}" SOC_LOWER)
if(NOT SOC_LOWER MATCHES "^ascend950" OR NOT RUN_MODE STREQUAL "npu")
  message(FATAL_ERROR "This experimental kernel requires Ascend950 and RUN_MODE=npu")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Build the AscendC prototype on Linux with CANN installed")
endif()
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
# The v3 builder reuses v2 SDK discovery and pins THIS exact recipe in the
# manifest. Never rediscover a different file after accepting that selection.
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
  message(FATAL_ERROR "Installed CANN ascendc.cmake not found; use tools/build_vq2a8_ascendc_v3.py for discovery")
endif()
include("${VQ2A8_ASCENDC_CMAKE}")
if(NOT COMMAND ascendc_library)
  message(FATAL_ERROR "Installed CANN CMake does not expose ascendc_library")
endif()
find_package(Torch REQUIRED)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
# Independent opt-in target: do not enable the unrelated direct kernels which
# the root build deliberately excludes on A5. CANN performs native compilation.
ascendc_library(vq2a8_ascendc_v3_kernel STATIC kernel.cpp)
ascendc_include_directories(vq2a8_ascendc_v3_kernel PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_library(vq2a8_ascendc_v3 SHARED torch_binding.cpp)
target_compile_options(vq2a8_ascendc_v3 PRIVATE ${TORCH_FLAGS})
target_include_directories(vq2a8_ascendc_v3 PRIVATE ${TORCH_INCLUDE_DIRS}
  "${TORCH_NPU_PATH}/include" "${ASCEND_HOME_PATH}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_directories(vq2a8_ascendc_v3 PRIVATE "${TORCH_NPU_PATH}/lib" "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(vq2a8_ascendc_v3 PRIVATE vq2a8_ascendc_v3_kernel ${TORCH_LIBRARIES}
  torch_npu ascendcl tiling_api platform)
set_target_properties(vq2a8_ascendc_v3 PROPERTIES BUILD_RPATH
  "${TORCH_NPU_PATH}/lib;${ASCEND_HOME_PATH}/lib64"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
