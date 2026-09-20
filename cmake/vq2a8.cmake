# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

option(VLLM_ASCEND_BUILD_VQ2A8 "Build the Ascend950 VQ2A8 kernels" OFF)

if(VLLM_ASCEND_BUILD_VQ2A8)
  string(TOLOWER "${SOC_VERSION}" VQ2A8_SOC_LOWER)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "VQ2A8 requires a Linux CANN/torch_npu build environment")
  endif()
  if(NOT VQ2A8_SOC_LOWER MATCHES "^ascend950" OR NOT RUN_MODE STREQUAL "npu")
    message(FATAL_ERROR "VQ2A8 requires an exact Ascend950 SOC_VERSION and RUN_MODE=npu")
  endif()
  add_subdirectory(csrc/vq2a8_ascendc_v4_v2)
  # setup.py builds this standard target rather than all CMake targets.
  add_dependencies(vllm_ascend_C vq2a8_ascendc_v4_v2)
endif()
