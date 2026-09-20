# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Build integration only; this license does not relicense the sources in NOTICE.

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# Use the same installed CANN toolchain as the standard package build. CANN
# generates the MIX_AIC_1_2 registration and host stub; no custom linker steps.
ascendc_library(vq2a8_ascendc_v4_v2_kernel STATIC
  kernel.cpp resident_prepare.cpp
  validity_kernel.cpp route_mapping_kernel.cpp select_sign_kernel.cpp)
ascendc_include_directories(vq2a8_ascendc_v4_v2_kernel PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(vq2a8_ascendc_v4_v2 SHARED
  torch_binding.cpp validity_binding.cpp
  route_mapping_binding.cpp select_sign_binding.cpp)
target_compile_options(vq2a8_ascendc_v4_v2 PRIVATE ${TORCH_FLAGS})
target_include_directories(vq2a8_ascendc_v4_v2 PRIVATE
  ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_PATH}/include"
  "${ASCEND_HOME_PATH}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_directories(vq2a8_ascendc_v4_v2 PRIVATE
  "${TORCH_NPU_PATH}/lib" "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(vq2a8_ascendc_v4_v2 PRIVATE vq2a8_ascendc_v4_v2_kernel
  ${TORCH_LIBRARIES} torch_npu ascendcl tiling_api platform)
set_target_properties(vq2a8_ascendc_v4_v2 PROPERTIES
  BUILD_RPATH "${TORCH_NPU_PATH}/lib;${ASCEND_HOME_PATH}/lib64"
  INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib;$ORIGIN/../torch_npu/lib;${ASCEND_HOME_PATH}/lib64")
install(TARGETS vq2a8_ascendc_v4_v2 LIBRARY DESTINATION "${VLLM_ASCEND_INSTALL_PATH}")
install(FILES NOTICE DESTINATION "${VLLM_ASCEND_INSTALL_PATH}" RENAME VQ2A8_NOTICE)
