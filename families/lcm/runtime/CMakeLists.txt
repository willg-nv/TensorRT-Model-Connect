# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_lcm SHARED bpe_tokenizer.cpp plugin.cpp)
target_include_directories(trtmc_model_lcm PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_lcm SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_lcm PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_lcm PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_lcm PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_lcm LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_lcm_scheduler ${PROJECT_SOURCE_DIR}/families/lcm/tests/cpp/test_scheduler.cpp)
  target_include_directories(test_lcm_scheduler PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_lcm_scheduler PRIVATE nlohmann_json::nlohmann_json)
  set_target_properties(test_lcm_scheduler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
