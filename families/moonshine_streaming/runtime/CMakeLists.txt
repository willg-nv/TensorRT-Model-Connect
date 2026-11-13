# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_moonshine_streaming SHARED bpe_tokenizer.cpp plugin.cpp)
target_include_directories(trtmc_model_moonshine_streaming PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_moonshine_streaming SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_moonshine_streaming PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_moonshine_streaming PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_moonshine_streaming PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_moonshine_streaming LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
