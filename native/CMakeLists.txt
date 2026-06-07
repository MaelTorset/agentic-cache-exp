cmake_minimum_required(VERSION 3.16)
project(agentic_cache_native_probes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_CPP_ROOT "/data/llama/llama.cpp" CACHE PATH "Path to a llama.cpp checkout")
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build" CACHE PATH "Path to a built llama.cpp tree")

add_executable(semantic-kv-probe semantic_kv_probe.cpp)

target_include_directories(semantic-kv-probe PRIVATE
  "${LLAMA_CPP_ROOT}/include"
  "${LLAMA_CPP_ROOT}/ggml/include"
)

target_link_directories(semantic-kv-probe PRIVATE
  "${LLAMA_CPP_BUILD}/bin"
)

target_link_libraries(semantic-kv-probe PRIVATE llama ggml ggml-base ggml-cpu)

set_target_properties(semantic-kv-probe PROPERTIES
  BUILD_RPATH "${LLAMA_CPP_BUILD}/bin"
  INSTALL_RPATH "${LLAMA_CPP_BUILD}/bin"
)
