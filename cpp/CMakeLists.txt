# OneTrans 精排系统（C++ 工程主线）
#
# 目标：
#   onetrans_core   静态库（common/engine/kv/serving/net 全部数据面与编排面）
#   onetrans_server 精排服务（HTTP 接入 + 两阶段推理）
#   verify_golden   数值对拍工具（C++ vs Python golden）
#
# 依赖：仅 C++17 标准库（无第三方依赖，Linux）。
cmake_minimum_required(VERSION 3.16)
project(onetrans_serving CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CORE_SOURCES
    src/common/json.cpp
    src/common/sha256.cpp
    src/common/tensor.cpp
    src/engine/frontend.cpp
    src/engine/model.cpp
    src/engine/two_stage.cpp
    src/kv/router.cpp
    src/kv/serialize.cpp
    src/kv/store.cpp
    src/net/http_server.cpp
    src/serving/embed_lookup.cpp
    src/serving/json_io.cpp
    src/serving/pipeline.cpp
)

add_library(onetrans_core STATIC ${CORE_SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(onetrans_core PUBLIC Threads::Threads)

add_executable(onetrans_server tools/server_main.cpp)
target_link_libraries(onetrans_server PRIVATE onetrans_core)

add_executable(verify_golden tools/verify_golden.cpp)
target_link_libraries(verify_golden PRIVATE onetrans_core)

enable_testing()
add_test(NAME verify_golden
         COMMAND verify_golden --weights ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/weights
                               --golden ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/golden)
