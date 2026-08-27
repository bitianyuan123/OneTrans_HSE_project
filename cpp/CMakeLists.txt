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
    src/common/executor.cpp
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
    src/serving/compute_bridge.cpp
    src/serving/embed_lookup.cpp
    src/serving/flow.cpp
    src/serving/json_io.cpp
    src/serving/pipeline.cpp
)

add_library(onetrans_core STATIC ${CORE_SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(onetrans_core PUBLIC Threads::Threads)

# ---- Python 计算桥（可选，§7.4.5）：嵌入解释器下发 PyTorch 算子 ---- #
# ONETRANS_PYTHON_BRIDGE=ON 且找到 Python3 Development 时启用；
# 否则编译为降级桩（available()==false，Stage II 走 C++ CPU）。
option(ONETRANS_PYTHON_BRIDGE "嵌入 Python 计算桥（Stage II → PyTorch）" ON)
if(ONETRANS_PYTHON_BRIDGE)
    # 优先用当前 PATH 里的 python3（与 torch 安装解释器一致），
    # 可 -DPython3_EXECUTABLE=... 显式覆盖
    if(NOT Python3_EXECUTABLE)
        execute_process(COMMAND python3 -c "import sys; print(sys.executable)"
                        OUTPUT_VARIABLE _PY_EXE OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_PY_EXE)
            set(Python3_EXECUTABLE ${_PY_EXE})
        endif()
    endif()
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_Development_FOUND)
        message(STATUS "python bridge: ${Python3_EXECUTABLE} (${Python3_VERSION})")
        target_compile_definitions(onetrans_core PUBLIC ONETRANS_WITH_PYTHON=1)
        target_link_libraries(onetrans_core PUBLIC Python3::Python)
    else()
        message(STATUS "python bridge: 未找到 Python3 Development（降级 C++ CPU）")
    endif()
endif()

add_executable(onetrans_server tools/server_main.cpp)
target_link_libraries(onetrans_server PRIVATE onetrans_core)

add_executable(verify_golden tools/verify_golden.cpp)
target_link_libraries(verify_golden PRIVATE onetrans_core)

enable_testing()
add_test(NAME verify_golden
         COMMAND verify_golden --weights ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/weights
                               --golden ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/golden)
