# OneTrans 精排系统（C++ 工程主线）
#
# 目标：
#   onetrans_core   静态库（common/engine/kv/serving/net 全部数据面与编排面）
#   onetrans_server 精排服务（HTTP 接入 + 两阶段推理）
#   verify_golden   数值对拍工具（C++ 与 golden.bin 数值等价校验）
#
# 依赖（§7.4.2 线程池 / §1.5 KV 存储）：
#   folly     —— SEDA 功能分池（CPUThreadPoolExecutor / IOThreadPoolExecutor），
#                find_package(folly CONFIG) 引入，不再自研 std::thread 池；
#   datasystem（可选）—— C++ KV 后端 DatasystemKVStore，ONETRANS_WITH_DATASYSTEM=ON
#                且 find_package(Datasystem) 命中 SDK 时才编译，否则回退 LocalKVStore；
#   Python3 （可选）—— Stage II 计算桥（PyTorch 算子下发），缺失时 C++ CPU 降级。
cmake_minimum_required(VERSION 3.16)
project(onetrans_serving CXX)

# folly（CPU/IOThreadPoolExecutor）依赖 C++20（concepts/span），故全线 C++20。
set(CMAKE_CXX_STANDARD 20)
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

# ---- KV 后端：可选 datasystem C++ SDK（§1.5）---- #
# ONETRANS_WITH_DATASYSTEM=ON 且 find_package(Datasystem) 命中 SDK 时才编译
# DatasystemKVStore（调 datasystem::KVClient set/get），否则热路径回退 LocalKVStore。
option(ONETRANS_WITH_DATASYSTEM "datasystem C++ SDK 后端（KV set/get）" OFF)
if(ONETRANS_WITH_DATASYSTEM)
    find_package(Datasystem QUIET)
    if(Datasystem_FOUND)
        list(APPEND CORE_SOURCES src/kv/datasystem_store.cpp)
        message(STATUS "datasystem: SDK 命中，启用 DatasystemKVStore")
    else()
        message(WARNING "datasystem: ONETRANS_WITH_DATASYSTEM=ON 但未找到 SDK，回退 LocalKVStore")
    endif()
endif()

add_library(onetrans_core STATIC ${CORE_SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(onetrans_core PUBLIC Threads::Threads)

# ---- folly（§7.4.2）：CPU/IOThreadPoolExecutor 功能分池 ---- #
# 引入真实 folly 而非自研 std::thread 池（不再维护等价轮子）。
find_package(folly CONFIG REQUIRED)
target_link_libraries(onetrans_core PUBLIC Folly::folly)

if(ONETRANS_WITH_DATASYSTEM AND Datasystem_FOUND)
    if(Datasystem_INCLUDE_DIR)
        target_include_directories(onetrans_core PUBLIC ${Datasystem_INCLUDE_DIR})
    elseif(Datasystem_INCLUDE_DIRS)
        target_include_directories(onetrans_core PUBLIC ${Datasystem_INCLUDE_DIRS})
    endif()
    target_link_libraries(onetrans_core PUBLIC datasystem)
    target_compile_definitions(onetrans_core PUBLIC ONETRANS_WITH_DATASYSTEM=1)
endif()

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
