cmake_minimum_required(VERSION 3.24)
project(VaidyLang LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    set(VAIDYLANG_BLAS ${ACCELERATE_FRAMEWORK})
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)
    set(VAIDYLANG_BLAS PkgConfig::OPENBLAS)
endif()

find_package(nlohmann_json 3.11 CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.12.0
        URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(vaidylang_cpu STATIC native/checkpoint.cpp native/qwen3.cpp native/inkling.cpp)
target_compile_features(vaidylang_cpu PUBLIC cxx_std_20)
set_target_properties(vaidylang_cpu PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_link_libraries(vaidylang_cpu PRIVATE ${VAIDYLANG_BLAS} nlohmann_json::nlohmann_json)
if(APPLE)
    target_sources(vaidylang_cpu PRIVATE native/bf16_projection.cpp)
    target_compile_definitions(vaidylang_cpu PRIVATE ACCELERATE_NEW_LAPACK)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(vaidylang_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(vaidylang-qwen3 native/main.cpp)
target_link_libraries(vaidylang-qwen3 PRIVATE vaidylang_cpu nlohmann_json::nlohmann_json)
set_target_properties(vaidylang-qwen3 PROPERTIES CXX_EXTENSIONS OFF)

add_library(vaidylang_native SHARED native/bindings.cpp native/hyper_connections.cpp native/quantize.cpp)
target_link_libraries(vaidylang_native PRIVATE vaidylang_cpu)
set_target_properties(vaidylang_native PROPERTIES CXX_EXTENSIONS OFF CXX_VISIBILITY_PRESET hidden)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME native_qwen3
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/native/test_qwen3.py $<TARGET_FILE:vaidylang-qwen3>)
    add_test(NAME native_deepseek_cpu
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/native/test_deepseek_cpu.py)
    add_test(NAME native_inkling COMMAND ${Python3_EXECUTABLE} -m unittest -v test_inkling)
    set_tests_properties(native_qwen3 native_deepseek_cpu native_inkling PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR};VAIDYLANG_CPU_LIBRARY=$<TARGET_FILE:vaidylang_native>")
endif()
