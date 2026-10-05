# Builds alight, the command-line front end. The Qt GUI is still built with qmake (burnback-3d.pro).
cmake_minimum_required(VERSION 3.16)
project(alight VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/submodules/nlohmann/single_include/nlohmann/json.hpp")
	message(FATAL_ERROR "nlohmann/json is missing: run `git submodule update --init`")
endif()

add_executable(alight cli/alight.cpp src/operations.cpp src/iosystem.cpp src/plotData.cpp)
# single_include comes first: include/nlohmann/json.hpp is a symlink, which a Windows checkout may not resolve
target_include_directories(alight PRIVATE submodules/nlohmann/single_include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(alight PRIVATE ALIGHT_HEADLESS ALIGHT_VERSION="${PROJECT_VERSION}")
if(MSVC)
	target_compile_definitions(alight PRIVATE _USE_MATH_DEFINES NOMINMAX)
elseif(MINGW)
	target_link_options(alight PRIVATE -static)
endif()

install(TARGETS alight RUNTIME DESTINATION bin)

enable_testing()
add_test(NAME cube COMMAND alight ${CMAKE_CURRENT_SOURCE_DIR}/cli/tests/cube.json ${CMAKE_CURRENT_BINARY_DIR}/cube --tol 1e-4 --iters 5000 --expect-max 1.0)
