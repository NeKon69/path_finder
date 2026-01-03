cmake_minimum_required(VERSION 3.20)

project(path_finder LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES "120")

find_package(CUDAToolkit REQUIRED)

message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")

include(FetchContent)

# when no network!!!!!!1
set(FETCHCONTENT_FULLY_DISCONECTED OFF CACHE BOOL "No network" FORCE)
set(FETCHCONTENT_UPDATES_DISCONECTED OFF CACHE BOOL "No network" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED OFF CACHE BOOL "No network" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED OFF CACHE BOOL "No network" FORCE)

FetchContent_Declare(
  cuda_wrappers
  GIT_REPOSITORY https://github.com/NeKon69/cuda_wrappers
  GIT_TAG main)

FetchContent_Declare(
  FastNoiseLiteCUDA
  GIT_REPOSITORY https://github.com/NeKon69/FastNoiseLiteCUDA
  GIT_TAG main)

FetchContent_MakeAvailable(FastNoiseLiteCUDA)

add_library(cuda_wrappers INTERFACE)

set(NOISE_HEADER_SOURCE
    ${fastnoiselitecuda_SOURCE_DIR}/include/FastNoiseLiteCUDA.h)
set(NOISE_HEADER_DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/noise)

file(MAKE_DIRECTORY ${NOISE_HEADER_DESTINATION})
file(COPY ${NOISE_HEADER_SOURCE} DESTINATION ${NOISE_HEADER_DESTINATION})

file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

target_compile_options(
  ${PROJECT_NAME}
  PRIVATE $<$<CONFIG:Debug>:-g>
          $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Zi>
          $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0>
          $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-O0>
          $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
          $<$<CONFIG:Release>:-O3>
          $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/Ox>
          $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:GNU>>:-O2,-g>
          $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:Clang>>:-O2,-g>
          $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:MSVC>>:/O2,/Zi>
          $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>
          $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:GNU>>:-Os>
          $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:Clang>>:-Os>
          $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:MSVC>>:/Os>)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(cuda_wrappers
                           INTERFACE ${cuda_wrappers_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_wrappers CUDA::cudart)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    add_custom_target(
    copy_compile_commands ALL
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_BINARY_DIR}/compile_commands.json
      ${CMAKE_SOURCE_DIR}/compile_commands.json
    COMMENT "Copying
 compile_commands.json to source dir")
endif()

set_target_properties(path_finder PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
