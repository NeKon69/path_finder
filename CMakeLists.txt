cmake_minimum_required(VERSION 4.0)
project(path_finder LANGUAGES CXX CUDA)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "120")
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
find_package(CUDAToolkit REQUIRED)

message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")

include(FetchContent)
FetchContent_Declare(
        cuda_wrappers
        GIT_REPOSITORY https://github.com/NeKon69/cuda_wrappers
        GIT_TAG main
)
add_library(cuda_wrappers INTERFACE)
find_package(CUDAToolkit REQUIRED)

#file(MAKE_DIRECTORY ${CUDA_WRAPPERS_DESTINATION})
#file(COPY ${CUDA_WRAPPERS_SOURCE} DESTINATION ${CUDA_WRAPPERS_DESTINATION})



file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/Zi>
        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0>

        $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-O0>

        $<$<CONFIG:Release>:-O3>
        $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/Ox>


        $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:GNU>>:-O2 -g>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:Clang>>:-O2 -g>
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<CXX_COMPILER_ID:MSVC>>:/O2 /Zi>

        $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:GNU>>:-Os>
        $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:Clang>>:-Os>
        $<$<AND:$<CONFIG:MinSizeRel>,$<CXX_COMPILER_ID:MSVC>>:/Os>
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(cuda_wrappers INTERFACE ${cuda_wrappers_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_wrappers CUDA::cudart)
