cmake_minimum_required(VERSION 3.20)

project(Elf_star_g LANGUAGES CXX CUDA)

# Set C++ standard version
set(CMAKE_CXX_STANDARD 17)

# -O3 Optimization for release version
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Set parallel compilation level as 4
set(CMAKE_BUILD_PARALLEL_LEVEL 4)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE LIB_SRC *.cpp *.c *.cc *.cu)

add_library(elf_star_g SHARED ${LIB_SRC}
        utils.cu
        defs.cuh
        BitStream/BitDefine.cuh
        BitStream/BitWriter.cuh
        BitStream/BitReader.cuh
        utils/post_office_solver.cu
        utils/post_office_solver.cuh
)
