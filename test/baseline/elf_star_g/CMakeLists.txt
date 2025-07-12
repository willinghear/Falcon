cmake_minimum_required(VERSION 3.20)

project(Elf_star_g LANGUAGES CXX CUDA)

# Set C++ standard version
set(CMAKE_CXX_STANDARD 17)

# -O3 Optimization for release version
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Set parallel compilation level as 4
set(CMAKE_BUILD_PARALLEL_LEVEL 4)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(ELF_STAR_SOURCES
        ElfStarComHost.cu
        ElfStarComKernel.cu
        utils.cu
        utils/post_office_solver.cu
        ElfStarDecKernel.cu
        ElfStarDecHost.cu
        ElfStarDecHost.cuh
        Elf_Star_g_Kernel.cuh
        # 如果你有其他的 .cpp 文件，也在这里添加
)

add_library(elf_star_g SHARED ${ELF_STAR_SOURCES}
        ElfStarDecKernel.cu
)

# --- 最佳实践：使用 target_include_directories ---
# 这会告诉 'elf_star_g' 这个目标去哪里寻找它需要的头文件。
target_include_directories(elf_star_g PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/BitStream
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# --- 最佳实践：链接必要的库 ---
# 将CUDA运行时库链接到你的目标
target_link_libraries(elf_star_g PRIVATE CUDA::cudart)