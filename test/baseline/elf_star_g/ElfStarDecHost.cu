//
// Created by lizhzz on 25-7-10.
//

#include "ElfStarDecHost.cuh"

#include <vector>

#include "BitStream/BitReader.cuh"

#include "Elf_Star_g_Kernel.cuh"

ssize_t elf_star_decode(const uint8_t *all_in_data,
                        const size_t *in_offsets_bytes,
                        const size_t *in_lengths_bytes,
                        double *all_out_data,
                        const size_t *out_offsets,
                        int num_blocks) {
    if (num_blocks <= 0) {
        return 0;
    }
    // a. 计算总输入大小。它由偏移量数组的最后一个元素给出。
    const int64_t total_in_bytes = in_offsets_bytes[num_blocks];

    // b. 计算总输出大小。这一步仍然是必须的，我们需要预读每个块的头部来确定解压后的大小。
    int64_t total_out_elements = 0;
    for (int i = 0; i < num_blocks; ++i) {
        // 定位到第 i 个块的起始位置
        const uint8_t *chunk_start = all_in_data + in_offsets_bytes[i];
        // 读取前4个字节作为解压后的元素数量 (double 的数量)
        total_out_elements += *(reinterpret_cast<const uint32_t *>(chunk_start));
    }

    const int64_t total_out_bytes = total_out_elements * sizeof(double);

    uint8_t *d_in_data = nullptr;
    double *d_out_data = nullptr;
    size_t *d_in_offsets = nullptr;
    size_t *d_out_offsets = nullptr;

    cudaMalloc(&d_in_data, total_in_bytes);
    cudaMalloc(&d_out_data, total_out_bytes);
    cudaMalloc(&d_in_offsets, (num_blocks + 1) * sizeof(size_t));
    cudaMalloc(&d_out_offsets, num_blocks * sizeof(size_t));

    cudaMemcpy(d_in_data, all_in_data, total_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_offsets, in_offsets_bytes, (num_blocks + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_offsets, out_offsets, num_blocks * sizeof(size_t), cudaMemcpyHostToDevice);
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    decompress_kernel<<<blocks_per_grid, threads_per_block>>>(d_in_data, d_in_offsets, d_out_data, d_out_offsets, num_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(all_out_data, d_out_data, total_out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in_data);
    cudaFree(d_out_data);
    cudaFree(d_in_offsets);
    cudaFree(d_out_offsets);

    return total_out_bytes;

}
