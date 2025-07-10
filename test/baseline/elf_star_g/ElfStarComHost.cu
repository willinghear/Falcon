//
// Created by lizhzz on 25-7-10.
//

#include "ElfStarComHost.cuh"

#include <cstdint>
#include <iostream>
#include <ostream>
#include <vector>

#include "ElfStarComKernel.cuh"
#include <cub/cub.cuh> // 引入 CUB

#define CHUNK_SIZE 1024
#define MAX_CHUNK_BYTES 8196

#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
__FILE__, __LINE__, cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while (0)

ssize_t elf_star_encode(double *in, ssize_t len, uint8_t **out) {
    if (len <= 0) {
        *out = nullptr;
        return 0;
    }

    double *d_in;
    uint8_t *d_out_chunks;
    int *d_chunk_sizes;

    int num_chunks = (len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    cudaMalloc(&d_in, len * sizeof(double));
    cudaMalloc(&d_out_chunks, num_chunks * MAX_CHUNK_BYTES);
    cudaMalloc(&d_chunk_sizes, num_chunks * sizeof(int));

    cudaMemcpy(d_in, in, len * sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridDim(num_chunks,1,1);
    dim3 blockDim(1,1,1);

    std::cout << "Launching " << num_chunks << " CUDA blocks..." << std::endl;

    elf_star_compress_kernel<<<gridDim,blockDim>>>(d_in, d_out_chunks, d_chunk_sizes, len);
    //
    // // 计算前缀和
    // int* d_chunk_offsets; // 存储每个块的起始偏移
    // cudaMalloc(&d_chunk_offsets, num_chunks * sizeof(int));
    //
    // // 使用 CUB 执行 Exclusive Scan (前缀和)
    // void* d_temp_storage = nullptr;
    // size_t temp_storage_bytes = 0;
    //
    // // 第一次调用：获取执行所需的临时存储空间大小
    // cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_chunk_sizes, d_chunk_offsets, num_chunks);
    // // 分配临时存储空间
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // // 第二次调用：真正执行前缀和
    // cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_chunk_sizes, d_chunk_offsets, num_chunks);
    //
    //
    // // --- 步骤 5: 计算总大小并分配最终的连续缓冲区 ---
    // int total_compressed_size = 0;
    // if (num_chunks > 0) {
    //     // 总大小 = 最后一个块的偏移 + 最后一个块的大小
    //     // 我们需要从GPU把这两个值拷贝回来
    //     int last_chunk_offset, last_chunk_size;
    //     cudaMemcpy(&last_chunk_offset, d_chunk_offsets + num_chunks - 1, sizeof(int), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(&last_chunk_size, d_chunk_sizes + num_chunks - 1, sizeof(int), cudaMemcpyDeviceToHost);
    //     total_compressed_size = last_chunk_offset + last_chunk_size;
    // }
    // uint8_t* d_out_contiguous = nullptr;
    // if (total_compressed_size > 0) {
    //     cudaMalloc(&d_out_contiguous, total_compressed_size);
    // }
    //
    // // --- 步骤 6: 启动“整理”内核，执行并行拷贝 ---
    // if (total_compressed_size > 0) {
    //     // 配置内核启动参数
    //     // 我们可以使用更多的线程来加速拷贝
    //     int threads_per_block_assemble = 256;
    //     int blocks_per_grid_assemble = (num_chunks + threads_per_block_assemble - 1) / threads_per_block_assemble;
    //
    //     assemble_chunks_kernel<<<blocks_per_grid_assemble, threads_per_block_assemble>>>(
    //         d_out_chunks, d_out_contiguous, d_chunk_sizes, d_chunk_offsets, num_chunks
    //     );
    //     CUDA_CHECK(cudaGetLastError());
    // }


    std::vector<int> h_chunk_sizes(num_chunks);
    std::vector<uint8_t> h_out_chunks((size_t)num_chunks * MAX_CHUNK_BYTES);

    cudaDeviceSynchronize();

    std::cout << "GPU compression finished." << std::endl;

    // 复制每个块的压缩后大小
    CUDA_CHECK(cudaMemcpy(h_chunk_sizes.data(), d_chunk_sizes, num_chunks * sizeof(int), cudaMemcpyDeviceToHost));

    // 复制所有压缩数据块
    CUDA_CHECK(cudaMemcpy(h_out_chunks.data(), d_out_chunks, (size_t)num_chunks * MAX_CHUNK_BYTES, cudaMemcpyDeviceToHost));


    std::cout << "Assembling final compressed data on host..." << std::endl;

    // 计算压缩后的总大小
    size_t total_compressed_size = 0;
    for (int size : h_chunk_sizes) {
        total_compressed_size += size;
    }


    // 分配最终的、连续的输出缓冲区
    *out = (uint8_t*)malloc(total_compressed_size);
    if (total_compressed_size > 0 && *out == nullptr) {
        fprintf(stderr, "Error: Failed to allocate memory for final output.\n");
        // 清理GPU内存
        cudaFree(d_in);
        cudaFree(d_out_chunks);
        cudaFree(d_chunk_sizes);
        return -1; // 表示错误
    }

    // 将各个有效的压缩块拼接到最终输出缓冲区
    uint8_t* current_pos = *out;
    for (int i = 0; i < num_chunks; ++i) {
        int chunk_size = h_chunk_sizes[i];
        if (chunk_size > 0) {
            // 源地址：指向第i个块的起始位置
            uint8_t* chunk_start = h_out_chunks.data() + (size_t)i * MAX_CHUNK_BYTES;
            // 复制数据
            memcpy(current_pos, chunk_start, chunk_size);
            // 移动目标指针
            current_pos += chunk_size;
        }
    }

    // ===================================================================
    // 6. 释放GPU内存
    // ===================================================================
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_chunks));
    CUDA_CHECK(cudaFree(d_chunk_sizes));

    std::cout << "Compression successful. Original size: " << len * sizeof(double)
              << " bytes, Compressed size: " << total_compressed_size << " bytes." << std::endl;

    return total_compressed_size;
}


int main() {
    // 1. 准备测试数据
    const ssize_t data_len = 5000; // 测试用例大小
    std::vector<double> host_data(data_len);

    // 生成一些有规律的、可压缩的数据
    for (ssize_t i = 0; i < data_len; ++i) {
        host_data[i] = 1.2345678e10 + (double)(i % 100);
    }

    // 2. 调用压缩函数
    uint8_t* compressed_data = nullptr;
    ssize_t compressed_size = elf_star_encode(host_data.data(), data_len, &compressed_data);

    if (compressed_size >= 0) {
        std::cout << "Final compressed size: " << compressed_size << " bytes." << std::endl;
        double ratio = (double)compressed_size / (data_len * sizeof(double));
        std::cout << "Compression ratio: " << ratio << std::endl;

        // 在这里可以使用 compressed_data，例如写入文件
        // FILE *fp = fopen("compressed.bin", "wb");
        // if(fp) {
        //     fwrite(compressed_data, 1, compressed_size, fp);
        //     fclose(fp);
        // }

    } else {
        std::cerr << "Compression failed." << std::endl;
    }

    // 3. 清理压缩后分配的内存
    if (compressed_data != nullptr) {
        free(compressed_data);
    }

    return 0;
}