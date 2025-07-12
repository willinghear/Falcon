//
// Created by lizhzz on 25-7-11.
//

#ifndef ELF_STAR_G_KERNEL_CUH
#define ELF_STAR_G_KERNEL_CUH
#include <cuda/std/cstdint>

__global__ void decompress_kernel(const uint8_t* d_in_data,
                                const size_t* d_in_offsets,
                                double* d_out_data,
                                const size_t* d_out_offsets,
                                int num_chunks);

ssize_t elf_star_encode(double *in, ssize_t len, uint8_t **out, int64_t **out_compressed_lengths,
                        int64_t **out_compressed_offsets, int64_t **out_decompressed_offsets, int *out_num_blocks);
#endif //ELF_STAR_G_KERNEL_CUH
