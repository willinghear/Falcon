//
// Created by lizhzz on 25-7-7.
//

#ifndef ELFSTARCOMKERNEL_CUH
#define ELFSTARCOMKERNEL_CUH
#include <cstdint>



__global__ void elf_star_compress_kernel(double *d_in, uint8_t *d_out_chunks, int *d_chunk_sizes, int total_values);



#endif //ELFSTARCOMKERNEL_CUH
