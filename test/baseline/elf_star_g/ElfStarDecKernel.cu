//
// Created by lizhzz on 25-7-10.
//

#include "Elf_Star_g_Kernel.cuh"

#include <array.h>
#include <BitReader.cuh>
#include <defs.cuh>
#include <post_office_solver.cuh>
#include <cuda/std/cstdint>

typedef union {
    double d;
    uint64_t i;
} DOUBLE;

class ElfStarXORDecompressor_GPU {
private:
    DOUBLE storedVal = {.i = 0};
    int storedLeadingZeros = INT_MAX;
    int storedTrailingZeros = INT_MAX;
    bool first = true;
    bool endOfStream = false;
    BitReader reader;

    int leadingRepresentation[64];
    int trailingRepresentation[64];
    int leadingRepresentationSize;
    int trailingRepresentationSize;

    int leadingBitsPerValue;
    int trailingBitsPerValue;

    __device__ __forceinline__ int read_int(int length) { return readInt(&reader, length); }
    __device__ __forceinline__ int read_bit() { return readInt(&reader, 1); }
    __device__ __forceinline__ uint64_t read_long(int length) { return readLong(&reader, length); }

    __device__ __forceinline__ void initLeadingRepresentation() {
        int num = read_int(5);
        if (num == 0) {
            num = 32;
        }
        leadingBitsPerValue = kPositionLength2Bits[num];
        leadingRepresentationSize = num;
        for (int i = 0; i < num; i++) {
            leadingRepresentation[i] = read_int(6);
        }
    }

    __device__ __forceinline__ void initTrailingRepresentation() {
        int num = read_int(5);
        if (num == 0) {
            num = 32;
        }
        trailingBitsPerValue = kPositionLength2Bits[num];
        trailingRepresentationSize = num;
        for (int i = 0; i < num; i++) {
            trailingRepresentation[i] = read_int(6);
        }
    }

    __device__ __forceinline__ void next() {
        if (first) {
            initLeadingRepresentation();
            initTrailingRepresentation();
            first = false;
            int trailingZeros = read_int(7);
            if (trailingZeros < 64) {
                storedVal.i = ((read_long(63 - trailingZeros) << 1) + 1) << trailingZeros;
            } else {
                storedVal.i = 0;
            }
            if (isnan(storedVal.d)) {
                endOfStream = true;
            }
        } else {
            nextValue();
        }
    }

    __device__ __forceinline__ void nextValue() {
        DOUBLE value;
        int centerBits;

        if (read_bit() == 1) {
            // case 1
            centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
            value.i = read_long(centerBits) << storedTrailingZeros;
            value.i = storedVal.i ^ value.i;
            if (isnan(value.d)) {
                endOfStream = true;
            } else {
                storedVal = value;
            }
        } else if (read_bit() == 0) {
            // case  00
            int leadAndTrail = read_int(leadingBitsPerValue + trailingBitsPerValue);
            int lead = leadAndTrail >> trailingBitsPerValue;
            int trail = ~(0xffffffff << trailingBitsPerValue) & leadAndTrail;

            storedLeadingZeros = leadingRepresentation[lead];
            storedTrailingZeros = trailingRepresentation[trail];
            centerBits = 64 - storedLeadingZeros - storedTrailingZeros;

            value.i = read_long(centerBits) << storedTrailingZeros;
            value.i = storedVal.i ^ value.i;
            if (isnan(value.d)) {
                endOfStream = true;
            } else {
                storedVal = value;
            }
        }
    }

public:
    size_t length = 0;

    __device__ __forceinline__ void init(uint32_t *in, size_t len) {
        initBitReader(&reader, in + 1, len - 1);
        length = in[0];
    }

    __device__ __forceinline__ double readValue() {
        next();
        if (endOfStream) {
            return -1;
        }
        return storedVal.d;
    }

    __device__ __forceinline__ BitReader *getReader() {
        return &reader;
    }
};


class ElfStarDecompressor_GPU {
private:
    ElfStarXORDecompressor_GPU xorDecompressor;
    int lastBetaStar = INT_MAX;

    __device__ __forceinline__ double recoverVByBetaStar() {
        double v;
        double vPrime = xorDecompressor.readValue();
        int sp = getSP(abs(vPrime));
        if (lastBetaStar == 0) {
            v = get10iN(-sp - 1);
            if (vPrime < 0) {
                v = -v;
            }
        } else {
            int alpha = lastBetaStar - sp - 1;
            v = roundUp(vPrime, alpha);
        }
        return v;
    }

    __device__ __forceinline__ double nextValue() {
        double v;
        if (read_int(1) == 0) {
            // case 0
            v = recoverVByBetaStar();
        } else if (read_int(1) == 0) {
            // case 10
            v = xorDecompressor.readValue();
        } else {
            // case 11
            lastBetaStar = read_int(4);
            v = recoverVByBetaStar();
        }
        return v;
    }

protected:
    __device__ int read_int(int len) {
        int res = readInt(xorDecompressor.getReader(), len);
        return res;
    }
    __device__ int getLength() {
        return xorDecompressor.length;
    }

public:
    __device__ void init(uint32_t *in, size_t len) { xorDecompressor.init(in, len); }


    __device__ int decompress(double *output) {
        int len = getLength();
        for (int i = 0; i < len; i++) {
            if (i == 4219) {
                asm("nop");
            }
            output[i] = nextValue();
        }
        return len;
    }

    // __device__ void refresh() {
    //     lastBetaStar = __INT32_MAX__;
    //     xorDecompressor.refresh();
    // }
};

__device__ void decompress_method(uint8_t *d_in, ssize_t len, double *d_out_chunks) {
    ElfStarDecompressor_GPU decompressor;
    decompressor.init((uint32_t*)d_in, len / 4);
    decompressor.decompress(d_out_chunks);
}


__global__ void decompress_kernel(const uint8_t* d_in_data,
                                const size_t* d_in_offsets,
                                double* d_out_data,
                                const size_t* d_out_offsets,
                                int num_chunks) {
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (chunk_idx >= num_chunks) {
        return;
    }

    // 3. 确定当前线程负责的输入数据块的指针和长度。
    // 第 i 个块的长度等于 d_in_offsets[i+1] - d_in_offsets[i]。
    const size_t in_offset_start = d_in_offsets[chunk_idx];
    const size_t in_offset_end   = d_in_offsets[chunk_idx + 1];

    // 指向当前线程要处理的压缩数据的起始位置。
    // 这是安全的，因为函数实际上只会从输入缓冲区读取。
    uint8_t* p_in_chunk = const_cast<uint8_t*>(d_in_data + in_offset_start);
    const ssize_t in_chunk_len_bytes = in_offset_end - in_offset_start;

    // 4. 确定当前线程的输出位置。
    const size_t out_offset_start = d_out_offsets[chunk_idx];
    double* p_out_chunk = d_out_data + out_offset_start;

    // 5. 调用设备端辅助函数，传入计算好的指针，执行单个数据块的解压。
    // 所有复杂的解压逻辑都封装在这个函数内部。
    decompress_method(p_in_chunk, in_chunk_len_bytes, p_out_chunk);
}
