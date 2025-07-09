//
// Created by lizhzz on 25-7-7.
//

#include "ElfStarComKernel.cuh"

#include <cstdint>
#include "defs.cuh"

#include "BitStream/BitWriter.cuh"
#include "utils/post_office_solver.cuh"

#define CHUNK_SIZE 1024

#define LOG_2_10 3.32192809489

#define F_TABLE_SIZE 64 // 示例大小，请替换为实际大小

#define MAX_CHUNK_BYTES 8192

__global__ void elf_star_compress_kernel(double *d_in, uint8_t *d_out_chunks, int *d_chunk_sizes, int total_values) {
    int chunk_idx = blockIdx.x;
    int start_idx = chunk_idx * CHUNK_SIZE;

    // 如果这个块超出了总数据范围，则直接返回
    if (start_idx >= total_values) {
        return;
    }

    int current_chunk_size = min(CHUNK_SIZE, total_values - start_idx);

    // 状态管理：在栈上创建局部变量，替代C++类的成员变量
    // --- ElfStarCompressor state ---
    size_t total_bits_size = 0;
    int lastBetaStar = 0x7FFFFFFF; // __INT32_MAX__
    int numberOfValues = 0;
    int betaStarList[CHUNK_SIZE];
    uint64_t vPrimeList[CHUNK_SIZE];
    int leadDistribution[64] = {0};
    int trailDistribution[64] = {0};

    // --- ElfStarXORCompressor state ---
    int storedLeadingZeros = 0x7FFFFFFF;
    int storedTrailingZeros = 0x7FFFFFFF;
    uint64_t storedVal = 0;
    bool first = true;
    int leading_representation[64] = {0};
    int trailing_representation[64] = {0};
    int leading_round[64] = {0};
    int trailing_round[64] = {0};
    int leading_bits_per_value;
    int trailing_bits_per_value;

    // 先一遍addValue()
    for (int i = 0; i < current_chunk_size; i++) {
        double v = d_in[start_idx + i];
        union {
            double d;
            long i;
        } data = {.d = v};
        if (v == 0.0 || isinf(v)) {
            vPrimeList[i] = data.i;
            betaStarList[i] = 0x7FFFFFFF; // __INT32_MAX__
        } else if (isnan(v)) {
            vPrimeList[i] = 0xfff8000000000000L & data.i;
            betaStarList[i] = 0x7FFFFFFF; // __INT32_MAX__
        } else {
            int alphaAndBetaStar[2]; // 在栈上安全地创建数组
            getAlphaAndBetaStar(v, lastBetaStar, alphaAndBetaStar);
            int e = ((long) (data.i >> 52)) & 0x7ff;
            int gAlpha = getFAlpha(alphaAndBetaStar[0]) + e - 1023;
            int eraseBits = 52 - gAlpha;

            long mask = 0xffffffffffffffffL << eraseBits;
            long delta = (~mask) & data.i;
            if (delta != 0 && eraseBits > 4) {
                // 更新状态变量 lastBetaStar
                if (alphaAndBetaStar[1] != lastBetaStar) {
                    lastBetaStar = alphaAndBetaStar[1];
                }
                betaStarList[i] = lastBetaStar;
                vPrimeList[i] = mask & data.i;
            } else {
                betaStarList[i] = 0x7FFFFFFF; // __INT32_MAX__
                vPrimeList[i] = data.i;
            }
            // 不需要 delete[] 了！
        }
        numberOfValues++; // 可以在循环后直接使用 current_chunk_size，但这里保留是为了逻辑清晰
    }

    if (numberOfValues > 1) {
        uint64_t lastValue = vPrimeList[0];
        for (int i = 1; i < numberOfValues; i++) {
            uint64_t xor_ = lastValue ^ vPrimeList[i];
            if (xor_ != 0) {
                leadDistribution[__clzll(xor_)]++;
                trailDistribution[__ffsll(xor_) - 1]++;
                lastValue = vPrimeList[i];
            }
        }
    }

    BitWriter writer;
    // uint32_t *my_out_buffer = reinterpret_cast<uint32_t *>(d_out_chunks + chunk_idx * MAX_CHUNK_BYTES);
    // initBitWriter(&writer, my_out_buffer,MAX_CHUNK_BYTES);

    uint8_t* chunk_base_ptr = d_out_chunks + (size_t)chunk_idx * MAX_CHUNK_BYTES;

    // 2. 计算实际写入比特流的起始地址。
    //    这个地址是基地址向后偏移4个字节。
    uint8_t* bitstream_start_ptr = chunk_base_ptr + 4;

    // 3. 将这个新的起始地址转换为 BitWriter 需要的 uint32_t* 类型
    uint32_t* my_out_buffer_for_writing = reinterpret_cast<uint32_t*>(bitstream_start_ptr);

    // 4. 初始化 BitWriter，让它从新的起始地址开始写入。
    //    注意：容量也应该相应地减少4字节。
    initBitWriter(&writer, my_out_buffer_for_writing, MAX_CHUNK_BYTES - 4);


    // 计算positions
    int lead_positions[64], trail_positions[64];
    int lead_positions_len, trail_positions_len;

    lead_positions_len = initRoundAndRepresentation(leadDistribution, leading_representation, leading_round,
                                                    lead_positions);

    leading_bits_per_value = kPositionLength2Bits[lead_positions_len];

    trail_positions_len = initRoundAndRepresentation(trailDistribution, trailing_representation, trailing_round,
                                                     trail_positions);

    trailing_bits_per_value = kPositionLength2Bits[trail_positions_len];

    // 压缩addValue()
    lastBetaStar = 0x7FFFFFFF;
    for (int i = 0; i < numberOfValues; i++) {
        if (betaStarList[i] == INT_MAX) {
            write(&writer, 2, 2);
            total_bits_size += 2;
        } else if (betaStarList[i] == lastBetaStar) {
            write(&writer, 0, 1);
            total_bits_size += 1;
        } else {
            write(&writer, betaStarList[i] | 0x30, 6);
            total_bits_size += 6;
            lastBetaStar = betaStarList[i];
        }
        //==========================================================
        // xor.addValue()
        //==========================================================
        int bits_for_this_value = 0;
        long value = vPrimeList[i];
        if (first) {
            first = false;
            storedVal = value;
            int trailingZeros = __builtin_ctzl(value);
            write(&writer, trailingZeros, 7);
            bits_for_this_value += 7;
            if (value != 0) {
                writeLong(&writer, storedVal >> (trailingZeros + 1), 63 - trailingZeros);
                bits_for_this_value += 63 - trailingZeros;
            }
        } else {
            uint64_t _xor = storedVal ^ value;
            if (_xor == 0) {
                write(&writer, 1, 2);
                bits_for_this_value += 2;
            } else {
                int leading_count = __clzll(_xor);
                int trailing_count = __builtin_ctzll(_xor);

                int leadingZeros = leading_round[leading_count];
                int trailingZeros = trailing_round[trailing_count];
                if (leadingZeros >= storedLeadingZeros && trailingZeros >= storedTrailingZeros &&
                    (leadingZeros - storedLeadingZeros) + (trailingZeros - storedTrailingZeros) < 1 +
                    leading_bits_per_value + trailing_bits_per_value) {
                    //case 1
                    int centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
                    int len = 1 + centerBits;
                    if (len > 64) {
                        write(&writer, 1, 1);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                    } else {
                        writeLong(&writer, (1L << centerBits) | (_xor >> storedTrailingZeros), 1 + centerBits);
                    }
                    bits_for_this_value += len;
                } else {
                    storedLeadingZeros = leadingZeros;
                    storedTrailingZeros = trailingZeros;
                    int centerBits = 64 - storedLeadingZeros - storedTrailingZeros;

                    //case 00
                    int len = 2 + leading_bits_per_value + trailing_bits_per_value +
                              centerBits;
                    if (len > 64) {
                        write(&writer,
                              (leading_representation[storedLeadingZeros]
                               << trailing_bits_per_value) |
                              trailing_representation[storedTrailingZeros],
                              2 + leading_bits_per_value + trailing_bits_per_value);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                    } else {
                        long tmp = ((((uint64_t) leading_representation[storedLeadingZeros]
                                      << trailing_bits_per_value) |
                                     trailing_representation[storedTrailingZeros])
                                    << centerBits) |
                                   (_xor >> storedTrailingZeros);
                        writeLong(&writer,
                                  ((((uint64_t) leading_representation[storedLeadingZeros]
                                     << trailing_bits_per_value) |
                                    trailing_representation[storedTrailingZeros])
                                   << centerBits) |
                                  (_xor >> storedTrailingZeros),
                                  len);
                    }
                    bits_for_this_value += len;
                }
                storedVal = value;
            }
        }
        total_bits_size += bits_for_this_value;
    }
    total_bits_size += flush(&writer);
    uint32_t final_byte_size = (total_bits_size + 7) / 8;

    *((uint32_t*)chunk_base_ptr) = final_byte_size;
}
