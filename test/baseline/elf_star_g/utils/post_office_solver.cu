//
// Created by lizhzz on 25-7-8.
//

#include "post_office_solver.cuh"

#include "BitStream/BitWriter.cuh"

__device__ int initRoundAndRepresentation(const int *distribution, // 输入：分布数组 (大小 64)
                                          int *representation, // 输出：representation 数组 (大小 64)
                                          int *round, // 输出：round 数组 (大小 64)
                                          int *out_positions // 输出：找到的最佳 positions (大小 64, 空间足够)
) {
    int pre_non_zeros_count[64];
    // 当前后面的非零个数（不包括当前）
    int post_non_zeros_count[64];


    int non_zeros_count = 64;
    int total_count = distribution[0];

    //=======================================================
    // CalTotalCountAndNonZerosCounts(distribution, pre_non_zeros_count, post_non_zeros_count);
    // START
    //=======================================================

    pre_non_zeros_count[0] = 1;
    for (int i = 0; i < 64; ++i) {
        total_count += distribution[i];
        int magic_code = (distribution[i] == 0);
        non_zeros_count -= magic_code;
        pre_non_zeros_count[i] = pre_non_zeros_count[i - 1] + !magic_code;
    }
    for (int i = 0; i < 64; ++i) {
        post_non_zeros_count[i] = non_zeros_count - pre_non_zeros_count[i];
    }

    //=======================================================
    // CalTotalCountAndNonZerosCounts(distribution, pre_non_zeros_count, post_non_zeros_count);
    // END
    //=======================================================

    int max_z = min((int) kPositionLength2Bits[non_zeros_count], 5);
    int total_cost = 0x7FFFFFFF;

    int best_positions_len = 0;
    int present_cost;
    for (int z = 0; z <= max_z && (present_cost = total_count * z) < total_cost; ++z) {
        int num = kPow2z[z];

        //===================================================
        // BuildPostOffice(distribution, num, total_count_and_non_zeros_counts[1],
        //                 pre_non_zeros_count, post_non_zeros_count)；
        //===================================================

        int temp_total_app_cost; // 对应POR.total_app_cost
        int temp_positions[64]; // 对应POR.office_positions
        int temp_positions_len; // 对应POR.office_positions.len

        int original_num = num;
        num = min(num, non_zeros_count);

        int dp[64][num];
        int pre[64][num];

        dp[0][0] = 0;
        pre[0][0] = -1;

        for (int i = 1; i <= 64; ++i) {
            if (distribution[i] == 0) {
                continue;
            }
            for (int j = max(1, num + i - 64); j <= i && j < num; j++) {
                if (i > 1 && j == 1) {
                    dp[i][j] = 0;
                    for (int k = 1; k < i; k++) {
                        dp[i][j] += distribution[k] * k;
                    }
                    pre[i][j] = 0;
                } else {
                    if (pre_non_zeros_count[i] < j + 1 || post_non_zeros_count[i] < num - 1 - j) {
                        continue;
                    }
                    int app_cost = 0x7FFFFFFF;
                    int pre_k = 0;
                    for (int k = j - 1; k <= i - 1; ++k) {
                        if (distribution[k] == 0 && k > 0 || post_non_zeros_count[k] < j || post_non_zeros_count[k] <
                            num - j) {
                            continue;
                        }
                        int sum = dp[k][j - 1];
                        for (int p = k + 1; p <= i - 1; ++p) {
                            sum += distribution[p] * (p - k);
                        }
                        if (app_cost > sum) {
                            app_cost = sum;
                            pre_k = k;
                            if (sum == 0) {
                                break;
                            }
                        }
                    }
                    if (app_cost != 0x7FFFFFFF) {
                        dp[i][j] = app_cost;
                        pre[i][j] = pre_k;
                    }
                }
            }
        }
        temp_total_app_cost = 0x7FFFFFFF;
        int temp_best_last = 0x7FFFFFFF;
        for (int i = num - 1; i < 64; ++i) {
            if (num - 1 == 0 && i > 0) {
                break;
            }
            if (distribution[i] == 0 && i > 0 || pre_non_zeros_count[i] < num) {
                continue;
            }
            int sum = dp[i][num - 1];
            for (int j = i + 1; j < 64; ++j) {
                sum += distribution[j] * (j - i);
            }
            if (temp_total_app_cost > sum) {
                temp_total_app_cost = sum;
                temp_best_last = i;
            }
        }

        int office_positions[num];
        int i = 1;
        while (temp_best_last != -1) {
            office_positions[num - i] = temp_best_last;
            temp_best_last = pre[temp_best_last][num - i];
            ++i;
        }

        if (original_num > non_zeros_count) {
            int modifying_office_positions[original_num];
            int j = 0, k = 0;
            while (j < original_num && k < num) {
                if (j - k < original_num - num && j < office_positions[k]) {
                    modifying_office_positions[j] = j;
                    ++j;
                } else {
                    modifying_office_positions[j] = office_positions[k];
                    ++j;
                    ++k;
                }
            }
            for (int i = 0; i < original_num; ++i) {
                temp_positions[i] = modifying_office_positions[i];
            }
            temp_positions_len = original_num;
        } else {
            // 如果不需要扩展，直接复制DP的结果
            temp_positions_len = num;
            for (int i = 0; i < num; ++i) {
                temp_positions[i] = office_positions[i];
            }
        }
        int temp_total_cost = temp_total_app_cost + present_cost;
        if (temp_total_cost < total_cost) {
            total_cost = temp_total_cost;
            best_positions_len = temp_positions_len;
            for (int i=0;i<best_positions_len;++i) {
                out_positions[i] = temp_positions[i];
            }
        }

    }

    representation[0] = 0;
    round[0] = 0;
    int i = 1;
    for (int j = 1; j < 64; ++j) {
        // Bad but useful code
        int magic_code = (i < best_positions_len && j == out_positions[i]);
        representation[j] = representation[j - 1] + magic_code;
        round[j] = magic_code ? j : round[j - 1];
        i += magic_code;
        //    if (i < positions.length() && j == positions[i]) {
        //      representation[j] = representation[j - 1] + 1;
        //      round[j] = j;
        //      ++i;
        //    } else {
        //      representation[j] = representation[j - 1];
        //      round[j] = round[j - 1];
        //    }
    }
    return best_positions_len; // 返回 positions 的有效长度
}

__forceinline__ __device__ int write_positions_device(
    BitWriter* writer,
    const int* positions,
    int positions_len
) {
    write(writer, positions_len, 5);
    int this_size = 5;
    for (int i = 0; i < positions_len; i++) {
        write(writer, positions[i], 6);
        this_size += 6;
    }
    return this_size;
}
