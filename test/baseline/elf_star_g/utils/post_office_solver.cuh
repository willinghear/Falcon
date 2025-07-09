//
// Created by lizhzz on 25-7-8.
//

#ifndef POST_OFFICE_SOLVER_CUH
#define POST_OFFICE_SOLVER_CUH


__device__ __constant__ int kPositionLength2Bits[] = {
    0, 0, 1, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6
};


__device__ __constant__ int kPow2z[] = {1, 2, 4, 8, 16, 32};

__device__ int initRoundAndRepresentation(const int *distribution, // 输入：分布数组 (大小 64)
                                          int *representation, // 输出：representation 数组 (大小 64)
                                          int *round, // 输出：round 数组 (大小 64)
                                          int *out_positions // 输出：找到的最佳 positions (大小 64, 空间足够)
);

#endif //POST_OFFICE_SOLVER_CUH
