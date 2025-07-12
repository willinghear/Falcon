//
// Created by lizhzz on 25-7-8.
//

#ifndef DEFS_CUH
#define DEFS_CUH
#include <cstdint>

#endif //DEFS_CUH

union DOUBLE {
    double d;
    uint64_t i;
};

union FLOAT {
    float f;
    uint32_t i;
};

// Utils
__device__ int getFAlpha(int alpha);
__device__ void getAlphaAndBetaStar(double v, int lastBetaStar, int result[2]);

__device__ double roundUp(double v, int alpha);
__device__ double get10iN(int i);
__device__ int getSP(double v);

__device__ int getSignificantCount(double v, int sp, int lastBetaStar)

__device__ double get10iP(int i)

__device__ void getSPAnd10iNFlag(double v, int result_sp_flag[2])