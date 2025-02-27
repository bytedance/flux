/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure xor
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "atomic_ping_pong_common.h"

/* add */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, add, (value * (1 + i)),
                                      (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int, int, add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(long, long, add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(size_t, size, add, (value * (1 + i)), (value));

/* fetch_add */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, fetch_add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, fetch_add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, fetch_add, (value * (1 + i)),
                                      (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, fetch_add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, fetch_add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, fetch_add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int, int, fetch_add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(long, long, fetch_add, (value * (1 + i)), (value));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(size_t, size, fetch_add, (value * (1 + i)), (value));

/* and */
/* should get flag set to 0b1, 0b11, 0b111, etc. */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, and, (value << (i + 1)),
                                      (value << (i + 1)));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, and, (value << (i + 1)),
                                      (value << (i + 1)));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, and, (value << (i + 1)),
                                      (value << (i + 1)));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, and, (value << (i + 1)), (value << (i +
 * 1))); */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, and, (value << (i + 1)),
                                      (value << (i + 1)));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, and, (value << (i + 1)),
                                      (value << (i + 1)));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, and, (value << (i + 1)), (value << (i +
 * 1))); */

/* fetch_and */
/* should get flag set to 0b1, 0b11, 0b111, etc. */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, fetch_and, (value << (i + 1)),
                                      (value << (i + 1)));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, fetch_and, (value << (i + 1)),
                                      (value << (i + 1)));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, fetch_and, (value << (i + 1)),
                                      (value << (i + 1)));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, fetch_and, (value << (i + 1)), (value << (i
 * + 1))); */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, fetch_and, (value << (i + 1)),
                                      (value << (i + 1)));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, fetch_and, (value << (i + 1)),
                                      (value << (i + 1)));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, fetch_and, (value << (i + 1)), (value << (i
 * + 1))); */

/* inc */
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned int, uint, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned long, ulong, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned long long, ulonglong, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(int32_t, int32, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(uint32_t, uint32, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(uint64_t, uint64, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(int, int, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(long, long, inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(size_t, size, inc, (i + 1));

/* fetch_inc */
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned int, uint, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned long, ulong, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(unsigned long long, ulonglong, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(int32_t, int32, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(uint32_t, uint32, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(uint64_t, uint64, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(int, int, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(long, long, fetch_inc, (i + 1));
DEFINE_PING_PONG_TEST_FOR_AMO_NO_ARG(size_t, size, fetch_inc, (i + 1));

/* or */
/* should get flag set to 0b1, 0b11, 0b111, etc. */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, or, (cmp >> (iter - (i + 1))),
                                      (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, or, (cmp >> (iter - (i + 1))),
                                      (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, or, (cmp >> (iter - (i + 1))),
                                      (value << i));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, or, (cmp >> (iter - (i + 1))), (value <<
 * i)); */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, or, (cmp >> (iter - (i + 1))),
                                      (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, or, (cmp >> (iter - (i + 1))),
                                      (value << i));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, or, (cmp >> (iter - (i + 1))), (value <<
 * i)); */

/* fetch_or */
/* should get flag set to 0b1, 0b11, 0b111, etc. */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, fetch_or, (cmp >> (iter - (i + 1))),
                                      (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, fetch_or, (cmp >> (iter - (i + 1))),
                                      (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, fetch_or,
                                      (cmp >> (iter - (i + 1))), (value << i));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, fetch_or, (cmp >> (iter - (i + 1))), (value
 * << i)); */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, fetch_or, (cmp >> (iter - (i + 1))),
                                      (value << i));
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, fetch_or, (cmp >> (iter - (i + 1))),
                                      (value << i));
/* DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, fetch_or, (cmp >> (iter - (i + 1))), (value
 * << i)); */

/* xor */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, xor, i % 2, 1);

/* fetch_xor */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, fetch_xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, fetch_xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, fetch_xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, fetch_xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, fetch_xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, fetch_xor, i % 2, 1);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int64_t, int64, fetch_xor, i % 2, 1);

/* set */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int, int, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(long, long, set, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(size_t, size, set, i, i);

/* swap */
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned int, uint, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long, ulong, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(unsigned long long, ulonglong, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int32_t, int32, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint32_t, uint32, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(uint64_t, uint64, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(int, int, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(long, long, swap, i, i);
DEFINE_PING_PONG_TEST_FOR_AMO_ONE_ARG(size_t, size, swap, i, i);

/* compare_swap */
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(unsigned int, uint, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(unsigned long, ulong, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(unsigned long long, ulonglong, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(int32_t, int32, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(uint32_t, uint32, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(uint64_t, uint64, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(int, int, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(long, long, compare_swap, i, i + 1);
DEFINE_PING_PONG_TEST_FOR_AMO_TWO_ARG(size_t, size, compare_swap, i, i + 1);

int main(int c, char *v[]) {
    int mype, npes;
    int iter, skip;
    int rc = 0;

    void *flag_d = NULL;
    cudaStream_t stream;
    nvshmemi_amo_t op;

    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    MAIN_SETUP(c, v, mype, npes, flag_d, stream, h_size_arr, h_tables, h_lat, &op);

    switch (op) {
        case NVSHMEMI_AMO_INC: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITHOUT_ARG(unsigned int, uint, inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(unsigned long, ulong, inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(unsigned long long, ulonglong, inc, flag_d, mype, iter, skip,
                                 h_lat, h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(int32_t, int32, inc, flag_d, mype, iter, skip, h_lat, h_size_arr,
                                 0);
            RUN_TEST_WITHOUT_ARG(uint32_t, uint32, inc, flag_d, mype, iter, skip, h_lat, h_size_arr,
                                 0);
            RUN_TEST_WITHOUT_ARG(uint64_t, uint64, inc, flag_d, mype, iter, skip, h_lat, h_size_arr,
                                 0);
            RUN_TEST_WITHOUT_ARG(int, int, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(long, long, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(size_t, size, inc, flag_d, mype, iter, skip, h_lat, h_size_arr, 0);
            break;
        }
        case NVSHMEMI_AMO_SET: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITH_ARG(unsigned int, uint, set, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 551);
            RUN_TEST_WITH_ARG(unsigned long, ulong, set, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 551);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, set, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 551);
            RUN_TEST_WITH_ARG(int32_t, int32, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415,
                              0, 551);
            RUN_TEST_WITH_ARG(uint32_t, uint32, set, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 551);
            RUN_TEST_WITH_ARG(uint64_t, uint64, set, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 551);
            RUN_TEST_WITH_ARG(int, int, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0,
                              551);
            RUN_TEST_WITH_ARG(long, long, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0,
                              551);
            RUN_TEST_WITH_ARG(size_t, size, set, flag_d, mype, iter, skip, h_lat, h_size_arr, 415,
                              0, 551);
            break;
        }
        case NVSHMEMI_AMO_ADD: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITH_ARG(unsigned int, uint, add, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 0);
            RUN_TEST_WITH_ARG(unsigned long, ulong, add, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, add, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(int32_t, int32, add, flag_d, mype, iter, skip, h_lat, h_size_arr, 415,
                              0, 0);
            RUN_TEST_WITH_ARG(uint32_t, uint32, add, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 0);
            RUN_TEST_WITH_ARG(uint64_t, uint64, add, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 0);
            RUN_TEST_WITH_ARG(int, int, add, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0,
                              0);
            RUN_TEST_WITH_ARG(long, long, add, flag_d, mype, iter, skip, h_lat, h_size_arr, 415, 0,
                              0);
            RUN_TEST_WITH_ARG(size_t, size, add, flag_d, mype, iter, skip, h_lat, h_size_arr, 415,
                              0, 0);
            break;
        }
        case NVSHMEMI_AMO_AND: {
            iter = 64;
            skip = 0;
            /* TODO: Figure out a good way to do this with signed types. The bit shifts we do don't
             * mix with signed types. */
            RUN_TEST_WITH_ARG(unsigned long, ulong, and, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, and, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF);
            RUN_TEST_WITH_ARG(uint64_t, uint64, and, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF);
            /* RUN_TEST_WITH_ARG(int64_t, int64, and, flag_d, mype, iter, skip, h_lat, h_size_arr,
             * 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF); */
            iter = 32;
            /* RUN_TEST_WITH_ARG(int64_t, int64, and, flag_d, mype, iter, skip, h_lat, h_size_arr,
             * 0xFFFFFFFF, 0, 0xFFFFFFFF); */
            RUN_TEST_WITH_ARG(uint32_t, uint32, and, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              0xFFFFFFFF, 0, 0xFFFFFFFF);
            RUN_TEST_WITH_ARG(unsigned int, uint, and, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              0xFFFFFFFF, 0, 0xFFFFFFFF);
            break;
        }
        case NVSHMEMI_AMO_OR: {
            iter = 64;
            skip = 0;
            /* TODO: Figure out a good way to do this with signed types. The bit shifts we do don't
             * mix with signed types. */
            RUN_TEST_WITH_ARG(unsigned long, ulong, or, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              1, 0xFFFFFFFFFFFFFFFF, 0);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, or, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0);
            RUN_TEST_WITH_ARG(uint64_t, uint64, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1,
                              0xFFFFFFFFFFFFFFFF, 0);
            /* RUN_TEST_WITH_ARG(int64_t, int64, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1,
             * 0xFFFFFFFFFFFFFFFF, 0); */
            iter = 32;
            /* RUN_TEST_WITH_ARG(int64_t, int64, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1,
             * 0xFFFFFFFFFFFFFFFF, 0); */
            RUN_TEST_WITH_ARG(uint32_t, uint32, or, flag_d, mype, iter, skip, h_lat, h_size_arr, 1,
                              0xFFFFFFFF, 0);
            RUN_TEST_WITH_ARG(unsigned int, uint, or, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              1, 0xFFFFFFFF, 0);
            break;
        }
        case NVSHMEMI_AMO_XOR: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITH_ARG(unsigned long, ulong, xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(uint64_t, uint64, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0,
                              0, 1);
            RUN_TEST_WITH_ARG(int64_t, int64, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0,
                              0, 1);
            RUN_TEST_WITH_ARG(int64_t, int64, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0,
                              0, 1);
            RUN_TEST_WITH_ARG(uint32_t, uint32, xor, flag_d, mype, iter, skip, h_lat, h_size_arr, 0,
                              0, 1);
            RUN_TEST_WITH_ARG(unsigned int, uint, xor, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              0, 0, 1);
            break;
        }
        case NVSHMEMI_AMO_FETCH_INC: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITHOUT_ARG(unsigned int, uint, fetch_inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(unsigned long, ulong, fetch_inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(unsigned long long, ulonglong, fetch_inc, flag_d, mype, iter, skip,
                                 h_lat, h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(int32_t, int32, fetch_inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(uint32_t, uint32, fetch_inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(uint64_t, uint64, fetch_inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            RUN_TEST_WITHOUT_ARG(int, int, fetch_inc, flag_d, mype, iter, skip, h_lat, h_size_arr,
                                 0);
            RUN_TEST_WITHOUT_ARG(long, long, fetch_inc, flag_d, mype, iter, skip, h_lat, h_size_arr,
                                 0);
            RUN_TEST_WITHOUT_ARG(size_t, size, fetch_inc, flag_d, mype, iter, skip, h_lat,
                                 h_size_arr, 0);
            break;
        }
        case NVSHMEMI_AMO_FETCH_ADD: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITH_ARG(unsigned int, uint, fetch_add, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(unsigned long, ulong, fetch_add, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, fetch_add, flag_d, mype, iter, skip,
                              h_lat, h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(int32_t, int32, fetch_add, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(uint32_t, uint32, fetch_add, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(uint64_t, uint64, fetch_add, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(int, int, fetch_add, flag_d, mype, iter, skip, h_lat, h_size_arr, 415,
                              0, 0);
            RUN_TEST_WITH_ARG(long, long, fetch_add, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 0);
            RUN_TEST_WITH_ARG(size_t, size, fetch_add, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 0);
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND: {
            iter = 64;
            skip = 0;
            /* TODO: Figure out a good way to do this with signed types. The bit shifts we do don't
             * mix with signed types. */
            RUN_TEST_WITH_ARG(unsigned long, ulong, fetch_and, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, fetch_and, flag_d, mype, iter, skip,
                              h_lat, h_size_arr, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF);
            RUN_TEST_WITH_ARG(uint64_t, uint64, fetch_and, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF);
            /* RUN_TEST_WITH_ARG(int64_t, int64, fetch_and, flag_d, mype, iter, skip, h_lat,
             * h_size_arr, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF); */
            iter = 32;
            /* RUN_TEST_WITH_ARG(int64_t, int64, fetch_and, flag_d, mype, iter, skip, h_lat,
             * h_size_arr, 0xFFFFFFFF, 0, 0xFFFFFFFF); */
            RUN_TEST_WITH_ARG(uint32_t, uint32, fetch_and, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0xFFFFFFFF, 0, 0xFFFFFFFF);
            RUN_TEST_WITH_ARG(unsigned int, uint, fetch_and, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0xFFFFFFFF, 0, 0xFFFFFFFF);
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR: {
            iter = 64;
            skip = 0;
            /* TODO: Figure out a good way to do this with signed types. The bit shifts we do don't
             * mix with signed types. */
            RUN_TEST_WITH_ARG(unsigned long, ulong, fetch_or, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, fetch_or, flag_d, mype, iter, skip,
                              h_lat, h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0);
            RUN_TEST_WITH_ARG(uint64_t, uint64, fetch_or, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0);
            /* RUN_TEST_WITH_ARG(int64_t, int64, fetch_or, flag_d, mype, iter, skip, h_lat,
             * h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0); */
            iter = 32;
            /* RUN_TEST_WITH_ARG(int64_t, int64, fetch_or, flag_d, mype, iter, skip, h_lat,
             * h_size_arr, 1, 0xFFFFFFFFFFFFFFFF, 0); */
            RUN_TEST_WITH_ARG(uint32_t, uint32, fetch_or, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 1, 0xFFFFFFFF, 0);
            RUN_TEST_WITH_ARG(unsigned int, uint, fetch_or, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 1, 0xFFFFFFFF, 0);
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITH_ARG(unsigned long, ulong, fetch_xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, fetch_xor, flag_d, mype, iter, skip,
                              h_lat, h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(uint64_t, uint64, fetch_xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(int64_t, int64, fetch_xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(int64_t, int64, fetch_xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(uint32_t, uint32, fetch_xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(unsigned int, uint, fetch_xor, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            break;
        }
        case NVSHMEMI_AMO_SWAP: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITH_ARG(unsigned int, uint, swap, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              0, 0, 1);
            RUN_TEST_WITH_ARG(unsigned long, ulong, swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(int32_t, int32, swap, flag_d, mype, iter, skip, h_lat, h_size_arr, 0,
                              0, 1);
            RUN_TEST_WITH_ARG(uint32_t, uint32, swap, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              0, 0, 1);
            RUN_TEST_WITH_ARG(uint64_t, uint64, swap, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              0, 0, 1);
            RUN_TEST_WITH_ARG(int, int, swap, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0, 1);
            RUN_TEST_WITH_ARG(long, long, swap, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0,
                              1);
            RUN_TEST_WITH_ARG(size_t, size, swap, flag_d, mype, iter, skip, h_lat, h_size_arr, 0, 0,
                              1);
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            iter = 500;
            skip = 50;
            RUN_TEST_WITH_ARG(unsigned int, uint, compare_swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(unsigned long, ulong, compare_swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(unsigned long long, ulonglong, compare_swap, flag_d, mype, iter, skip,
                              h_lat, h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(int32_t, int32, compare_swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(uint32_t, uint32, compare_swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(uint64_t, uint64, compare_swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            RUN_TEST_WITH_ARG(int, int, compare_swap, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 0);
            RUN_TEST_WITH_ARG(long, long, compare_swap, flag_d, mype, iter, skip, h_lat, h_size_arr,
                              415, 0, 0);
            RUN_TEST_WITH_ARG(size_t, size, compare_swap, flag_d, mype, iter, skip, h_lat,
                              h_size_arr, 415, 0, 0);
            break;
        }
        default: {
            fprintf(stderr, "Error, unsupported Atomic op %d.\n", op);
            rc = -1;
            break;
        }
    }

    MAIN_CLEANUP(flag_d, stream, h_tables, 2);
    return rc;
}