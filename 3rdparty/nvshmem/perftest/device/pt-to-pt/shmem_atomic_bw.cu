/*
 * Copyright (c) 2021, NVIDIA CORPORATION   All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto   Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "atomic_bw_common.h"

DEFINE_ATOMIC_BW_FN_NO_ARG(inc);
DEFINE_ATOMIC_BW_FN_NO_ARG(fetch_inc);

DEFINE_ATOMIC_BW_FN_ONE_ARG(add, 1);
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_add, 1);

DEFINE_ATOMIC_BW_FN_ONE_ARG(and, (*(data_d + idx) << (i + 1)));
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_and, (*(data_d + idx) << (i + 1)));

DEFINE_ATOMIC_BW_FN_ONE_ARG(or, (*(data_d + idx) << i));
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_or, (*(data_d + idx) << i));

DEFINE_ATOMIC_BW_FN_ONE_ARG(xor, 1);
DEFINE_ATOMIC_BW_FN_ONE_ARG(fetch_xor, 1);

DEFINE_ATOMIC_BW_FN_ONE_ARG(swap, i + 1);
DEFINE_ATOMIC_BW_FN_ONE_ARG(set, i + 1);

DEFINE_ATOMIC_BW_FN_TWO_ARG(compare_swap, i, i + 1);

int main(int argc, char *argv[]) {
    int mype, npes;
    uint64_t *data_d = NULL;
    uint64_t set_value;
    unsigned int *counter_d;
    int max_blocks = BLOCKS, max_threads = THREADS;
    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw;
    char perf_table_name[30];
    nvshmemi_amo_t atomic_op = NVSHMEMI_AMO_ACK;

    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    int max_msg_size = MAX_MSG_SIZE;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper(&argc, &argv);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes   \n");
        goto finalize;
    }

    while (1) {
        int c;
        c = getopt(argc, argv, "a:c:t:h");
        if (c == -1) break;

        switch (c) {
            case 'a':
                atomic_op_parse(optarg, &atomic_op);
                break;
            case 'c':
                max_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                max_threads = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf("-a [atomic op] -c [CTAs] -t [THREADS]   \n");
                atomic_usage();
                goto finalize;
        }
    }

    if (atomic_op == NVSHMEMI_AMO_ACK) {
        printf("-a [atomic op] -c [CTAs] -t [THREADS]   \n");
        atomic_usage();
        goto finalize;
    }

    array_size = floor(std::log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    data_d = (uint64_t *)nvshmem_malloc(max_msg_size);
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    switch (atomic_op) {
        case NVSHMEMI_AMO_INC: {
            strncpy(perf_table_name, "shmem_at_inc_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_SET: {
            strncpy(perf_table_name, "shmem_at_set_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_ADD: {
            strncpy(perf_table_name, "shmem_at_add_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_AND: {
            strncpy(perf_table_name, "shmem_at_and_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_OR: {
            strncpy(perf_table_name, "shmem_at_or_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_XOR: {
            strncpy(perf_table_name, "shmem_at_xor_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_FETCH_INC: {
            strncpy(perf_table_name, "shmem_at_finc_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_FETCH_ADD: {
            strncpy(perf_table_name, "shmem_at_fadd_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND: {
            strncpy(perf_table_name, "shmem_at_fand_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR: {
            strncpy(perf_table_name, "shmem_at_for_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR: {
            strncpy(perf_table_name, "shmem_at_fxor_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_SWAP: {
            strncpy(perf_table_name, "shmem_at_swap_bw", 30);
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            strncpy(perf_table_name, "shmem_at_cswap_bw", 30);
            break;
        }
        default: {
            /* Should be unreachable */
            fprintf(stderr, "Error, unsupported Atomic op %d.\n", atomic_op);
            printf("-a [atomic op] -c [CTAs] -t [THREADS]   \n");
            atomic_usage();
            goto finalize;
        }
    }

    int size;
    i = 0;
    if (mype == 0) {
        for (size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            int blocks = max_blocks, threads = max_threads;
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

            /* Do warmup round for NIC cache. */
            switch (atomic_op) {
                case NVSHMEMI_AMO_INC: {
                    atomic_inc_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_SET: {
                    atomic_set_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_ADD: {
                    atomic_add_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_AND: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    atomic_and_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_OR: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    atomic_or_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                      mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_XOR: {
                    set_value = 1;
                    for (size_t j = 0; j < size / sizeof(uint64_t); j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    atomic_xor_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_INC: {
                    atomic_fetch_inc_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_ADD: {
                    atomic_fetch_add_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_AND: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    atomic_fetch_and_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_OR: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    atomic_fetch_or_bw<<<blocks, threads>>>(data_d, counter_d,
                                                            size / sizeof(uint64_t), mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_XOR: {
                    for (size_t j = 0; j < size / sizeof(uint64_t); j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    atomic_fetch_xor_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_SWAP: {
                    atomic_swap_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                        mype, skip);
                    break;
                }
                case NVSHMEMI_AMO_COMPARE_SWAP: {
                    atomic_compare_swap_bw<<<blocks, threads>>>(
                        data_d, counter_d, size / sizeof(uint64_t), mype, skip);
                    break;
                }
                default: {
                    /* Should be unreachable */
                    fprintf(stderr, "Error, unsupported Atomic op %d.\n", atomic_op);
                    printf("-a [atomic op] -c [CTAs] -t [THREADS]   \n");
                    atomic_usage();
                    goto finalize;
                }
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            /* reset values in code. */
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            switch (atomic_op) {
                case NVSHMEMI_AMO_AND: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    break;
                }
                case NVSHMEMI_AMO_OR: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    break;
                }
                case NVSHMEMI_AMO_XOR: {
                    set_value = 1;
                    for (size_t j = 0; j < size / sizeof(uint64_t); j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    break;
                }
                case NVSHMEMI_AMO_FETCH_AND: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    break;
                }
                case NVSHMEMI_AMO_FETCH_OR: {
                    CUDA_CHECK(cudaMemset(data_d, 0xFF, size));
                    break;
                }
                case NVSHMEMI_AMO_FETCH_XOR: {
                    for (size_t j = 0; j < size / sizeof(uint64_t); j++) {
                        cudaMemcpy((data_d + j), &set_value, sizeof(uint64_t),
                                   cudaMemcpyHostToDevice);
                    }
                    break;
                }
                default: { break; }
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            cudaEventRecord(start);
            switch (atomic_op) {
                case NVSHMEMI_AMO_INC: {
                    atomic_inc_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_SET: {
                    atomic_set_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_ADD: {
                    atomic_add_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_AND: {
                    atomic_and_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_OR: {
                    atomic_or_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                      mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_XOR: {
                    atomic_xor_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                       mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_INC: {
                    atomic_fetch_inc_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_ADD: {
                    atomic_fetch_add_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_AND: {
                    atomic_fetch_and_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_OR: {
                    atomic_fetch_or_bw<<<blocks, threads>>>(data_d, counter_d,
                                                            size / sizeof(uint64_t), mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_FETCH_XOR: {
                    atomic_fetch_xor_bw<<<blocks, threads>>>(data_d, counter_d,
                                                             size / sizeof(uint64_t), mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_SWAP: {
                    atomic_swap_bw<<<blocks, threads>>>(data_d, counter_d, size / sizeof(uint64_t),
                                                        mype, iter);
                    break;
                }
                case NVSHMEMI_AMO_COMPARE_SWAP: {
                    atomic_compare_swap_bw<<<blocks, threads>>>(
                        data_d, counter_d, size / sizeof(uint64_t), mype, iter);
                    break;
                }
                default: {
                    /* Should be unreachable */
                    fprintf(stderr, "Error, unsupported Atomic op %d.\n", atomic_op);
                    printf("-a [atomic op] -c [CTAs] -t [THREADS]   \n");
                    atomic_usage();
                    goto finalize;
                }
            }
            cudaEventRecord(stop);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&milliseconds, start, stop);

            h_bw[i] = size / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
            nvshmem_barrier_all();
            i++;
        }
    } else {
        for (size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            nvshmem_barrier_all();
            nvshmem_barrier_all();
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        print_table_v1(perf_table_name, "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr,
                       h_bw, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
