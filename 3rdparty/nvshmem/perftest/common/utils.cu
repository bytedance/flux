/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "utils.h"
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <tuple>

double *d_latency = NULL;
double *d_avg_time = NULL;
double *latency = NULL;
double *avg_time = NULL;
int mype = 0;
int npes = 0;
int use_mpi = 0;
int use_shmem = 0;
int use_uid = 0;
__device__ int clockrate;

#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "unistd.h"
static uint64_t getHostHash() {
    char hostname[1024];
    uint64_t result = 5381;
    int status = 0;

    status = gethostname(hostname, 1024);
    if (status) ERROR_EXIT("gethostname failed \n");

    for (int c = 0; c < 1024 && hostname[c] != '\0'; c++) {
        result = ((result << 5) + result) + hostname[c];
    }

    return result;
}

/* This is a special function that is a WAR for a bug in OSHMEM
implementation. OSHMEM erroneosly sets the context on device 0 during
shmem_init. Hence before nvshmem_init() is called, device must be
set correctly */
void select_device_shmem() {
    cudaDeviceProp prop;
    int dev_count;
    int mype_node;
    int mype, n_pes;

    mype = shmem_my_pe();
    n_pes = shmem_n_pes();
    mype_node = 0;
    uint64_t host = getHostHash();
    uint64_t *hosts = (uint64_t *)shmem_malloc(sizeof(uint64_t) * (n_pes + 1));
    hosts[0] = host;
    long *pSync = (long *)shmem_malloc(SHMEM_COLLECT_SYNC_SIZE * sizeof(long));
    shmem_fcollect64(hosts + 1, hosts, 1, 0, 0, n_pes, pSync);
    for (int i = 0; i < n_pes; i++) {
        if (i == mype) break;
        if (hosts[i + 1] == host) mype_node++;
    }

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node % dev_count));

    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    fprintf(stderr, "mype: %d mype_node: %d device name: %s bus id: %d \n", mype, mype_node,
            prop.name, prop.pciBusID);
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0,
                                  cudaMemcpyHostToDevice));
}
#endif

void select_device() {
    cudaDeviceProp prop;
    int dev_count;
    int mype_node;
    int mype;

    mype = nvshmem_my_pe();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node % dev_count));

    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    fprintf(stderr, "mype: %d mype_node: %d device name: %s bus id: %d \n", mype, mype_node,
            prop.name, prop.pciBusID);
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0,
                                  cudaMemcpyHostToDevice));
}

void init_wrapper(int *c, char ***v) {
#ifdef NVSHMEMTEST_MPI_SUPPORT
    {
        char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
        if (value) use_mpi = atoi(value);
        char *uid_value = getenv("NVSHMEMTEST_USE_UID_BOOTSTRAP");
        if (uid_value) use_uid = atoi(uid_value);
    }
#endif

#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    {
        char *value = getenv("NVSHMEMTEST_USE_SHMEM_LAUNCHER");
        if (value) use_shmem = atoi(value);
    }
#endif

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi) {
        MPI_Init(c, v);
        int rank, nranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        DEBUG_PRINT("MPI: [%d of %d] hello MPI world! \n", rank, nranks);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        select_device();
        nvshmem_barrier_all();

        return;
    } else if (use_uid) {
        MPI_Init(c, v);
        int rank, nranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        DEBUG_PRINT("MPI: [%d of %d] hello MPI world! \n", rank, nranks);
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
        if (rank == 0) {
            nvshmemx_get_uniqueid(&id);
        }

        MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
        select_device();
        nvshmem_barrier_all();
        return;
    }
#endif

#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_init();
        mype = shmem_my_pe();
        npes = shmem_n_pes();
        DEBUG_PRINT("SHMEM: [%d of %d] hello SHMEM world! \n", mype, npes);

        latency = (double *)shmem_malloc(sizeof(double));
        if (!latency) ERROR_EXIT("(shmem_malloc) failed \n");

        avg_time = (double *)shmem_malloc(sizeof(double));
        if (!avg_time) ERROR_EXIT("(shmem_malloc) failed \n");

        select_device_shmem();

        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);

        nvshmem_barrier_all();
        return;
    }
#endif

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    select_device();

    nvshmem_barrier_all();
    d_latency = (double *)nvshmem_malloc(sizeof(double));
    if (!d_latency) ERROR_EXIT("nvshmem_malloc failed \n");

    d_avg_time = (double *)nvshmem_malloc(sizeof(double));
    if (!d_avg_time) ERROR_EXIT("nvshmem_malloc failed \n");

    DEBUG_PRINT("end of init \n");
    return;
}

void finalize_wrapper() {
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_free(latency);
        shmem_free(avg_time);
    }
#endif

#if !defined(NVSHMEMTEST_SHMEM_SUPPORT) && !defined(NVSHMEMTEST_MPI_SUPPORT)
    if (!use_mpi && !use_shmem) {
        nvshmem_free(d_latency);
        nvshmem_free(d_avg_time);
    }
#endif
    nvshmem_finalize();

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi || use_uid) {
        MPI_Finalize();
    }
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    if (use_shmem) shmem_finalize();
#endif
}

void alloc_tables(void ***table_mem, int num_tables, int num_entries_per_table) {
    void **tables;
    int i, dev_property;
    int dev_count;

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(
        cudaDeviceGetAttribute(&dev_property, cudaDevAttrUnifiedAddressing, mype_node % dev_count));
    assert(dev_property == 1);

    assert(num_tables >= 1);
    assert(num_entries_per_table >= 1);
    CUDA_CHECK(cudaHostAlloc(table_mem, num_tables * sizeof(void *), cudaHostAllocMapped));
    tables = *table_mem;

    /* Just allocate an array of 8 byte values. The user can decide if they want to use double or
     * uint64_t */
    for (i = 0; i < num_tables; i++) {
        CUDA_CHECK(
            cudaHostAlloc(&tables[i], num_entries_per_table * sizeof(double), cudaHostAllocMapped));
        memset(tables[i], 0, num_entries_per_table * sizeof(double));
    }
}

void free_tables(void **tables, int num_tables) {
    int i;
    for (i = 0; i < num_tables; i++) {
        CUDA_CHECK(cudaFreeHost(tables[i]));
    }
    CUDA_CHECK(cudaFreeHost(tables));
}

uint64_t get_coll_info(double *algBw, double *busBw, const char *job_name, uint64_t size,
                       double usec, int npes) {
    double baseBw, factor;
    // size == count * typesize
    // convert to seconds
    double sec = usec / 1.0E6;
    uint64_t total_bytes = 0;

    if (strcmp(job_name, "reduction") == 0 || strcmp(job_name, "reduction_on_stream") == 0 ||
        strcmp(job_name, "device_reduction") == 0) {
        baseBw = (double)(size) / 1.0E9 / sec;
        factor = ((double)2 * (npes - 1)) / ((double)(npes));
        total_bytes = size;
    } else if (strcmp(job_name, "broadcast") == 0 || strcmp(job_name, "broadcast_on_stream") == 0 ||
               strcmp(job_name, "bcast_device") == 0) {
        baseBw = (double)(size) / 1.0E9 / sec;
        factor = 1;
        total_bytes = size;
    } else if (strcmp(job_name, "alltoall") == 0 || strcmp(job_name, "alltoall_on_stream") == 0 ||
               strcmp(job_name, "alltoall_device") == 0 || strcmp(job_name, "fcollect") == 0 ||
               strcmp(job_name, "fcollect_on_stream") == 0 ||
               strcmp(job_name, "fcollect_device") == 0 || strcmp(job_name, "reducescatter") == 0 ||
               strcmp(job_name, "reducescatter_on_stream") == 0 ||
               strcmp(job_name, "device_reducescatter") == 0) {
        baseBw = (double)(size * npes) / 1.0E9 / sec;
        factor = ((double)(npes - 1)) / ((double)(npes));
        total_bytes = size * npes;
    } else {
        printf("Job Name %s bandwidth factor not set. Using 1 values for bw.\n", job_name);
        *algBw = 1;
        *busBw = 1;
        return size;
    }
    *algBw = baseBw;
    *busBw = baseBw * factor;
    return total_bytes;
}

tuple<double, double, double> get_latency_metrics(double *values) {
    double min, max, sum;
    int i = 0;
    min = max = values[0];
    sum = 0.0;

    while (values[i] != 0.00) {
        auto v = values[i];
        if (v < min) {
            min = v;
        }
        if (v > max) {
            max = v;
        }
        sum += v;
        i++;
    }
    double avg = (double)sum / i;
    return make_tuple(avg, min, max);
}

void print_table_basic(const char *job_name, const char *subjob_name, const char *var_name,
                       const char *output_var, const char *units, const char plus_minus,
                       uint64_t *size, double *value, int num_entries) {
    bool machine_readable = false;
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);
    int i;

    if (machine_readable) {
        printf("%s\n", job_name);
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name,
                       size[i], output_var, value[i], plus_minus, units);
            }
        }
    } else {
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-16s\n", "size(B)", "scope", "latency(us)");
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("%-10lu  %-8s  %-16.6lf", size[i], subjob_name, value[i]);
                printf("\n");
            }
        }
    }
}

void print_table_v1(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double *value, int num_entries) {
    bool machine_readable = false;
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);
    int i;

    int npes = nvshmem_n_pes();
    double avg, algbw, busbw, avgBusBw = 0;

    char **tokens = (char **)malloc(3 * sizeof(char *));
    const char *delim = "-";
    char copy[strlen(subjob_name) + 1];
    strcpy(copy, subjob_name);
    char *token = strtok(copy, delim);
    i = 0;
    while (token != NULL) {
        tokens[i] = strdup(token);
        token = strtok(NULL, delim);
        i++;
    }
    /* Used for automated test output. It outputs the data in a non human-friendly format. */
    if (machine_readable) {
        printf("%s\n", job_name);
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name,
                       size[i], output_var, value[i], plus_minus, units);
            }
        }
    } else if (strcmp(job_name, "device_reduction") == 0 ||
               strcmp(job_name, "device_reducescatter") == 0) {
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-8s  %-8s  %-16s  %-12s  %-12s\n", "size(B)", "type", "redop",
               "scope", "latency(us)", "algbw(GB/s)", "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                avg = value[i];
                uint64_t total_bytes = get_coll_info(&algbw, &busbw, job_name, size[i], avg, npes);
                avgBusBw += busbw;
                printf("%-10lu  %-8s  %-8s  %-8s  %-16.6lf  %-12.3lf  %-12.3lf", total_bytes,
                       tokens[0], tokens[1], tokens[2], avg, algbw, busbw);
                printf("\n");
            }
        }

    } else {
        // recombine first two tokens of subjob_name
        char type[strlen(subjob_name)];
        strcpy(type, subjob_name);
        char *last_delim = strrchr(type, '-');
        if (last_delim != NULL) *last_delim = '\0';
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-8s  %-16s  %-12s  %-12s\n", "size(B)", "type", "scope",
               "latency(us)", "algbw(GB/s)", "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                avg = value[i];
                uint64_t total_bytes = get_coll_info(&algbw, &busbw, job_name, size[i], avg, npes);
                avgBusBw += busbw;
                printf("%-10lu  %-8s  %-8s  %-16.6lf  %-12.3lf  %-12.3lf", total_bytes, type,
                       tokens[2], avg, algbw, busbw);
                printf("\n");
            }
        }
    }
    avgBusBw = avgBusBw / num_entries;
    printf("\n# Avg bus bandwidth    : %g\n\n", avgBusBw);
}

void print_table_v2(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double **values, int num_entries) {
    bool machine_readable = false;
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);
    int i;

    int npes = nvshmem_n_pes();
    double avgBusBw = 0;
    double avg, min, max, algbw, busbw = 0;

    /* Used for automated test output. It outputs the data in a non human-friendly format. */
    if (machine_readable) {
        printf("%s\n", job_name);
        for (i = 0; i < num_entries; i++) {
            auto value = values[i];
            tie(avg, min, max) = get_latency_metrics(value);
            if (size[i] != 0 && value[i] != 0.00) {
                printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name,
                       size[i], output_var, avg, plus_minus, units);
            }
        }
    } else if (strcmp(job_name, "reduction") == 0) {
        /* Splits subjob_name into data type and operation name */
        char **tokens = (char **)malloc(2 * sizeof(char *));
        const char *delim = "-";
        char copy[strlen(subjob_name) + 1];
        strcpy(copy, subjob_name);
        char *token = strtok(copy, delim);
        if (token != NULL) {
            tokens[0] = strdup(token);
            token = strtok(NULL, delim);
            if (token != NULL) {
                tokens[1] = strdup(token);
            } else {
                tokens[1] = strdup("None");
            }
        }
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-8s  %-16s  %-16s  %-16s  %-12s  %-12s\n", "size(B)", "type", "redop",
               "latency(us)", "min_lat(us)", "max_lat(us)", "algbw(GB/s)", "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            auto value = values[i];
            if (size[i] != 0 && value[i] != 0.00) {
                tie(avg, min, max) = get_latency_metrics(value);
                uint64_t total_bytes = get_coll_info(&algbw, &busbw, job_name, size[i], avg, npes);
                avgBusBw += busbw;
                printf("%-10.1lu  %-8s  %-8s  %-16.6lf  %-16.3lf  %-16.3lf  %-12.3lf  %-12.3lf",
                       total_bytes, tokens[0], tokens[1], avg, min, max, algbw, busbw);
                printf("\n");
            }
        }
    } else {
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-16s  %-16s  %-16s  %-12s  %-12s\n", "size(B)", "type", "latency(us)",
               "min_lat(us)", "max_lat(us)", "algbw(GB/s)", "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            auto value = values[i];
            if (size[i] != 0 && value[i] != 0.00) {
                tie(avg, min, max) = get_latency_metrics(value);
                uint64_t total_bytes = get_coll_info(&algbw, &busbw, job_name, size[i], avg, npes);
                avgBusBw += busbw;
                printf("%-10.1lu  %-8s  %-16.6lf  %-16.3lf  %-16.3lf  %-12.3lf  %-12.3lf",
                       total_bytes, subjob_name, avg, min, max, algbw, busbw);
                printf("\n");
            }
        }
    }
    avgBusBw = avgBusBw / num_entries;
    printf("\n# Avg bus bandwidth    : %g\n\n", avgBusBw);
}
