/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include <stdio.h>
#include "shmem.h"
#include "nvshmem.h"

int main(int c, char *v[]) {
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

    shmem_init();
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);

    nvshmem_finalize();
    shmem_finalize();
    return 0;
}
