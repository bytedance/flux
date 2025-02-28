#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include "internal/host/nvshmem_internal.h"  // for nvshmemi_shared_memory_info

int shared_memory_create(const char *name, size_t sz, nvshmemi_shared_memory_info *info);
int shared_memory_open(const char *name, size_t sz, nvshmemi_shared_memory_info *info);
void shared_memory_close(char *shm_name, nvshmemi_shared_memory_info *info);

#endif
