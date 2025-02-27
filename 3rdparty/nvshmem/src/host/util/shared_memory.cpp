#include <assert.h>                        // for assert
#include <errno.h>                         // for errno
#include <fcntl.h>                         // for O_RDWR, O_CREAT
#include <stdint.h>                        // for intmax_t
#include <stdio.h>                         // for NULL, size_t
#include <sys/mman.h>                      // for mmap, shm_open, munmap
#include <sys/stat.h>                      // for fstat, stat
#include "internal/host/debug.h"           // for INFO, NVSHMEM_INIT
#include "internal/host/nvshmemi_types.h"  // for nvshmemi_shared_memory_info
#include "unistd.h"                        // for close, ftruncate

int shared_memory_create(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
    int status = 0;

    info->size = sz;

    shm_unlink(name); /* Remove any previous stale file handle */
    info->shm_fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0777);
    if (info->shm_fd < 0) {
        INFO(NVSHMEM_INIT, "shm_open failed. Reason: %s\n", strerror(errno));
        return errno;
    }

    status = ftruncate(info->shm_fd, sz);
    if (status != 0) {
        INFO(NVSHMEM_INIT, "ftruncate failed. Reason: %s\n", strerror(errno));
        return status;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
    if (info->addr == MAP_FAILED) {
        INFO(NVSHMEM_INIT, "mmap failed. Reason: %s\n", strerror(errno));
        return errno;
    }

    return status;
}

int shared_memory_open(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
    int status = 0;
    info->size = sz;
    struct stat stat_shm;

    info->shm_fd = shm_open(name, O_RDWR, 0777);
    if (info->shm_fd < 0) {
        INFO(NVSHMEM_INIT, "shm_open failed. Reason: %s\n", strerror(errno));
        return errno;
    }

    status = fstat(info->shm_fd, &stat_shm);
    if (status != 0) {
        INFO(NVSHMEM_INIT, "fstat failed. Reason: %s", strerror(errno));
        return status;
    }
    assert(stat_shm.st_size == (intmax_t)sz);

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shm_fd, 0);
    if (info->addr == MAP_FAILED) {
        INFO(NVSHMEM_INIT, "mmap failed. Reason: %s", strerror(errno));
        return errno;
    }

    return status;
}

void shared_memory_close(char *shm_name, nvshmemi_shared_memory_info *info) {
    if (info->shm_fd) close(info->shm_fd);
    if (info->addr) {
        munmap(info->addr, info->size);
    }

    shm_unlink(shm_name);
}
