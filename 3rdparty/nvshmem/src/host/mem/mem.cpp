/*
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>                                              // for assert
#include <cuda_runtime.h>                                        // for cudaHostUn...
#include <driver_types.h>                                        // for cudaPointe...
#include <errno.h>                                               // for EBUSY
#include <pthread.h>                                             // for pthread_rw...
#include <stdint.h>                                              // for SIZE_MAX
#include <stdio.h>                                               // for size_t, NULL
#include <stdlib.h>                                              // for free, calloc
#include <string.h>                                              // for memmove
#include "device_host/nvshmem_types.h"                           // for nvshmemi_d...
#include "device_host/nvshmem_common.cuh"                        // for nvshmemi_d...
#include "host/nvshmemx_api.h"                                   // for nvshmemx_b...
#include "non_abi/nvshmemx_error.h"                              // for NVSHMEMI_E...
#include "non_abi/nvshmem_build_options.h"                       // IWYU pragma: keep
#include "internal/host/nvshmem_internal.h"                      // for nvshmem_lo...
#include "internal/host/error_codes_internal.h"                  // for NVSHMEMI_I...
#include "internal/host/nvshmemi_types.h"                        // for nvshmemi_s...
#include "internal/host/util.h"                                  // for CUDA_RUNTI...
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvshmem_me...
#include "internal/host_transport/transport.h"                   // for nvshmem_tr...

#ifdef NVSHMEM_USE_DLMALLOC
#include "dlmalloc.h"
#endif

static int buffer_register(nvshmem_transport_t transport, void *addr, size_t length) {
    nvshmem_local_buf_cache_t *cache = (nvshmem_local_buf_cache_t *)transport->cache_handle;
    nvshmem_local_buf_handle_t *handle;
    size_t i;
    size_t selected_index;
    size_t number_of_handles;
    size_t remaining_length;
    void *heap_end;
    char *addr_calc;
    int status = 0;
    int lock_status = EBUSY;
    cudaPointerAttributes attr;

    if (length == 0) {
        NVSHMEMI_ERROR_PRINT("Unable to register zero length buffers.\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    number_of_handles = (length + NVSHMEMI_MAX_HANDLE_LENGTH - 1) / NVSHMEMI_MAX_HANDLE_LENGTH;
    status = cudaPointerGetAttributes(&attr, addr);
    if (status != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("Unable to query pointer attributes.\n");
        /* clear CUDA error string. */
        cudaGetLastError();
        return NVSHMEMX_ERROR_INTERNAL;
    }

    if (attr.type == cudaMemoryTypeManaged) {
        NVSHMEMI_ERROR_PRINT("Unable to register managed memory as it can migrate.\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    heap_end = (void *)((char *)nvshmemi_device_state.heap_base + nvshmemi_device_state.heap_size);
    if (addr >= nvshmemi_device_state.heap_base && addr < heap_end) {
        NVSHMEMI_ERROR_PRINT(
            "Unable to register nvshmem heap memory. It is registered by default.\n");
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    handle =
        (nvshmem_local_buf_handle_t *)calloc(number_of_handles, sizeof(nvshmem_local_buf_handle_t));
    if (handle == NULL) {
        NVSHMEMI_ERROR_PRINT("Unable to resize the registered buffer array.\n");
        return NVSHMEMX_ERROR_OUT_OF_MEMORY;
    }

    if (transport) {
        for (i = 0; i < number_of_handles; i++) {
            handle[i].handle = (nvshmem_mem_handle_t *)calloc(1, sizeof(nvshmem_mem_handle_t));
            handle[i].linked_with_prev = true;
            handle[i].linked_with_next = true;
            if (handle[i].handle == NULL) {
                NVSHMEMI_ERROR_PRINT("Unable to resize the registered buffer array.\n");
                status = NVSHMEMX_ERROR_OUT_OF_MEMORY;
                goto out_error_unlocked;
            }
        }
        handle[0].linked_with_prev = false;
        handle[number_of_handles - 1].linked_with_next = false;
    }

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_wrlock(&cache->buffer_lock);
    }

    if (lock_status != 0) {
        NVSHMEMI_ERROR_PRINT("Unable to acquire buffer registration lock with errno %d\n",
                             lock_status);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out_error_unlocked;
    }

    if (attr.type == cudaMemoryTypeUnregistered) {
        if (!nvshmemi_state->host_memory_registration_supported) {
            NVSHMEMI_ERROR_PRINT(
                "Unable to register host memory for this device as it doesn't support UVA.\n");
            status = NVSHMEMX_ERROR_INVALID_VALUE;
            goto out_unlock;
        }
        status = cudaHostRegister(addr, length, cudaHostRegisterDefault);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to register host memory with CUDA.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out_unlock;
        }
        for (i = 0; i < number_of_handles; i++) {
            handle[i].registered_by_us = true;
        }
    }

    /* We only need to register unregistered host buffers if there is no remote transport.
     * CUDA memory and registered host memory are already mapped into the address space
     * so nothing to register.
     */
    if (transport == NULL && handle->registered_by_us == false) {
        status = 0;
        NVSHMEMU_HOST_PTR_FREE(handle);
        goto out_unlock;
    }

    if ((cache->array_used + number_of_handles) >= cache->array_size) {
        size_t new_array_size;
        void *new_buf;

        if (number_of_handles > cache->array_size) {
            new_array_size = cache->array_size + number_of_handles;
        } else {
            new_array_size = cache->array_size * 2;
        }

        assert(new_array_size < (SIZE_MAX / sizeof(nvshmem_local_buf_handle_t)));
        new_buf = realloc(cache->buffers, new_array_size * sizeof(nvshmem_local_buf_handle_t *));
        if (new_buf == NULL) {
            NVSHMEMI_ERROR_PRINT("Unable to resize the registered buffer array.\n");
            status = NVSHMEMX_ERROR_OUT_OF_MEMORY;
            goto out_unlock;
        }
        cache->buffers = (nvshmem_local_buf_handle_t **)new_buf;
        cache->array_size = new_array_size;
    }

    for (i = 0; i < cache->array_used; i++) {
        nvshmem_local_buf_handle_t *tmp_handle = cache->buffers[i];
        if (addr > tmp_handle->ptr) {
            void *max_addr;
            max_addr = (void *)((char *)tmp_handle->ptr + tmp_handle->length);
            if (addr < max_addr) {
                NVSHMEMI_ERROR_PRINT("Unable to register overlapping memory regions.\n");
                status = NVSHMEMX_ERROR_INVALID_VALUE;
                goto out_unlock;
            }
            continue;
        } else if (addr == tmp_handle->ptr) {
            NVSHMEMI_ERROR_PRINT("Unable to double register memory regions.\n");
            status = NVSHMEMX_ERROR_INVALID_VALUE;
            goto out_unlock;
            /* addr < tmp_handle->ptr */
        } else {
            break;
        }
    }

    selected_index = i;
    remaining_length = length;
    addr_calc = (char *)addr;
    for (i = 0; i < number_of_handles; i++) {
        size_t register_length = remaining_length > NVSHMEMI_MAX_HANDLE_LENGTH
                                     ? NVSHMEMI_MAX_HANDLE_LENGTH
                                     : remaining_length;
        if (NULL != transport && NVSHMEMI_TRANSPORT_OPS_IS_GET_MEM(transport)) {
            assert(register_length < NVSHMEMI_DMA_BUF_MAX_LENGTH);
            status = transport->host_ops.get_mem_handle(handle[i].handle, NULL, (void *)addr_calc,
                                                        register_length, transport, true);
            if (status) {
                NVSHMEMI_ERROR_PRINT("Unable to assign new memory handle.\n");
                goto out_unlock;
            }
        }
        handle[i].ptr = (void *)addr_calc;
        handle[i].length = register_length;

        addr_calc += register_length;
        remaining_length -= register_length;
    }

    assert(selected_index < cache->array_size);
    if (selected_index < cache->array_used) {
        memmove(&cache->buffers[selected_index + number_of_handles],
                &cache->buffers[selected_index],
                sizeof(nvshmem_local_buf_handle_t *) * (cache->array_used - selected_index));
    }
    for (i = 0; i < number_of_handles; i++) {
        cache->buffers[selected_index + i] = &handle[i];
        cache->array_used++;
    }
    status = nvshmemi_update_device_state();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out_unlock,
                          "nvshmemi_update_device_state failed\n");

out_unlock:
    pthread_rwlock_unlock(&cache->buffer_lock);
    if (status == 0) {
        return 0;
    }

out_error_unlocked:
    if (handle) {
        if (handle->registered_by_us) {
            CUDA_RUNTIME_CHECK(cudaHostUnregister(addr));
        }

        for (i = 0; i < number_of_handles; i++) {
            NVSHMEMU_HOST_PTR_FREE(handle[i].handle);
        }
        NVSHMEMU_HOST_PTR_FREE(handle);
    }
    return status;
}

int nvshmemx_buffer_register(void *addr, size_t length) {
    int status_global = NVSHMEMX_SUCCESS;
    int status_local;

    for (int i = 0; i < nvshmemi_state->num_initialized_transports; i++) {
        if (NVSHMEMU_IS_BIT_SET(nvshmemi_state->transport_bitmap, i)) {
            status_local = buffer_register(nvshmemi_state->transports[i], addr, length);
            if (status_local) {
                NVSHMEMI_ERROR_PRINT("Buffer addition for transport %d failed with error %d\n", i,
                                     status_local);
                status_global = status_local;
            }
        }
    }

    return status_global;
}

static int buffer_unregister(nvshmem_transport_t transport, void *addr) {
    nvshmem_local_buf_cache_t *cache = (nvshmem_local_buf_cache_t *)transport->cache_handle;
    size_t i;
    size_t num_mem_handles = 0;
    int lock_status = EBUSY;
    bool linked_with_next;
    int status = 0;

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_wrlock(&cache->buffer_lock);
    }

    if (lock_status != 0) {
        NVSHMEMI_ERROR_PRINT("Unable to acquire buffer registration lock with errno %d\n",
                             lock_status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    for (i = 0; i < cache->array_used; i++) {
        nvshmem_local_buf_handle_t *tmp_handle = cache->buffers[i];
        if (addr > tmp_handle->ptr) {
            continue;
        } else if (addr == tmp_handle->ptr) {
            do {
                linked_with_next = tmp_handle->linked_with_next;
                if (NULL != transport && NVSHMEMI_TRANSPORT_OPS_IS_RELEASE_MEM(transport)) {
                    transport->host_ops.release_mem_handle(tmp_handle->handle, transport);
                    NVSHMEMU_HOST_PTR_FREE(tmp_handle->handle);
                }

                if (tmp_handle->registered_by_us && !tmp_handle->linked_with_prev) {
                    CUDA_RUNTIME_CHECK(cudaHostUnregister(tmp_handle->ptr));
                }
                num_mem_handles++;
                tmp_handle = cache->buffers[i + num_mem_handles];
            } while (linked_with_next);
            NVSHMEMU_HOST_PTR_FREE(cache->buffers[i]);

            if ((i + num_mem_handles) < cache->array_used) {
                memmove(&cache->buffers[i], &cache->buffers[i + num_mem_handles],
                        sizeof(nvshmem_local_buf_handle_t *) *
                            (cache->array_used - (i + num_mem_handles)));
            }

            cache->array_used -= num_mem_handles;
            status = nvshmemi_update_device_state();
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out_unlock,
                                  "nvshmemi_update_device_state failed\n");
            goto out_unlock;
            /* addr < tmp_handle->ptr*/
        } else {
            break;
        }
    }

    status = NVSHMEMX_ERROR_INVALID_VALUE;
out_unlock:
    pthread_rwlock_unlock(&cache->buffer_lock);
    return status;
}

int nvshmemx_buffer_unregister(void *addr) {
    int status_global = NVSHMEMX_SUCCESS;
    int status_local;

    for (int i = 0; i < nvshmemi_state->num_initialized_transports; i++) {
        if (NVSHMEMU_IS_BIT_SET(nvshmemi_state->transport_bitmap, i)) {
            status_local = buffer_unregister(nvshmemi_state->transports[i], addr);
            if (status_local) {
                NVSHMEMI_ERROR_PRINT("Buffer removal for transport %d failed with error %d\n", i,
                                     status_local);
                status_global = status_local;
            }
        }
    }

    return status_global;
}

static void buffer_unregister_all(nvshmem_transport_t transport) {
    nvshmem_local_buf_cache_t *cache = (nvshmem_local_buf_cache_t *)transport->cache_handle;
    int lock_status = EBUSY;
    size_t num_entries = 0;

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_wrlock(&cache->buffer_lock);
    }

    if (lock_status != 0) {
        NVSHMEMI_ERROR_PRINT(
            "Unable to acquire buffer registration lock with errno %d. Unregister all function "
            "failed.\n",
            lock_status);
        return;
    }

    for (size_t i = 0; i < cache->array_used; i++) {
        if (NULL != transport && NVSHMEMI_TRANSPORT_OPS_IS_RELEASE_MEM(transport)) {
            transport->host_ops.release_mem_handle(cache->buffers[i]->handle, transport);
            NVSHMEMU_HOST_PTR_FREE(cache->buffers[i]->handle);
        }

        if (cache->buffers[i]->registered_by_us && !cache->buffers[i]->linked_with_prev) {
            CUDA_RUNTIME_CHECK(cudaHostUnregister(cache->buffers[i]->ptr));
        }

        if (!cache->buffers[i]->linked_with_next) {
            NVSHMEMU_HOST_PTR_FREE(cache->buffers[i - num_entries]);
            num_entries = 0;
        } else {
            num_entries++;
        }
    }

    cache->array_used = 0;
    pthread_rwlock_unlock(&cache->buffer_lock);

    return;
}

void nvshmemi_transport_buffer_unregister_all(nvshmem_transport_t transport) {
    buffer_unregister_all(transport);
    return;
}

void nvshmemx_buffer_unregister_all() {
    for (int i = 0; i < nvshmemi_state->num_initialized_transports; i++) {
        if (NVSHMEMU_IS_BIT_SET(nvshmemi_state->transport_bitmap, i)) {
            buffer_unregister_all(nvshmemi_state->transports[i]);
        }
    }

    return;
}

struct nvshmem_mem_handle *nvshmemi_get_registered_buffer_handle(nvshmem_transport_t transport,
                                                                 void *addr, size_t *len) {
    nvshmem_local_buf_cache_t *cache = (nvshmem_local_buf_cache_t *)transport->cache_handle;
    nvshmem_local_buf_handle_t *tmp_handle;
    size_t min, max, mid;
    void *max_addr;
    size_t max_len;
    int lock_status = EBUSY;
    struct nvshmem_mem_handle *ret_handle = NULL;

    while (lock_status == EBUSY) {
        lock_status = pthread_rwlock_rdlock(&cache->buffer_lock);
    }

    if (lock_status != 0) {
        NVSHMEMI_ERROR_PRINT("Unable to acquire buffer registration lock with errno %d.\n",
                             lock_status);
        return ret_handle;
    }

    if (cache->array_used == 0) {
        goto out_unlock;
    }

    min = 0;
    max = cache->array_used;
    do {
        mid = (max - min) / 2 + min;
        /* We have gone past the end of the loop. */
        if (mid >= cache->array_used) {
            break;
        }
        tmp_handle = cache->buffers[mid];
        if (addr > tmp_handle->ptr) {
            max_addr = (void *)((char *)tmp_handle->ptr + tmp_handle->length);
            max_len = (uint64_t)((char *)max_addr - (char *)addr);
            if (addr < max_addr) {
                *len = *len < max_len ? *len : max_len;
                ret_handle = tmp_handle->handle;
                goto out_unlock;
            }
            min = mid + 1;
        } else if (addr == tmp_handle->ptr) {
            *len = *len < tmp_handle->length ? *len : tmp_handle->length;
            ret_handle = tmp_handle->handle;
            goto out_unlock;
        } else {
            if (mid == 0) {
                break;
            }
            max = mid - 1;
        }
    } while (max >= min);

    NVSHMEMI_ERROR_PRINT("Unable to find a reference to the requested buffer address.\n");

out_unlock:
    pthread_rwlock_unlock(&cache->buffer_lock);
    return ret_handle;
}

void nvshmemi_local_mem_cache_fini(nvshmem_local_buf_cache_t *cache) {
    pthread_rwlock_destroy(&cache->buffer_lock);
    NVSHMEMU_HOST_PTR_FREE(cache->buffers);
    NVSHMEMU_HOST_PTR_FREE(cache);
}

int nvshmemi_local_mem_cache_init(nvshmem_local_buf_cache_t **cache) {
    nvshmem_local_buf_cache_t *cache_ptr;
    int status;

    cache_ptr = (nvshmem_local_buf_cache_t *)calloc(1, sizeof(nvshmem_local_buf_cache_t));
    NVSHMEMI_NULL_ERROR_JMP(cache_ptr, status, NVSHMEMI_INTERNAL_ERROR, err,
                            "Unable to allocate cache.\n");
    cache_ptr->buffers = (nvshmem_local_buf_handle_t **)calloc(
        NVSHMEMI_LOCAL_BUF_CACHE_DEFAULT_SIZE, sizeof(nvshmem_local_buf_handle_t *));
    cache_ptr->array_size = NVSHMEMI_LOCAL_BUF_CACHE_DEFAULT_SIZE;
    cache_ptr->array_used = 0;
    status = pthread_rwlock_init(&cache_ptr->buffer_lock, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, err, "Unable to init cache lock.\n");

    *cache = cache_ptr;
    return NVSHMEMI_SUCCESS;
err:
    NVSHMEMU_HOST_PTR_FREE(cache_ptr->buffers);
    NVSHMEMU_HOST_PTR_FREE(cache_ptr);
    return status;
}
