/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

/*
 *   mspace is an opaque type representing an independent
 *     region of space that supports mspace_malloc, etc.
 *     */

#include <stddef.h>

#ifndef _DLMALLOC_H
#define _DLMALLOC_H
typedef void* mspace;

/*XXX: same definitions in dlmalloc.c because dlmalloc.c does not include me*/
#define NVSHMEM_SINGLE_HEAP 1
#ifndef MALLOC_ALIGNMENT
#if NVSHMEM_SINGLE_HEAP
#define MALLOC_ALIGNMENT ((size_t)512U)
#else
#define MALLOC_ALIGNMENT ((size_t)256U)
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* API to create and destroy mspace*/
mspace create_mspace_with_base(void* base, size_t capacity, int locked);
void mspace_add_new_chunk(mspace msp, void* ptr, size_t bytes);
size_t destroy_mspace(mspace msp);

/* API that prevents large chunks from being allocated with system alloc*/
int mspace_track_large_chunks(mspace msp, int enable);

/* API for allocation and deallocation*/
void* mspace_malloc(mspace msp, size_t bytes);
void* mspace_calloc(mspace msp, size_t n_elements, size_t elem_size);
void mspace_free(mspace msp, void* mem);
void* mspace_memalign(mspace msp, size_t alignment, size_t bytes);
void* mspace_realloc(mspace msp, void* ptr, size_t bytes);

#ifdef __cplusplus
}
#endif
#endif
