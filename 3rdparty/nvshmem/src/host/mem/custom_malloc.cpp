/*NVSHMEM specific macros
 * only using mspaces
 * single space
 * no mmap
 * no thread safety
 * only linux*/

#include <assert.h>                       // for assert
#include <cuda_runtime.h>                 // for cudaMemcpy, cudaMemset
#include <driver_types.h>                 // for cudaMemcpyDeviceToDevice
#include <stdint.h>                       // for uint64_t
#include <stdio.h>                        // for size_t, printf, NULL
#include <stdlib.h>                       // for exit
#include <iosfwd>                         // for std
#include <map>                            // for map, operator!=, map<>::ite...
#include <utility>                        // for pair
#include "internal/host/debug.h"          // for INFO, NVSHMEM_MEM
#include "internal/host/custom_malloc.h"  // for mspace, NVSHMEMI_MALLOC_ALI...
#include "internal/host/util.h"           // for CUDA_RUNTIME_CHECK

using namespace std;

#define SIZE_T_ONE ((size_t)1)
#define CHUNK_ALIGN_MASK (NVSHMEMI_MALLOC_ALIGNMENT - SIZE_T_ONE)

#define align_request(req) (((req) + CHUNK_ALIGN_MASK) & ~CHUNK_ALIGN_MASK)
/* the number of bytes to offset an address to align it */
#define align_offset(A)                    \
    ((((size_t)(A)&CHUNK_ALIGN_MASK) == 0) \
         ? 0                               \
         : ((NVSHMEMI_MALLOC_ALIGNMENT - ((size_t)(A)&CHUNK_ALIGN_MASK)) & CHUNK_ALIGN_MASK))

#ifdef _NVSHMEM_DEBUG
static size_t get_total_size(std::map<void *, size_t> chunk_map) {
    size_t sum = 0;
    for (map<void *, size_t>::iterator it = chunk_map.begin(); it != chunk_map.end(); it++) {
        sum += it->second;
    }
    return sum;
}

#define ASSERT_CORRECTNESS                                                                         \
    INFO(NVSHMEM_MEM,                                                                              \
         "get_total_size(free_chunks_start): %zu, get_total_size(in_use_cunks): %zu, total_size: " \
         "%zu\n",                                                                                  \
         get_total_size(free_chunks_start), get_total_size(inuse_chunks), total_size);             \
    assert(get_total_size(free_chunks_start) == get_total_size(free_chunks_end));                  \
    assert(get_total_size(free_chunks_start) + get_total_size(inuse_chunks) == total_size);
#else
#define ASSERT_CORRECTNESS
#endif

void mspace::print() {
    printf("free_chunks_start: ");
    for (map<void *, size_t>::iterator it = free_chunks_start.begin();
         it != free_chunks_start.end(); it++) {
        printf("(%p, %zu) ", it->first, it->second);
    }
    printf("\n");

    printf("free_chunks_end: ");
    for (map<void *, size_t>::iterator it = free_chunks_end.begin(); it != free_chunks_end.end();
         it++) {
        printf("(%p, %zu) ", it->first, it->second);
    }
    printf("\n");

    printf("inuse_chunks: ");
    for (map<void *, size_t>::iterator it = inuse_chunks.begin(); it != inuse_chunks.end(); it++) {
        printf("(%p, %zu) ", it->first, it->second);
    }
    printf("\n");
}

mspace::mspace(void *base, size_t capacity) {
    char *start_addr = (char *)base;
    size_t offset = align_offset(start_addr);
    start_addr += offset;
    capacity -= offset;
    if (capacity > 0) {
        char *end_addr = start_addr + capacity;

        free_chunks_start[start_addr] = capacity;
        free_chunks_end[end_addr] = capacity;
        total_size = capacity;
    }
    // print();
    ASSERT_CORRECTNESS
}

void mspace::add_free_chunk(char *base, size_t capacity) {
    bool merged = 0;
    /* check if previous chunk is free */
    if (free_chunks_end.find(base) != free_chunks_end.end()) {
        size_t psize = free_chunks_end[base];
        free_chunks_end.erase(base);
        free_chunks_end[base + capacity] = capacity + psize;
        base = base - psize;
        capacity += psize;
        free_chunks_start[base] = capacity;
        merged = 1;
    }
    /* check if next chunk is free */
    if (free_chunks_start.find(base + capacity) != free_chunks_start.end()) {
        size_t nsize = free_chunks_start[base + capacity];
        free_chunks_end.erase(base + capacity);
        free_chunks_start.erase(base + capacity);
        free_chunks_start[base] = capacity + nsize;
        free_chunks_end[base + capacity + nsize] = capacity + nsize;
        merged = 1;
    }

    if (!merged) {
        free_chunks_start[base] = capacity;
        free_chunks_end[base + capacity] = capacity;
    }
    ASSERT_CORRECTNESS
}

void mspace::add_new_chunk(void *base, size_t capacity) {
    total_size += capacity;
    add_free_chunk((char *)base, capacity);
}

int mspace::track_large_chunks(int enable) { return 0; }

void *mspace::allocate(size_t bytes) {
    INFO(NVSHMEM_MEM, "mspace_malloc called with %zu bytes", bytes);
    if (bytes == 0) return NULL;
    bytes = align_request(bytes);
    for (map<void *, size_t>::iterator it = free_chunks_start.begin();
         it != free_chunks_start.end(); it++) {
        if (it->second >= bytes) {
            INFO(NVSHMEM_MEM, "free chunk with size = %zu bytes found", it->second);
            char *start_addr = (char *)it->first;
            size_t rsize = it->second - bytes;
            if (rsize > 0) {
                free_chunks_start[start_addr + bytes] = rsize;
                free_chunks_end[start_addr + it->second] = rsize;
                free_chunks_start.erase(start_addr);
            } else {
                free_chunks_end.erase(start_addr + it->second);
                free_chunks_start.erase(start_addr);
            }
            inuse_chunks[start_addr] = bytes;
            ASSERT_CORRECTNESS
            return start_addr;
        }
    }
    return NULL;
}

void mspace::deallocate(void *mem) {
    INFO(NVSHMEM_MEM, "mspace_free called on %p", mem);
    if (inuse_chunks.find(mem) == inuse_chunks.end()) {
        printf("Free called on an invalid pointer\n");
        exit(-1);
    }
    size_t bytes = inuse_chunks[mem];
    inuse_chunks.erase(mem);

    add_free_chunk((char *)mem, bytes);
    ASSERT_CORRECTNESS
}

void *mspace::allocate_zeroed(size_t n_elements, size_t elem_size) {
    INFO(NVSHMEM_MEM, "mspace_calloc called with n_elements = %zu, elem_size = %zu", n_elements,
         elem_size);
    size_t bytes = n_elements * elem_size;
    void *ptr = allocate(bytes);
    if (ptr) CUDA_RUNTIME_CHECK(cudaMemset(ptr, 0, bytes));
    ASSERT_CORRECTNESS
    return ptr;
}

void *mspace::allocate_aligned(size_t alignment, size_t bytes) {
    INFO(NVSHMEM_MEM, "mspace_memalign called with alignment = %zu, bytes = %zu", alignment, bytes);
    assert((alignment % sizeof(void *)) == 0 && ((alignment & (alignment - 1)) == 0));
    /* Request bytes + alignment for simplicity */
    bytes += alignment;
    char *ptr = (char *)allocate(bytes);
    if (!ptr) return NULL;
    char *ret_ptr = (char *)(alignment * (((uint64_t)ptr + (alignment - 1)) / alignment));
    if (ret_ptr - ptr) {
        inuse_chunks[ret_ptr] = inuse_chunks[ptr] - (ret_ptr - ptr);
        inuse_chunks.erase(ptr);
        add_free_chunk(ptr, ret_ptr - ptr);
    }
    ASSERT_CORRECTNESS
    return ret_ptr;
}

void *mspace::reallocate(void *ptr, size_t size) {
    INFO(NVSHMEM_MEM, "mspace_realloc called with ptr = %p, size = %zu", ptr, size);
    size = align_request(size);
    size_t current_size = inuse_chunks[ptr];
    if (size < current_size) {
        inuse_chunks[ptr] = size;
        add_free_chunk((char *)ptr + size, current_size - size);
        ASSERT_CORRECTNESS
        return ptr;
    } else if (size > current_size) {
        if (free_chunks_start.find((char *)ptr + current_size) != free_chunks_start.end()) {
            size_t chunk_size = free_chunks_start[(char *)ptr + current_size];
            if (current_size + chunk_size >= size) {
                inuse_chunks[ptr] = size;
                free_chunks_start.erase((char *)ptr + current_size);
                free_chunks_end.erase((char *)ptr + current_size + chunk_size);
                if (current_size + chunk_size > size)
                    add_free_chunk((char *)ptr + size, size - current_size);
                ASSERT_CORRECTNESS
                return ptr;
            }
        }
        void *new_ptr = allocate(size);
        if (new_ptr == NULL) return NULL;
        CUDA_RUNTIME_CHECK(cudaMemcpy(new_ptr, ptr, size, cudaMemcpyDeviceToDevice));
        inuse_chunks.erase(ptr);
        add_free_chunk((char *)ptr, current_size);
        ASSERT_CORRECTNESS
        return new_ptr;
    } else {
        return ptr;
    }
}
