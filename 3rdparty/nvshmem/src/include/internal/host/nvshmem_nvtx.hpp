/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

/* needed for definition of NULL. */
// IWYU pragma: no_include <nvtx3/nvtxDetail/nvtxImplCore.h>
#include <cstddef>  // for NULL

#ifndef _NVSHMEM_NVTX_HPP_
#define _NVSHMEM_NVTX_HPP_

#ifndef NVTX_DISABLE

extern int nvshmem_nvtx_options;
extern void nvshmem_nvtx_print_options();

typedef enum nvtxOpt {
    INIT_OPT = (1 << 0),
    ALLOC_OPT = (1 << 1),
    LAUNCH_OPT = (1 << 2),
    COLL_OPT = (1 << 3),
    WAIT_OPT = (1 << 4),
    WAIT_ON_STREAM_OPT = (1 << 5),
    TEST_OPT = (1 << 6),
    MEMORDER_OPT = (1 << 7),
    QUIET_ON_STREAM_OPT = (1 << 8),
    ATOMIC_FETCH_OPT = (1 << 9),
    ATOMIC_SET_OPT = (1 << 10),
    RMA_BLOCKING_OPT = (1 << 11),
    RMA_NONBLOCKING_OPT = (1 << 12),
    PROXY_OPT = (1 << 13),

    DEFAULT_OPT = INIT_OPT | ALLOC_OPT | LAUNCH_OPT | COLL_OPT | WAIT_OPT | MEMORDER_OPT |
                  ATOMIC_FETCH_OPT | RMA_BLOCKING_OPT | PROXY_OPT,
    ALL_OPT = DEFAULT_OPT | WAIT_ON_STREAM_OPT | TEST_OPT | QUIET_ON_STREAM_OPT | ATOMIC_SET_OPT |
              RMA_NONBLOCKING_OPT
} nvtxOpt_t;

#include "internal/host/nvtx3.hpp"  // for domain, domain::global

template <class D = nvtx3::domain::global>
class nvtx_cond_range {
   public:
    nvtx_cond_range() = default;
    explicit nvtx_cond_range(nvtx3::event_attributes const& attr) noexcept : generate(true) {
        nvtxDomainRangePushEx(nvtx3::domain::get<D>(), attr.get());
    }

    ~nvtx_cond_range() noexcept {
        if (generate) {
            nvtxDomainRangePop(nvtx3::domain::get<D>());
        }
    }

    nvtx_cond_range(const nvtx_cond_range& other) = delete;
    nvtx_cond_range& operator=(const nvtx_cond_range& other) = delete;
    nvtx_cond_range(nvtx_cond_range&& other) noexcept {
        generate = other.generate;
        other.generate = false;
    }
    nvtx_cond_range& operator=(nvtx_cond_range&& other) noexcept {
        generate = other.generate;
        other.generate = false;

        return *this;
    }

   private:
    bool generate = false;
};  // class nvshmem_range

#define NVTX3_FUNC_RANGE_IN_IF(D, C)                                                     \
    nvtx_cond_range<D> range__{};                                                        \
    if (C) {                                                                             \
        static ::nvtx3::v1::registered_string<D> const nvtx3_func_name__{__func__};      \
        static ::nvtx3::v1::event_attributes const nvtx3_func_attr__{nvtx3_func_name__}; \
        range__ = nvtx_cond_range<D>{nvtx3_func_attr__};                                 \
    }

#define NVTX3_SCOPE_RANGE_IN_IF(D, C, N)                                                     \
    nvtx_cond_range<D> range_##N##__{};                                                      \
    if (C) {                                                                                 \
        static ::nvtx3::v1::registered_string<D> const nvtx3_string_##N##__{#N};             \
        static ::nvtx3::v1::event_attributes const nvtx3_attr_##N##__{nvtx3_string_##N##__}; \
        range_##N##__ = nvtx_cond_range<D>{nvtx3_attr_##N##__};                              \
    }

/* if NVSHMEM_NVTX_DOMAIN is defined before including nvshmem_nvtx.hpp, its
   value is used a domain name instead of the default domain name "NVSHMEM" */
#ifdef NVSHMEM_NVTX_DOMAIN
#define NVTX_TOSTRING(s) _NVTX_TOSTRING(s)  // expands s, if it is a macro
#define _NVTX_TOSTRING(s) #s

#define NVTX_CONCAT(a, b) _NVTX_CONCAT(a, b)  // expands a and b if they are macros
#define _NVTX_CONCAT(a, b) a##b

struct NVTX_CONCAT(nvshmem_domain, NVSHMEM_NVTX_DOMAIN) {
    static constexpr char const* name = NVTX_TOSTRING(NVSHMEM_NVTX_DOMAIN);
};
#define NVTX_SCOPE_IN_GROUP(G, N)                                             \
    NVTX3_SCOPE_RANGE_IN_IF(NVTX_CONCAT(nvshmem_domain, NVSHMEM_NVTX_DOMAIN), \
                            nvshmem_nvtx_options& G##_OPT, N)

#else /* NVSHMEM_NVTX_DOMAIN */

struct nvshmem_domain {
    static constexpr char const* name = "NVSHMEM";
};

// NVSHMEM-specific short versions
#define NVTX_FUNC_RANGE_IN_GROUP(G) \
    NVTX3_FUNC_RANGE_IN_IF(nvshmem_domain, nvshmem_nvtx_options& G##_OPT)

#define NVTX_SCOPE_IN_GROUP(G, N) \
    NVTX3_SCOPE_RANGE_IN_IF(nvshmem_domain, nvshmem_nvtx_options& G##_OPT, N)

#endif /* NVSHMEM_NVTX_DOMAIN */

extern void nvshmem_nvtx_set_thread_name(int pe, const char* suffix = NULL);

#else /* !NVTX_DISABLE */

static inline void nvshmem_nvtx_set_thread_name(int pe, const char* suffix = NULL) {}

#define NVTX_FUNC_RANGE_IN_GROUP(G)
#define NVTX_SCOPE_IN_GROUP(G, N)

#endif /* !NVTX_DISABLE */

extern void nvshmem_nvtx_init(void);

#endif /* _NVSHMEM_NVTX_HPP_ */
