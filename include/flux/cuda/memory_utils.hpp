//===- memory_utils.hpp ------------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cute/arch/copy.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "flux/cuda/system_barrier.hpp"

namespace bytedance::flux {

enum class FluxNamedBarriers : int {
  FirstBarrier = static_cast<int>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier),
  ReduceScatterEpilogue = FirstBarrier,
  ReduceScatterFetch = FirstBarrier + 1,
  ReduceScatterReduce = FirstBarrier + 2,
  AGScatterGather = FirstBarrier,
  AGScatterFetcher = FirstBarrier + 1,
  GatherRSProducer = FirstBarrier,
  GatherRSConsumer = FirstBarrier + 1
};
}  // namespace bytedance::flux

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

// use red.sys.global to reduce on another GPU
template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes,
    /// Element type for reduction
    typename ElementType>
struct global_red;

template <typename AccessType>
struct global_red<AccessType, 4, half_t> {
  CUTLASS_DEVICE
  global_red(AccessType const &D, void *ptr, bool pred_guard) {
    uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p red.global.sys.add.noftz.f16x2 [%0], %1;\n"
        "}\n"
        :
        : "l"(ptr), "r"(data), "r"((int)pred_guard));
  }
};

template <typename AccessType>
struct global_red<AccessType, 4, bfloat16_t> {
  CUTLASS_DEVICE
  global_red(AccessType const &D, void *ptr, bool pred_guard) {
    uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p red.global.sys.add.noftz.bf16x2 [%0], %1;\n"
        "}\n"
        :
        : "l"(ptr), "r"(data), "r"((int)pred_guard));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};
template <typename AccessType>
struct global_red<AccessType, 4, float> {
  CUTLASS_DEVICE
  global_red(AccessType const &D, void *ptr, bool pred_guard) {
    uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
    // SM80 or higher
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p red.sys.global.add.f32  [%0], %1;\n"
        "}\n"
        :
        : "l"(ptr), "r"(data), "r"((int)pred_guard));
  }
};

template <typename AccessType>
struct global_red<AccessType, 8, float> {
  CUTLASS_DEVICE
  global_red(AccessType const &D, void *ptr, bool pred_guard) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t data2[2];
    *(uint64_t *)data2 = reinterpret_cast<uint64_t const &>(D);
    // SM90 or higher
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %3, 0;\n"
        "  @p red.sys.global.add.v2.f32  [%0], {%1, %2};\n"
        "}\n"
        :
        : "l"(ptr), "r"(data2[0]), "r"(data2[1]), "r"((int)pred_guard));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

template <const int StoreBytes>
struct async_load;

template <>
struct async_load<4> {
  CUTLASS_DEVICE
  async_load(int smem_addr, void *gmem_addr, bool pred_guard) {
    asm("{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 4;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_addr), "r"((int)pred_guard));
  }
};

template <>
struct async_load<8> {
  CUTLASS_DEVICE
  async_load(int smem_addr, void *gmem_addr, bool pred_guard) {
    asm("{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 8;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_addr), "r"((int)pred_guard));
  }
};

template <>
struct async_load<16> {
  CUTLASS_DEVICE
  async_load(int smem_addr, void *gmem_addr, bool pred_guard) {
    asm("{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_addr), "r"((int)pred_guard));
  }
};

// use red.global to reduce on local GPU
template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes,
    /// Element type for reduction
    typename ElementType>
struct local_red;

template <typename AccessType>
struct local_red<AccessType, 16, half_t> {
  CUTLASS_DEVICE
  local_red(AccessType const &D, void *ptr, bool pred_guard) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Registers = uint16_t[8];
    Registers const &data = reinterpret_cast<Registers const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %1, 0;\n"
        "  @p red.global.add.noftz.v8.f16 [%0], {%2, %3, %4, %5, %6, %7, %8, %9};\n"
        "}\n"
        :
        : "l"(ptr),
          "r"((int)pred_guard),
          "h"(data[0]),
          "h"(data[1]),
          "h"(data[2]),
          "h"(data[3]),
          "h"(data[4]),
          "h"(data[5]),
          "h"(data[6]),
          "h"(data[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

template <typename AccessType>
struct local_red<AccessType, 16, bfloat16_t> {
  CUTLASS_DEVICE
  local_red(AccessType const &D, void *ptr, bool pred_guard) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    using Registers = uint16_t[8];
    Registers const &data = reinterpret_cast<Registers const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %1, 0;\n"
        "  @p red.global.add.noftz.v8.bf16 [%0], {%2, %3, %4, %5, %6, %7, %8, %9};\n"
        "}\n"
        :
        : "l"(ptr),
          "r"((int)pred_guard),
          "h"(data[0]),
          "h"(data[1]),
          "h"(data[2]),
          "h"(data[3]),
          "h"(data[4]),
          "h"(data[5]),
          "h"(data[6]),
          "h"(data[7]));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass

namespace cute {

struct SM90_TMA_STORE_ADD_3D {
  CUTE_HOST_DEVICE static void
  copy(
      void const *const desc_ptr,
      void const *const smem_ptr,
      int32_t const &crd0,
      int32_t const &crd1,
      int32_t const &crd2) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group "
        "[%0, {%2, %3, %4}], [%1];"
        :
        : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_ADD_4D {
  CUTE_HOST_DEVICE static void
  copy(
      void const *const desc_ptr,
      void const *const smem_ptr,
      int32_t const &crd0,
      int32_t const &crd1,
      int32_t const &crd2,
      int32_t const &crd3) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.bulk_group [%0, {%2, %3, %4, %5}], "
        "[%1];"
        :
        : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_ADD_5D {
  CUTE_HOST_DEVICE static void
  copy(
      void const *const desc_ptr,
      void const *const smem_ptr,
      int32_t const &crd0,
      int32_t const &crd1,
      int32_t const &crd2,
      int32_t const &crd3,
      int32_t const &crd4) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.bulk_group [%0, {%2, %3, %4, %5, "
        "%6}], [%1];"
        :
        : "l"(gmem_int_desc),
          "r"(smem_int_ptr),
          "r"(crd0),
          "r"(crd1),
          "r"(crd2),
          "r"(crd3),
          "r"(crd4)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

struct SM90_TMA_STORE_ADD {
  CUTE_HOST_DEVICE static void
  copy(
      void const *const desc_ptr,
      void const *const smem_ptr,
      int32_t const &crd0,
      int32_t const &crd1,
      int32_t const &crd2) {
    return SM90_TMA_STORE_ADD_3D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2);
  }
  CUTE_HOST_DEVICE static void
  copy(
      void const *const desc_ptr,
      void const *const smem_ptr,
      int32_t const &crd0,
      int32_t const &crd1,
      int32_t const &crd2,
      int32_t const &crd3) {
    return SM90_TMA_STORE_ADD_4D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3);
  }
  CUTE_HOST_DEVICE static void
  copy(
      void const *const desc_ptr,
      void const *const smem_ptr,
      int32_t const &crd0,
      int32_t const &crd1,
      int32_t const &crd2,
      int32_t const &crd3,
      int32_t const &crd4) {
    return SM90_TMA_STORE_ADD_5D::copy(desc_ptr, smem_ptr, crd0, crd1, crd2, crd3, crd4);
  }
};

// The executable SM90_TMA_STORE_CUSTOM with tma_desc
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_STORE_ADD, NumBitsPerTMA, AuxParams_> {
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_STORE arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr TmaDescriptor const *
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr auto
  get_tma_tensor(GShape const &g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr void
  copy_unpack_(void const *const src_ptr, Coord const &dst_coord, seq<Is...>) const {
    SM90_TMA_STORE_ADD::copy(&tma_desc_, src_ptr, get<Is>(dst_coord)...);
  }

  // This is the copy_unpack dispatch for this Copy_Traits
  // Src needs to be a smem tensor
  // Dst needs to be a gmem tensor with TmaCoordIterator .data()
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(
      Copy_Traits const &traits, Tensor<TS, SLayout> const &src, Tensor<TD, DLayout> &dst) {
    static_assert(is_smem<TS>::value, "Expected smem src for SM90_TMA_STORE_ADD");
    // static_assert(is_gmem<TD>::value, "Expected gmem dst for SM90_TMA_STORE_ADD");  // TMA
    // spoofed src tensor

    traits.copy_unpack_(
        cute::raw_pointer_cast(src.data()),
        dst.data().coord_,
        tuple_seq<decltype(dst.data().coord_)>{});
  }
};

}  // namespace cute
