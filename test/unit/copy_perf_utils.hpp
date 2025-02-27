//===- copy_perf_utils.hpp ------------------------------------------ C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cutlass/numeric_types.h>
#include <cute/numeric/numeric_types.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>
#include <cute/util/print.hpp>
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/cuda_common.h"

template <typename T>
constexpr static bool kIsFp16 = std::is_same_v<T, half> || std::is_same_v<T, cutlass::half_t>;
template <typename T>
constexpr static bool kIsBf16 =
    std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, cutlass::bfloat16_t>;

template <typename T>
using ToVecType =
    std::conditional_t<kIsFp16<T>, __half2, std::conditional_t<kIsBf16<T>, __nv_bfloat162, void>>;

std::vector<std::pair<int, int>>
parse_links(const std::string &str, bool bidirectional, bool verbose = false) {
  std::vector<std::pair<int, int>> links;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ';')) {
    std::stringstream ss2(item);
    std::string item2;
    std::getline(ss2, item2, ',');
    int src = std::stoi(item2);
    std::getline(ss2, item2, ',');
    int dst = std::stoi(item2);
    links.emplace_back(src, dst);
  }
  size_t link_size = links.size();
  if (verbose) {
    std::cerr << "copy links: " << link_size << "\n";
  }
  // parse bidirectional
  if (bidirectional) {
    links.reserve(link_size * 2);
    for (size_t i = 0; i < link_size; i++) {
      auto &link = links[i];
      links.emplace_back(link.second, link.first);
    }
    if (verbose) {
      std::cerr << "copy links after bidirectional: " << links.size() << "\n";
    }
  }
  // sort and dedup
  sort(links.begin(), links.end());
  links.erase(unique(links.begin(), links.end()), links.end());
  if (verbose) {
    std::cerr << "copy links after dedup: " << links.size() << "\n";
    for (auto &link : links) {
      std::cerr << link.first << "->" << link.second << "\n";
    }
  }

  return links;
}

template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes,
    /// Element type for reduction
    typename ElementType>
struct global_red;

template <typename AccessType>
struct global_red<AccessType, 4, cute::half_t> {
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

template <typename T, typename VecType = uint4>
__global__ void
run_copy_kernel(const T *src, T *dst, size_t elems) {
  // copy as int4
  VecType *src_vec = (VecType *)src;
  VecType *dst_vec = (VecType *)dst;
  constexpr int kElemsPerVec = sizeof(VecType) / sizeof(T);
  elems = elems / kElemsPerVec;
#pragma unroll(8)
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += gridDim.x * blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
}

// NOTE: global_red is not atomic for PCI-e. using with care
template <typename T, typename PackedType = uint>
__global__ void
run_reduce_kernel(const T *src, T *dst, size_t elems) {
  constexpr int kElemsPerVec = sizeof(PackedType) / sizeof(T);
  using VecT = ToVecType<T>;
  static_assert(sizeof(PackedType) % sizeof(VecT) == 0, "can't packed");
  constexpr int kVecPerPack = sizeof(PackedType) / sizeof(VecT);

  elems = elems / kElemsPerVec;
  union Pack {
    VecT values[kVecPerPack];
    PackedType value;
  };
  PackedType *src_vec = (PackedType *)src;
  Pack *dst_vec = (Pack *)dst;

#pragma unroll
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += gridDim.x * blockDim.x) {
    Pack src_pack;
    src_pack.value = src_vec[i];
    Pack &dst_pack = dst_vec[i];
    static_assert(kIsFp16<T>, "only fp16 is supported");
    if constexpr (kVecPerPack == 1) {
      global_red<VecT, sizeof(VecT), cute::half_t>(
          src_pack.values[0], (void *)(dst_pack.values + 0), true);
    } else if constexpr (kVecPerPack == 2) {
      global_red<VecT, sizeof(VecT), cute::half_t>(
          src_pack.values[0], (void *)(dst_pack.values + 0), true);
      global_red<VecT, sizeof(VecT), cute::half_t>(
          src_pack.values[1], (void *)(dst_pack.values + 1), true);
    } else if constexpr (kVecPerPack == 4) {
      global_red<VecT, sizeof(VecT), cute::half_t>(
          src_pack.values[0], (void *)(dst_pack.values + 0), true);
      global_red<VecT, sizeof(VecT), cute::half_t>(
          src_pack.values[1], (void *)(dst_pack.values + 1), true);
      global_red<VecT, sizeof(VecT), cute::half_t>(
          src_pack.values[2], (void *)(dst_pack.values + 2), true);
      global_red<VecT, sizeof(VecT), cute::half_t>(
          src_pack.values[3], (void *)(dst_pack.values + 3), true);
    }
  }
}

template <
    typename VecType,
    typename CuteTensor,
    typename ThreadblockShape,  // tiled by ThreadblockShape
    typename ThreadMapShape>
CUTLASS_DEVICE auto
tile(
    CuteTensor &tensor,
    const ThreadblockShape &thread_block_shape,
    const ThreadMapShape &thread_map_shape,  // <vectorized_factor, tcol, trow, count>
    int m_idx,                               // tiled_m index
    int n_idx,                               // tiled_n index
    int tcol,                                // thread col
    int trow) {                              // thread row
  auto tile = cute::local_tile(
      tensor, thread_block_shape, make_coord(cute::_, cute::_), cute::Step<cute::_1, cute::_1>{})(
      cute::_, cute::_, m_idx, n_idx);
  auto tile_maped =
      cute::make_tensor(
          std::forward<decltype(tile)>(tile).data(),
          cute::make_layout(cute::get<1>(tile.layout()), cute::get<0>(tile.layout())))
          .compose(cute::make_layout(thread_map_shape));
  return cute::filter((cute::recast<VecType>(tile_maped(cute::_, tcol, trow, cute::_))));
}

template <
    typename T,
    typename VecType = uint4,
    typename CuteTensorSrc,
    typename CuteTensorDst,
    typename ThreadBlockShape,
    typename ThreadMapShape,
    typename ThreadShape>
__global__ void
run_copy_per_tb_kernel(
    CuteTensorSrc src_tensor,
    CuteTensorDst dst_tensor,
    cute::Shape<int, int> problem_shape,
    ThreadBlockShape thread_block_shape,
    ThreadMapShape thread_map_shape,
    ThreadShape thread_shape) {
  __syncthreads();
  int kM = cute::get<0>(thread_block_shape);
  int kN = cute::get<1>(thread_block_shape);
  int tiled_m = cute::get<0>(problem_shape) / kM, tiled_n = cute::size<1>(problem_shape) / kN;
  print_per_block("tiled_m: %d, tiled_n: %d\n", tiled_m, tiled_n);
  int tiled_mn = tiled_m * tiled_n;
  auto [tcol, trow] = cute::idx2crd(threadIdx.x, thread_shape);
  for (int tid = blockIdx.x; tid < tiled_mn; tid += gridDim.x) {
    int m = tid / tiled_n, n = tid % tiled_n;
    auto src_tile =
        tile<VecType>(src_tensor, thread_block_shape, thread_map_shape, m, n, tcol, trow);
    auto dst_tile =
        tile<VecType>(dst_tensor, thread_block_shape, thread_map_shape, m, n, tcol, trow);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(src_tile); i++) {  // no check bound. suppose we got perfect shape
      dst_tile(i) = src_tile(i);
    }
  }
}

template <typename T, typename VecType = uint4, int kTiledM = 128, int kTiledN = 128>
void
run_copy_per_tile_cute(
    const T *src,
    T *dst,
    int m,
    int n,
    int n_stride,
    int num_blocks,
    int num_threads,
    cudaStream_t stream) {
  constexpr int kTiledMN = kTiledM * kTiledN;
  constexpr auto kThreadblockShape = cute::make_coord(cute::Int<kTiledM>{}, cute::Int<kTiledN>{});
  static_assert(sizeof(VecType) % sizeof(T) == 0);
  constexpr int kVecLen = sizeof(VecType) / sizeof(T);
  constexpr int kThreadsCol = kTiledN / kVecLen;
  constexpr int kTileSizeVec = kTiledMN / kVecLen;
  FLUX_CHECK_DIV(num_threads, kThreadsCol);
  int thread_rows = num_threads / kThreadsCol;
  FLUX_CHECK_DIV(m, kTiledM);
  FLUX_CHECK_DIV(n, kTiledN);
  FLUX_CHECK_DIV(kTileSizeVec, num_threads);

  auto thread_map_shape = cute::make_shape(  // map ThreadblockShape tile to a threadblock
      cute::C<kVecLen>{},                    // for vectorize
      cute::C<kThreadsCol>{},
      thread_rows,
      kTileSizeVec / num_threads);
  auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, thread_rows);  // total num_threads
  auto src_tensor = cute::make_tensor(
      cute::make_gmem_ptr(src), cute::make_shape(m, n), cute::make_stride(n_stride, 1));
  auto dst_tensor = cute::make_tensor(
      cute::make_gmem_ptr(dst), cute::make_shape(m, n), cute::make_stride(n_stride, 1));
  auto problem_shape = cute::make_shape(m, n);
  run_copy_per_tb_kernel<T, VecType><<<num_blocks, num_threads, 0, stream>>>(
      src_tensor, dst_tensor, problem_shape, kThreadblockShape, thread_map_shape, thread_shape);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, typename VecType = uint4>
void
run_copy_continous_cute(
    const T *src,
    T *dst,
    int m,
    int n,
    int n_stride,
    int num_blocks,
    int num_threads,
    cudaStream_t stream) {
  // continous treated as a (m*n/VecLen, VecLen)
  // thread_map tile shape: (m*n/num_blocks/VecLen, VecLen)
  constexpr int kVecLen = sizeof(VecType) / sizeof(T);
  const int kM = m * n / kVecLen / num_blocks;
  constexpr int kN = kVecLen;
  const int kTileSizeVec = kM * kN / kVecLen;
  constexpr int kThreadsCol = 1;
  constexpr int _VecSize = cute::C<kVecLen>{};

  static_assert(sizeof(VecType) % sizeof(T) == 0);
  FLUX_CHECK_DIV(num_threads, kThreadsCol);
  FLUX_CHECK_DIV(m * n, num_blocks * kVecLen)
      << "(" << m << "x" << n << ") % (" << num_blocks << "x" << kVecLen << ")";
  FLUX_CHECK_DIV(kTileSizeVec, kThreadsCol);

  auto thread_map_shape = cute::make_shape(  // map ThreadblockShape tile to a threadblock
      _VecSize,                              // for vectorize
      cute::C<kThreadsCol>{},
      num_threads,
      kTileSizeVec / num_threads);
  auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, num_threads);  // total num_threads
  auto src_tensor = cute::make_tensor(
      cute::make_gmem_ptr(src),
      cute::make_shape(m * n / kVecLen, _VecSize),
      cute::make_stride(_VecSize, cute::_1{}));
  auto dst_tensor = cute::make_tensor(
      cute::make_gmem_ptr(dst),
      cute::make_shape(m * n / kVecLen, _VecSize),
      cute::make_stride(_VecSize, cute::_1{}));
  auto problem_shape = cute::make_shape(m * n / kVecLen, _VecSize);
  run_copy_per_tb_kernel<T, VecType><<<num_blocks, num_threads, 0, stream>>>(
      src_tensor,
      dst_tensor,
      problem_shape,
      cute::make_shape(kM, kN),
      thread_map_shape,
      thread_shape);
  CUDA_CHECK(cudaGetLastError());
}
