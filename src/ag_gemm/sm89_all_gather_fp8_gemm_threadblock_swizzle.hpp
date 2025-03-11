//===- sm89_all_gather_fp8_gemm_threadblock_swizzle.hpp ----------- C++ ---===//
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
/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to
      GEMM problems.
*/

#pragma once

#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/gemm/threadblock/index_remat.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"

#include "flux/flux.h"
#include "flux/utils.h"
#include "ag_gemm/all_gather_swizzle.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
template <int N = 1>
struct AGGemmIdentityThreadblockSwizzle : public GemmIdentityThreadblockSwizzle<N> {
 public:
  int problem_size_m;
  int TILE_SIZE_M;
  int nnodes;
  int node_id;
  int rank;
  int world_size;
  int local_rank;
  int local_world_size;
  int raster_order;

  int tile_m_offset;
  int tiled_m;
  int local_tiled_m;

  CUTLASS_HOST_DEVICE
  AGGemmIdentityThreadblockSwizzle() {}

  CUTLASS_HOST_DEVICE
  AGGemmIdentityThreadblockSwizzle(
      int problem_size_m_,
      int tile_size_m_,
      int nnodes_,
      int rank_,
      int world_size_,
      int raster_order_)
      : problem_size_m(problem_size_m_),
        TILE_SIZE_M(tile_size_m_),
        nnodes(nnodes_),
        rank(rank_),
        world_size(world_size_),
        raster_order(raster_order_) {
    local_world_size = world_size / nnodes;
    local_rank = rank % local_world_size;
    node_id = rank / local_world_size;

    int M_per_rank = problem_size_m / world_size;
    int M_start = M_per_rank * rank;
    tile_m_offset = (M_start + TILE_SIZE_M - 1) / TILE_SIZE_M;

    tiled_m = (problem_size_m + TILE_SIZE_M - 1) / TILE_SIZE_M;
    local_tiled_m = tiled_m / nnodes;
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *Gemm* problem size: gemm(M, N, K)
  CUTLASS_HOST_DEVICE
  static GemmCoord
  get_tiled_shape(GemmCoord problem_size, GemmCoord tile_size, int split_k_slices) {
    return GemmCoord(
        (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
        (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
        split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(GemmCoord tiled_shape) {
    int tile = 1 << get_log_tile(tiled_shape);
    return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  static int
  get_log_tile(GemmCoord tiled_shape) {
    auto n = tiled_shape.n();
    // Thresholds picked so that it doesn't cause too many no-op CTAs
    if (N >= 8 && n >= 6)
      return 3;
    else if (N >= 4 && n >= 3)
      return 2;
    else if (N >= 2 && n >= 2)
      return 1;
    else
      return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord
  get_tile_offset(int log_tile) const {
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();
    int block_idx_z = RematerializeBlockIdxZ();

    int tile_offset_x = block_idx_x >> log_tile;
    int new_block_idx_x = (tile_offset_x + tile_m_offset) % tiled_m;

    return GemmCoord{
        new_block_idx_x,
        (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
        block_idx_z};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  static GemmCoord
  get_tile_offset(GemmCoord tiled_shape) {
    int const kTile = N;
    int block_idx_x = RematerializeBlockIdxX();
    int block_idx_y = RematerializeBlockIdxY();

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ()};

    return GemmCoord{
        (block_idx_x / kTile),
        (block_idx_y * kTile) + (block_idx_x % kTile),
        RematerializeBlockIdxZ()};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
struct AGGemmHorizontalThreadblockSwizzle : GemmHorizontalThreadblockSwizzle {
 public:
  int problem_size_m;
  int TILE_SIZE_M;
  int nnodes;
  int node_id;
  int rank;
  int world_size;
  int local_rank;
  int local_world_size;
  int raster_order;

  int tile_m_offset;
  int tiled_m;
  int local_tiled_m;

  CUTLASS_HOST_DEVICE
  AGGemmHorizontalThreadblockSwizzle() = default;

  CUTLASS_HOST_DEVICE
  AGGemmHorizontalThreadblockSwizzle(
      int problem_size_m_,
      int tile_size_m_,
      int nnodes_,
      int rank_,
      int world_size_,
      int raster_order_)
      : problem_size_m(problem_size_m_),
        TILE_SIZE_M(tile_size_m_),
        nnodes(nnodes_),
        rank(rank_),
        world_size(world_size_),
        raster_order(raster_order_) {
    local_world_size = world_size / nnodes;
    local_rank = rank % local_world_size;
    node_id = rank / local_world_size;

    int M_per_rank = problem_size_m / world_size;
    int M_start = M_per_rank * rank;
    tile_m_offset = (M_start + TILE_SIZE_M - 1) / TILE_SIZE_M;

    tiled_m = (problem_size_m + TILE_SIZE_M - 1) / TILE_SIZE_M;
    local_tiled_m = tiled_m / nnodes;
  }

  CUTLASS_DEVICE
  GemmCoord
  get_tile_offset(int log_tile /*no used*/) {
    auto coord =
        GemmCoord{RematerializeBlockIdxY(), RematerializeBlockIdxX(), RematerializeBlockIdxZ()};

    int m = (coord.m() + tile_m_offset) % tiled_m;
    coord.m() = m;
    return coord;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
