//===- threadblock_swizzle_interleaved.hpp ------------------------ C++ ---===//
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

#include <cassert>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/gemm_coord.h"
#include "reduce_scatter/reduce_scatter_topos.hpp"
#include "flux/utils.h"
#include "flux/flux.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

namespace {
int
m_by_gemmk(int m, int tiled_size_m, int world_size) {
  int m_per_rank = m / world_size;
  int m_per_rank_fixed = (m_per_rank + tiled_size_m - 1) / tiled_size_m * tiled_size_m;
  return m_per_rank_fixed * world_size;
}
static constexpr int kLocalWorldSize = bytedance::flux::kLocalWorldSize;
static constexpr int kNumaWorldSize = 4;
}  // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
struct ThreadblockSwizzleStreamKInterleaved : public ThreadblockSwizzleStreamK {
 private:
  int world_size;
  int rank;
  int nnodes;
  int local_world_size;
  int local_rank;
  int tile_size_m;

 public:
  /// Constructor
  CUTLASS_HOST_DEVICE
  ThreadblockSwizzleStreamKInterleaved() : ThreadblockSwizzleStreamK() {}

  ThreadblockSwizzleStreamKInterleaved(
      GemmUniversalMode const mode_,
      GemmCoord const problem_size_,
      GemmCoord const tile_size_,
      int const batch_split_,  /// Either (mode == GemmUniversalMode::kBatched) the batch count, or
                               /// (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1
                               /// defaults to StreamK, >1 emulates Split-K)
      int const sm_occupancy_,
      int const device_sms_,
      int const avail_sms_,
      size_t const element_A_bytes_,
      size_t const element_B_bytes_,
      size_t const element_C_bytes_,
      int const epilogue_acc_fragments_,
      int rank_,
      int world_size_,
      int nnodes_)
      : ThreadblockSwizzleStreamK(
            mode_,
            problem_size_,
            tile_size_,
            batch_split_,
            sm_occupancy_,
            device_sms_,
            avail_sms_,
            element_A_bytes_,
            element_B_bytes_,
            element_C_bytes_,
            epilogue_acc_fragments_),
        world_size(world_size_),
        rank(rank_),
        nnodes(nnodes_),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        tile_size_m(tile_size_.m()) {
    FLUX_CHECK(world_size % nnodes == 0);
    FLUX_CHECK(rank < world_size);
  }

  /// Obtains the calling threadblock's tiled coordinates for the given tile index
  [[nodiscard]] CUTLASS_DEVICE GemmCoord
  get_tile_offset(int tile_idx) const {
    int tiled_m = ThreadblockSwizzleStreamK::tiled_shape().m();
    int tiled_n = ThreadblockSwizzleStreamK::tiled_shape().n();
    if (local_world_size <= 4) {  // using default
      auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);
      int m = coord.m(), n = coord.n();
      coord.m() = (coord.m() + tiled_m / local_world_size * (local_rank)) % tiled_m;
      return coord;
    } else {  // don't respect ThreadblockSwizzleStreamK::get_tile_offset
      int tiled_m_per_rank = tiled_m / local_world_size;
      if (tiled_m_per_rank * local_world_size == tiled_m) {
        int segment = tile_idx / (tiled_m_per_rank * tiled_n);
        int segment_new = bytedance::flux::kTopologys[0].rank_index[local_rank >= 4][segment];
        tile_idx = tile_idx % (tiled_m_per_rank * tiled_n);
        GemmCoord coord;
        coord.m() = tile_idx % tiled_m_per_rank + segment_new * tiled_m_per_rank;
        coord.n() = tile_idx / tiled_m_per_rank;
        // coord.n() += m_mappings[tiled_m - 1].len > 100;
        return coord;
      } else {
        int m = tile_idx / tiled_n, n = tile_idx % tiled_n;

        // const auto &mapping = m_mappings[m];
        // int tile_idx_new = (mapping.value - mapping.offset) * tiled_n + n;
        // GemmCoord coord;
        // coord.m() = tile_idx_new % mapping.len + mapping.offset;
        // coord.n() = tile_idx_new / mapping.len;
        // return coord;
        // printf("i will never show....\n");

        int m_per_rank = ThreadblockSwizzleStreamK::problem_size.m() / world_size;
        int kM = tile_size_m;
        const int *rank_index = &bytedance::flux::kTopologys[0].rank_index[local_rank >= 4][0];
        int m_offset = 0, m_extra = 0, m_len = 0;
        auto prev_rank_first = [&](int n) { return n != 0 && rank_index[n - 1] < rank_index[n]; };
        auto next_rank_first = [&](int n) { return n != 7 && rank_index[n + 1] < rank_index[n]; };
        for (int i = 0; i < 8; i++) {
          int rank = rank_index[i];
          int m_start = m_per_rank * rank, m_end = m_per_rank * (rank + 1) - 1;
          int tiled_m_start = m_start / kM, tiled_m_end = m_end / kM;  // close set
          int last_tiled_m_end = (m_start - 1) / kM, next_tiled_m_start = (m_end + 1) / kM;
          if (tiled_m_start == last_tiled_m_end && prev_rank_first(rank))
            tiled_m_start++;
          if (tiled_m_end == next_tiled_m_start && next_rank_first(rank))
            tiled_m_end--;
          // print(f"fixed tile_m: {tiled_m_start} {tiled_m_end}")
          m_len = tiled_m_end - tiled_m_start + 1;
          if (m >= m_offset && m < m_offset + m_len) {
            m_extra = m - m_offset;
            break;
          }
          m_offset += m_len;
        }
        int tile_idx_extra = m_extra * tiled_n + n;
        GemmCoord coord;
        coord.m() = tile_idx_extra % m_len + m_offset;
        coord.n() = tile_idx_extra / m_len;
        return coord;
      }
    }
  }
};

struct ThreadblockSwizzleStreamKInterleavedGemmk : public ThreadblockSwizzleStreamK {
 private:
  struct GroupInfo {
    int start, len;
  };
  int rank;
  int world_size;
  int nnodes;
  int local_rank;
  int local_world_size;
  bool use_gemmk;
  bool per_tile_flags;
  int tile_size_m;
  int m_per_rank;
  bool use_1d_ring;
  GroupInfo numa_infos[2];
  GroupInfo node_infos[2];

 public:
  /// Constructor
  CUTLASS_HOST_DEVICE
  ThreadblockSwizzleStreamKInterleavedGemmk() : ThreadblockSwizzleStreamK() {}

  ThreadblockSwizzleStreamKInterleavedGemmk(
      GemmUniversalMode const mode_,
      GemmCoord const problem_size_,
      GemmCoord const tile_size_,
      int const batch_split_,  /// Either (mode == GemmUniversalMode::kBatched) the batch count, or
                               /// (mode == GemmUniversalMode::kGemm) the tile-splitting factor (1
                               /// defaults to StreamK, >1 emulates Split-K)
      int const sm_occupancy_,
      int const device_sms_,
      int const avail_sms_,
      size_t const element_A_bytes_,
      size_t const element_B_bytes_,
      size_t const element_C_bytes_,
      int const epilogue_acc_fragments_,
      int rank_,
      int world_size_,
      int nnodes_,
      bool use_gemmk_,
      bool per_tile_flags_,
      bool use_1d_ring_)
      : ThreadblockSwizzleStreamK(
            mode_,
            GemmCoord(
                use_gemmk_ ? m_by_gemmk(problem_size_.m(), tile_size_.m(), world_size_)
                           : problem_size_.m(),
                problem_size_.n(),
                problem_size_.k()),
            tile_size_,
            batch_split_,
            sm_occupancy_,
            device_sms_,
            avail_sms_,
            element_A_bytes_,
            element_B_bytes_,
            element_C_bytes_,
            epilogue_acc_fragments_),
        rank(rank_),
        world_size(world_size_),
        nnodes(nnodes_),
        local_world_size(world_size_ / nnodes_),
        local_rank(rank_ % nnodes_),
        use_gemmk(use_gemmk_),
        per_tile_flags(per_tile_flags_),
        tile_size_m(tile_size_.m()),
        m_per_rank(problem_size.m() / world_size),
        use_1d_ring(use_1d_ring_) {
    if (use_gemmk) {
      int tiled_m = ThreadblockSwizzleStreamK::tiled_shape().m();
      FLUX_CHECK(tiled_m % world_size == 0)
          << "ThreadblockSwizzleStreamKInterleavedGemmk: tiled_m %% local_world_size != 0: "
          << tiled_m << " %% " << world_size;
    }
    FLUX_CHECK(world_size % nnodes == 0);
    FLUX_CHECK(rank < world_size);
    FLUX_CHECK(rank >= 0);
    if (nnodes > 1) {
      FLUX_CHECK(!per_tile_flags) << "multi-node does not support per_tile_flags";
    }
    if (nnodes != 1 && !per_tile_flags) {
      FLUX_CHECK(nnodes == 2);
      for (int i = 0; i < 2; i++) {
        auto &node_info = node_infos[i];
        node_info.start = (i * kLocalWorldSize * m_per_rank) / tile_size_m;
        int end = ((i + 1) * kLocalWorldSize * m_per_rank - 1) / tile_size_m;
        node_info.len = end - node_info.start + 1;
      }
    }
    if (local_world_size == kLocalWorldSize) {
      for (int i = 0; i < 2; i++) {
        auto &numa_info = numa_infos[i];
        numa_info.start = (i * kNumaWorldSize * m_per_rank) / tile_size_m;
        int end = ((i + 1) * kNumaWorldSize * m_per_rank - 1) / tile_size_m;
        numa_info.len = end - numa_info.start + 1;
      }
    }
  }

  /// Obtains the calling threadblock's tiled coordinates for the given tile index
  [[nodiscard]] CUTLASS_DEVICE GemmCoord
  get_tile_offset(int tile_idx) const {
    int tiled_m = ThreadblockSwizzleStreamK::tiled_shape().m();
    int tiled_n = ThreadblockSwizzleStreamK::tiled_shape().n();
    if (per_tile_flags) {
      if (use_gemmk) {
        int tiled_m_per_rank = tiled_m / world_size;
        // don't respect ThreadblockSwizzleStreamK::get_tile_offset
        int segment = tile_idx / (tiled_m_per_rank * tiled_n);
        GemmCoord coord;
        tile_idx = tile_idx % (tiled_m_per_rank * tiled_n);
        int segment_new =
            (use_1d_ring
                 ? (segment + rank + 1) % world_size
                 : bytedance::flux::kTopologys[0].rank_index[rank >= kNumaWorldSize][segment]);

        coord.m() = tile_idx % tiled_m_per_rank + segment_new * tiled_m_per_rank;
        coord.n() = tile_idx / tiled_m_per_rank;
        return coord;
      } else {
        int m_offset = problem_size.m() / world_size * (rank + 1);
        int tiled_m_offset = m_offset / tile_size_m;
        auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);
        coord.m() = (coord.m() + tiled_m_offset) % tiled_m;
        return coord;
      }
    } else {
      if (use_gemmk) {
        int tiled_m_idx = tile_idx / tiled_n, tiled_n_idx = tile_idx % tiled_n;
        int tiled_m_idx_new = 0;
        if (nnodes == 1) {
          if (use_1d_ring) {
            int tiled_m_per_segment = tiled_m / world_size;
            int segment = tiled_m_idx / tiled_m_per_segment;
            tiled_m_idx %= tiled_m_per_segment;
            segment = (segment + rank + 1) % world_size;
            tiled_m_idx_new = segment * tiled_m_per_segment + tiled_m_idx;
          } else {
            tiled_m_idx_new = reindex_single_node_gemmk(rank, tiled_m_idx, problem_size.m());
          }
        } else {
          tiled_m_idx_new =
              reindex_multi_node_gemmk(rank, tiled_m_idx, problem_size.m(), world_size);
        }
        return GemmCoord(tiled_m_idx_new, tiled_n_idx, 0);
      } else {  // as what all_gather does
        int m_offset = problem_size.m() / world_size * (rank + 1);
        int tiled_m_offset = m_offset / tile_size_m;
        int problem_size_m = problem_size.m();

        if (nnodes == 1) {
          if (use_1d_ring) {
            auto coord = ThreadblockSwizzleStreamK::get_tile_offset(tile_idx);
            coord.m() = (coord.m() + tiled_m_offset) % tiled_m;
            return coord;
          } else {
            int tile_idx_m = tile_idx / tiled_n, tile_idx_n = tile_idx % tiled_n;
            // along N
            tile_idx_m = reindex_single_node(rank, tile_idx_m, problem_size_m);
            return GemmCoord(tile_idx_m, tile_idx_n, 0);
          }
        } else {
          int tile_idx_m = tile_idx / tiled_n, tile_idx_n = tile_idx % tiled_n;
          // along N
          tile_idx_m = reindex_multi_node(rank, world_size, tile_idx_m, problem_size_m);
          return GemmCoord(tile_idx_m, tile_idx_n, 0);
        }
      }
    }
  }

 private:
  CUTLASS_DEVICE int
  get_local_index(int tiled_m_range, int tiled_m_target, int tiled_m_offset) const {
    if (tiled_m_range == 0) {
      return 0;
    }
    return (tiled_m_target - tiled_m_offset) % tiled_m_range + tiled_m_offset;
  }

  CUTLASS_DEVICE int
  reindex_single_node(int current_rank, int tile_idx_m, int problem_size_m) const {
    int tiled_m = (problem_size_m + tile_size_m - 1) / tile_size_m;
    if (current_rank < kNumaWorldSize) {
      int numa1_size = numa_infos[1].len;
      if (tile_idx_m < numa1_size) {
        return get_local_index(
            numa1_size,
            tile_idx_m + (current_rank + 1 + kNumaWorldSize) * m_per_rank / tile_size_m,
            numa_infos[1].start);
      } else {
        return get_local_index(
            tiled_m - numa1_size,
            tile_idx_m - numa1_size + (current_rank + 1) * m_per_rank / tile_size_m,
            0);
      }
    } else {  // numa0
      int offset = (current_rank + 1) * m_per_rank / tile_size_m;
      int numa0_size = numa_infos[0].len;
      if (tile_idx_m < numa0_size) {
        return get_local_index(numa0_size, tile_idx_m + offset, 0);
      } else {
        return get_local_index(tiled_m - numa0_size, tile_idx_m + offset, numa0_size);
      }
    }
  }

  CUTLASS_DEVICE int
  reindex_multi_node(
      int current_rank, int current_world_size, int tile_idx_m, int problem_size_m) const {
    int tiled_m = ThreadblockSwizzleStreamK::tiled_shape().m();
    if (current_rank < kLocalWorldSize) {
      // first node 1
      if (tile_idx_m < node_infos[1].len) {
        return node_infos[1].start +
               reindex_single_node(current_rank, tile_idx_m, node_infos[1].len * tile_size_m);
      } else {
        return reindex_single_node(
            current_rank,
            tile_idx_m - node_infos[1].len,
            problem_size_m - node_infos[1].len * tile_size_m);
      }
    } else {  // first node 0
      if (tile_idx_m < node_infos[0].len) {
        return node_infos[0].start +
               reindex_single_node(
                   current_rank - kLocalWorldSize, tile_idx_m, node_infos[0].len * tile_size_m);
      } else {
        return node_infos[0].len + reindex_single_node(
                                       current_rank - kLocalWorldSize,
                                       tile_idx_m - node_infos[0].len,
                                       problem_size_m - node_infos[0].len * tile_size_m);
      }
    }
  }

  CUTLASS_DEVICE int
  reindex_single_node_gemmk(int rank, int tile_idx_m, int m) const {
    int m_per_segment = m / kLocalWorldSize;
    int tiled_m_per_rank = (m_per_segment + tile_size_m - 1) / tile_size_m;
    int segment = tile_idx_m / tiled_m_per_rank;
    int segment_extra = tile_idx_m % tiled_m_per_rank;
    int numa_id_reindexed = (segment < kNumaWorldSize) ^ (rank < kNumaWorldSize) ? 0 : 1;
    int segment_new = (rank + segment + 1) % kNumaWorldSize + numa_id_reindexed * kNumaWorldSize;
    return segment_new * tiled_m_per_rank + segment_extra;
  }

  int CUTLASS_DEVICE
  reindex_multi_node_gemmk(int rank, int tile_idx_m, int m, int world_size) const {
    int m_per_segment = m / world_size;
    int tiled_m_per_rank = (m_per_segment + tile_size_m - 1) / tile_size_m;
    int segment = tile_idx_m / tiled_m_per_rank;
    int node_id_reindexed = (segment < kLocalWorldSize) ^ (rank < kLocalWorldSize) ? 0 : 1;
    int tiled_m_per_local = tiled_m_per_rank * kLocalWorldSize;
    int nnodes = world_size / kLocalWorldSize;
    int m_per_local = m / nnodes;
    return reindex_single_node_gemmk(
               rank % kLocalWorldSize, tile_idx_m % tiled_m_per_local, m_per_local) +
           node_id_reindexed * kLocalWorldSize * tiled_m_per_rank;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
