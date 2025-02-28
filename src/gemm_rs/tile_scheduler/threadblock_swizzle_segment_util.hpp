//===- threadblock_swizzle_segment_util.hpp ---------------------- C++ ---===//
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
#pragma once

#include "flux/flux.h"
#include "cutlass/gemm_coord.h"

namespace bytedance::flux {
struct SegmentInfo {
  int16_t segment_origin;       // from [0, world_size)
  int16_t size = 0;             // tile_m_start_new + size = tile_m_start_new
  int16_t tile_m_start_origin;  // mapping tile_m in [tile_m_start_origin, tile_m_end_origin] to
                                // [tile_m_start_new, tile_m_end_new]
  int16_t tile_m_start_new;
};
namespace {
std::ostream &
operator<<(std::ostream &os, const SegmentInfo &info) {
  os << "segment: " << info.segment_origin << ", size: " << info.size
     << ", tile_m_start_origin: " << info.tile_m_start_origin
     << ", tile_m_start_new: " << info.tile_m_start_new;
  return os;
}

std::vector<int>
get_segments_order_tiled_2d(int rank, int world_size, int sub_world_size) {
  // 1rd stage: 4 -> [0 -> 1 -> 2 -> 3] -> [7 -> 6 -> 5 -> 4] -> 0
  // 2rd stage: 5 -> [1 -> 2 -> 3 -> 0] -> [4 -> 7 -> 6 -> 5] -> 1
  // 3nd stage: 6 -> [2 -> 3 -> 0 -> 1] -> [5 -> 4 -> 7 -> 6] -> 2
  // 4st stage: 7 -> [3 -> 0 -> 1 -> 2] -> [6 -> 5 -> 4 -> 7] -> 3
  std::vector<int> segment_orders(world_size);
  int sub_node = rank / sub_world_size;
  // if pull mode
  for (int i = 0; i < world_size / 2; i++) {
    if (sub_node == 0) {
      segment_orders[i * 2] = (world_size - 1 + i) % sub_world_size + sub_world_size;
      segment_orders[i * 2 + 1] = (world_size - 1 + i) % sub_world_size;
    } else {
      segment_orders[i * 2] = i;
      segment_orders[i * 2 + 1] = i + sub_world_size;
    }
  }
  return segment_orders;
}

std::vector<int>
get_segments_order_per_segment(int rank, int world_size, int sub_world_size, int nnodes) {
  // first next node, then next subnode, then next sub rank
  std::vector<int> segment_orders(world_size);

  int local_world_size = world_size / nnodes;
  int sub_rank = rank % sub_world_size;
  int local_rank = rank % local_world_size;
  int sub_node = local_rank / sub_world_size;
  int node = rank / local_world_size;
  int sub_per_nnode = local_world_size / sub_world_size;

  int idx = 0;
  // if pull mode
  for (int k = 0; k < nnodes; k++) {
    int current_node = (node + k + 1) % nnodes;
    for (int j = 0; j < sub_per_nnode; j++) {
      int current_subnode = (sub_node + j + 1) % sub_per_nnode;
      for (int i = 0; i < sub_world_size; i++) {
        // segment_orders
        int current_sub_rank = (i + sub_rank + 1) % sub_world_size;
        segment_orders[idx++] =
            current_sub_rank + current_subnode * sub_world_size + current_node * local_world_size;
      }
    }
  }
  return segment_orders;
}

std::vector<int>
get_segments_order_tiled_1d(int rank, int world_size) {
  std::vector<int> segment_orders(world_size);
  // if pull mode
  for (int i = 0; i < world_size; i++) {
    segment_orders[i] = (rank + i + 1) % world_size;
  }
  return segment_orders;
}
}  // namespace

class ThreadBlockSwizzleSegmentUtils {
 private:
  cutlass::gemm::GemmCoord const problem_size;
  cutlass::gemm::GemmCoord const tiled_shape;  // (m/kM, n/kM)

  int rank;
  int world_size;
  int sub_world_size = 2;  // maybe world_size=4/8 and NUMA nnodes=2
  int nnodes = 1;
  int tile_size_m;
  bool use_2d_ring;
  bool per_tile_flags;

 public:
  ThreadBlockSwizzleSegmentUtils() = default;

  ThreadBlockSwizzleSegmentUtils(
      const cutlass::gemm::GemmCoord &problem_size_,
      cutlass::gemm::GemmCoord const tile_size_,
      int rank_,
      int world_size_,
      int sub_world_size_,
      int nnodes_,
      bool use_2d_ring_ = true,
      bool per_tile_flags_ = true)
      : problem_size(problem_size_),
        tiled_shape{
            (problem_size_.m() + tile_size_.m() - 1) / tile_size_.m(),
            (problem_size_.n() + tile_size_.n() - 1) / tile_size_.n(),
            1},
        rank(rank_),
        world_size(world_size_),
        sub_world_size(sub_world_size_),
        nnodes(nnodes_),
        tile_size_m(tile_size_.m()),
        use_2d_ring(use_2d_ring_),
        per_tile_flags(per_tile_flags_) {
    FLUX_CHECK_DIV(problem_size.m(), world_size);
    if (nnodes > 1) {
      FLUX_CHECK(!per_tile_flags) << "multi-node does not support per_tile_flags";
    }

    FLUX_CHECK(world_size % nnodes == 0);
    FLUX_CHECK(rank < world_size);
    FLUX_CHECK(rank >= 0);
  }

  std::vector<int>
  get_segments_order() {
    if (per_tile_flags) {
      if (use_2d_ring) {
        return get_segments_order_tiled_2d(rank, world_size, sub_world_size);
      } else {
        return get_segments_order_tiled_1d(rank, world_size);
      }
    }
    return get_segments_order_per_segment(rank, world_size, sub_world_size, nnodes);
  }

  std::vector<SegmentInfo>
  get_segments_info() {
    std::vector<SegmentInfo> segments(world_size);
    calc_segments_info(segments.data());
    return segments;
  }

  void
  calc_segments_info(SegmentInfo *segments) {
    auto segment_orders = get_segments_order();

    std::vector<bool> tile_m_visited(tiled_shape.m(), false);

    // std::cout << "m: " << problem_size.m() << "\n";
    int m_per_rank = problem_size.m() / world_size;
    for (int i = 0; i < world_size; i++) {
      int segment = segment_orders[i];
      int m_start = segment * m_per_rank;
      int m_end = (segment + 1) * m_per_rank - 1;
      int tile_m_start = m_start / tile_size_m;
      int tile_m_end = m_end / tile_size_m;
      // std::cout << " m: " << m_start << " " << m_end << " tile_m " << tile_m_start << " to "
      //           << tile_m_end << "\n";

      // tile_m may be visited because of overlap of segments for m_per_rank % kTileM != 0
      // segments may have no tiles at all
      int start = tile_m_start, end = tile_m_end;
      while (start <= tile_m_end && tile_m_visited[start] == 1)
        start++;
      while (end >= tile_m_start && tile_m_visited[end] == 1)
        end--;

      bool has_value = end >= tile_m_start && start <= tile_m_end;
      segments[i].segment_origin = segment;
      // segments[i].segment_new = i;
      segments[i].tile_m_start_origin =
          i == 0 ? 0 : (segments[i - 1].tile_m_start_origin + segments[i - 1].size);
      // std::cout << "start, end: " << tile_m_start << " " << tile_m_end << " -> " << start << " "
      //           << end << "\n";
      if (!has_value) {
        segments[i].size = 0;
        continue;
      }
      tile_m_visited[start] = true;
      tile_m_visited[end] = true;

      segments[i].size = (end - start + 1);
      segments[i].tile_m_start_new = start;
    }
  }
};

}  // namespace bytedance::flux
