//===- test_sort_utils.cc --------------------------------------- C++ ------===//
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
#include <iostream>
#include "moe_ag_scatter/sort_util.h"

bool
verify_sched(
    const std::vector<bytedance::flux::ProblemSchedule> &scheds,
    const std::vector<int32_t> &splits,
    int32_t world_size,
    int32_t tile_size_m) {
  int nexperts = splits.size() / world_size;
  for (int eid = 0; eid < nexperts; eid++) {
    int num_segments = (splits.at(eid) + tile_size_m - 1) / tile_size_m;
    std::vector<int> segment_count(num_segments, 0);
    for (const auto &sched : scheds) {
      if (sched.expert_id == eid) {
        for (int tiled_m = sched.m_start; tiled_m <= sched.m_end; tiled_m++) {
          segment_count[tiled_m]++;
        }
      }
    }
    for (int i = 0; i < num_segments; i++) {
      if (segment_count[i] != 1) {
        std::cerr << " expert " << eid << " segment " << i
                  << " schedule times: " << segment_count[i] << "\n";
        return false;
      }
    }
  }
  return true;
}

void
test1() {
  std::vector<int32_t> cumsum_per_rank = {147, 236, 299, 352, 402, 449, 491, 532};
  std::vector<int32_t> splits = {cumsum_per_rank.at(8 - 1)};

  auto scheds = bytedance::flux::get_sorted_problem_schedule_v2(
      splits.data(),
      1,  // rank
      8,  // world_size
      cumsum_per_rank.data(),
      0,
      splits.size(),
      128,
      1);
  for (const auto &sched : scheds) {
    std::cout << sched << "\n";
  }
  verify_sched(scheds, splits, 8, 128);
}

void
test2() {
  std::vector<int32_t> splits = {347};
  std::vector<int32_t> cumsum_per_rank = {0, 51, 114, 167, 218, 265, 307, 347};

  auto scheds = bytedance::flux::get_sorted_problem_schedule_v2(
      splits.data(),
      1,  // rank
      8,  // world_size
      cumsum_per_rank.data(),
      0,
      splits.size(),
      128,
      1);
  for (const auto &sched : scheds) {
    std::cout << sched << "\n";
  }
  verify_sched(scheds, splits, 8, 128);
}
void
test3() {
  std::vector<int32_t> splits = {76};
  std::vector<int32_t> cumsum_per_rank = {0, 0, 0, 0, 0, 0, 35, 76};

  auto scheds = bytedance::flux::get_sorted_problem_schedule_v2(
      splits.data(),
      1,  // rank
      8,  // world_size
      cumsum_per_rank.data(),
      0,
      splits.size(),
      128,
      1);
  for (const auto &sched : scheds) {
    std::cout << sched << "\n";
  }
  verify_sched(scheds, splits, 8, 128);
}

void
test4() {
  std::vector<int32_t> splits = {1110};
  std::vector<int32_t> cumsum_per_rank = {20, 256, 437, 596, 739, 871, 992, 1110};

  auto scheds = bytedance::flux::get_sorted_problem_schedule_v2(
      splits.data(),
      2,  // rank
      8,  // world_size
      cumsum_per_rank.data(),
      0,
      splits.size(),
      128,
      1);
  for (const auto &sched : scheds) {
    std::cout << sched << "\n";
  }
  verify_sched(scheds, splits, 8, 128);
}

void
test5() {
  std::vector<int32_t> splits = {2077};
  std::vector<int32_t> cumsum_per_rank = {401, 826, 1133, 1355, 1560, 1744, 1913, 2077};
  auto scheds = bytedance::flux::get_sorted_problem_schedule_v2(
      splits.data(),
      1,  // rank
      8,  // world_size
      cumsum_per_rank.data(),
      0,
      splits.size(),
      128,
      1);
  for (const auto &sched : scheds) {
    std::cout << sched << "\n";
  }
  verify_sched(scheds, splits, 8, 128);
}

void
test6() {
  std::vector<int32_t> splits = {401, 2077, 859};
  std::vector<int32_t> cumsum_per_rank = {
      /* E0 */ 0,   0,   0,    0,    0,    68,   237,  401,
      /* E1 */ 401, 826, 1133, 1355, 1560, 1744, 1913, 2077,
      /* E2 */ 0,   0,   0,    137,  342,  526,  695,  859};
  auto scheds = bytedance::flux::get_sorted_problem_schedule_v2(
      splits.data(),
      1,  // rank
      8,  // world_size
      cumsum_per_rank.data(),
      0,
      splits.size(),
      128,
      1);
  for (const auto &sched : scheds) {
    std::cout << sched << "\n";
  }
  verify_sched(scheds, splits, 8, 128);
}

void
test_random(int rank, int world_size, int nexperts, int max_m) {
  std::vector<int32_t> splits(nexperts, 0);
  std::vector<int32_t> cumsum_per_rank(nexperts * world_size, 0);
  for (int eid = 0; eid < nexperts; eid++) {
    // generate random splits for expert
    int m = std::max(rand() % max_m, world_size);
    std::vector<float> cumsum(world_size);
    for (int i = 0; i < world_size; i++) {
      cumsum[i] = (float)rand() / RAND_MAX;
    }
    std::sort(cumsum.begin(), cumsum.end());
    cumsum[world_size - 1] = 1.f;
    for (int i = 0; i < world_size; i++) {
      cumsum_per_rank[eid * world_size + i] = cumsum[i] * m;
    }
    splits[eid] = cumsum_per_rank[eid * world_size + world_size - 1];
  }
  auto scheds = bytedance::flux::get_sorted_problem_schedule_v2(
      splits.data(),
      rank,        // rank
      world_size,  // world_size
      cumsum_per_rank.data(),
      0,
      splits.size(),
      128,
      1);
  verify_sched(scheds, splits, world_size, 128);
}

void
test_very_long(int iters, int world_size) {
  for (int i = 0; i < iters; i++) {
    for (int rank = 0; rank < world_size; rank++) {
      test_random(rank, world_size, rand() % 32 + 1, 10000);
    }
    if (i % 1000 == 0) {
      std::cerr << "iter " << i << " done\n";
    }
  }
}

int
main() {
  std::cout << "\n\n>>>> test 1\n\n";
  test1();
  std::cout << "\n\n>>>> test 2\n\n";
  test2();
  std::cout << "\n\n>>>> test 3\n\n";
  test3();
  std::cout << "\n\n>>>> test 4\n\n";
  test4();
  std::cout << "\n\n>>>> test 5\n\n";
  test5();
  std::cout << "\n\n>>>> test 6\n\n";
  test6();

  std::cout << "\n\n>>>> test random\n\n";
  test_very_long(100000, 8);
  return 0;
}
