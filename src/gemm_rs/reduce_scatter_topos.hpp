//===- reduce_scatter_topos.hpp ----------------------------------- C++ ---===//
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

namespace bytedance::flux {
constexpr static int kLocalWorldSize = 8;
constexpr static int kStages = 4;
struct Topology {
  int rank_from[4][8];
  int rank_to[4][8];
  int unused_segments_push[8];
  int segments[4][2];
  int rank_index[2][8];
};
/*
  ring mode: topo 0
  1rd stage: 4 -> [0 -> 1 -> 2 -> 3] -> [7 -> 6 -> 5 -> 4] -> 0
  2rd stage: 5 -> [1 -> 2 -> 3 -> 0] -> [4 -> 7 -> 6 -> 5] -> 1
  3nd stage: 6 -> [2 -> 3 -> 0 -> 1] -> [5 -> 4 -> 7 -> 6] -> 2
  4st stage: 7 -> [3 -> 0 -> 1 -> 2] -> [6 -> 5 -> 4 -> 7] -> 3

  no ring mode: topo 1
  1rd stage: 4 -> [0 -> 1 -> 2 -> 3] -> [7 -> 6 -> 5 -> 4] -> 0
  2rd stage: 5 -> [1 -> 0 -> 3 -> 2] -> [6 -> 7 -> 4 -> 5] -> 1
  3nd stage: 6 -> [2 -> 3 -> 0 -> 1] -> [5 -> 4 -> 7 -> 6] -> 2
  4st stage: 7 -> [3 -> 2 -> 1 -> 0] -> [4 -> 5 -> 6 -> 7] -> 3

*/
constexpr static __device__ Topology kTopologys[] = {
    // topo 0
    {{{4, 0, 1, 2, 5, 6, 7, 3},
      {3, 5, 1, 2, 0, 6, 7, 4},
      {3, 0, 6, 2, 5, 1, 7, 4},
      {3, 0, 1, 7, 5, 6, 2, 4}},
     {{1, 2, 3, 7, 0, 4, 5, 6},
      {4, 2, 3, 0, 7, 1, 5, 6},
      {1, 5, 3, 0, 7, 4, 2, 6},
      {1, 2, 6, 0, 7, 4, 5, 3}},
     {3, 0, 1, 2, 5, 6, 7, 4},
     {{3, 4}, {0, 5}, {1, 6}, {2, 7}},
     {
         {7, 3, 4, 0, 5, 1, 6, 2},  // numa node 0
         {0, 4, 1, 5, 2, 6, 3, 7},  // numa node 1
     }},
    // topo 1
    {{{4, 0, 1, 2, 5, 6, 7, 3},
      {1, 5, 3, 0, 7, 4, 2, 6},
      {3, 0, 6, 2, 5, 1, 7, 4},
      {1, 2, 3, 7, 0, 4, 5, 6}},
     {{1, 2, 3, 7, 0, 4, 5, 6},
      {3, 0, 6, 2, 5, 1, 7, 4},
      {1, 5, 3, 0, 7, 4, 2, 6},
      {4, 0, 1, 2, 5, 6, 7, 3}},
     {3, 2, 1, 0, 7, 6, 5, 4},
     {{3, 4}, {2, 5}, {1, 6}, {0, 7}},
     {
         {7, 3, 6, 2, 5, 1, 4, 0},
         {0, 4, 1, 5, 2, 6, 3, 7},
     }}};
}  // namespace bytedance::flux
