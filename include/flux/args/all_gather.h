//===- all_gather.h ----------------------------------------------- C++ ---===//
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
namespace bytedance::flux {

struct AGKernelArguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  int nnodes;
  float alpha;
  float beta;
  void *input;
  void *input_buffer;
  void const *weight;
  void const *bias;
  void *output;
  void *barrier_buffer;
};

struct AllGatherGemmArguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  float alpha;
  float beta;
  void const *input;
  void const *weight;
  void const *bias;
  void *output;
};

}  // namespace bytedance::flux
