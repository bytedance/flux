//===- reduce_scatter.h ------------------------------------------- C++ ---===//
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

struct ReduceScatterArguments {
  int reduce_scatter_num_blocks = 12;
  void *rs_stream = nullptr;
  void *event = nullptr;
  bool use_barrier_queue = false;
  bool use_gemmk = true;             // use gemmk mechanism
  bool per_tile_flags = true;        // set flag per tile
  bool use_cudaMemcpyAsync = false;  // use cudaMemcpyAsync for memcpy or not
  int n_split = 1;                   // if also split n
  int sub_world_size = 1;
  void *opaque = nullptr;  // used to pass ncclComm_t for PCI-e cross node
  bool use_1d_ring = true;
  bool use_p2p_read = true;
};

struct GemmReduceScatterArguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  int nnodes;
  float alpha;
  float beta;
  void const *input;
  void const *weight;
  void const *bias;
  void **output_scatter_ptrs;
  void *local_reduce_buffer;
  void **barrier_ptrs = nullptr;
  int avail_sms = -1;
  ReduceScatterArguments reduce_scatter_args;
};

struct GemmReduceScatterFp8Arguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  int nnodes;
  float alpha;
  float beta;
  void const *input;
  void const *weight;
  void const *bias;
  void **output_scatter_ptrs;
  void *local_reduce_buffer;
  void **barrier_ptrs = nullptr;
  int avail_sms = -1;
  ReduceScatterArguments reduce_scatter_args;

  void *Aux;     // m * n
  void *Vector;  // bias: 1 * n
  float *abs_max_Aux;
  float *abs_max_D;
  // scaling tensors
  float const *scaleA;
  float const *scaleB;
  float const *scaleC;
  float const *scaleD;    // require if D is fp8
  float const *scaleAux;  // require if Aux is fp8
};

}  // namespace bytedance::flux
