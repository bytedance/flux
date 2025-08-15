//===- inplace_cast.hpp ------------------------------------------- C++ ---===//
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

#define INPLACE_CAST_TS 8
#define INPLACE_CAST_NUM_BLOCKS 184
#define INPLACE_CAST_BLOCK_SIZE 256
#define INPLACE_CAST_CHUNK_SIZE 2048
// TS controls how many registers are used to hold data temporarily.
// If too large, each block will use too many registers. This will reduce occupancy and so
// significantly reduce performance.
// If too small, each thread gets too few works to do and may increase scheduling overheads.
// On L20, empirically, it was found that TS=8 and block_size=256 gives the best performance. TS=16
// and block_size=256 will give slightly worse performance although both will have occupancy of
// 100%. If occupancy is not 100%, the performance will be very bad.
// To calculate occupancy, first, figure out the register usage per thread with
// `cuobjdump -res-usage inplace_cast`. Then, input the register per thread and threads per block
// (block size) into the occupancy calculator in nsight-compute. Please see
// https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator for
// detailed usage.

namespace bytedance {
namespace flux {

void inplace_cast_fp32_to_bf16_impl(
    void *data,
    size_t size,
    unsigned *flags,
    unsigned *counter,
    cudaStream_t stream,
    int num_blocks,
    int block_size);

}
}  // namespace bytedance
