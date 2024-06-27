//===- cuda_stub.h ----------------------------------------------- C++ ---===//
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

/*!
 * \file cuda_stub.h
 * \brief CUDA stub to avoid direct CUDA driver call
 */
#pragma once

#include <cuda.h>

namespace bytedance::flux {

#define FLUX_FORALL_CUDA(_)  \
  _(cuDeviceGetName)         \
  _(cuGetErrorString)        \
  _(cuGetErrorName)          \
  _(cuStreamWaitValue32_v2)  \
  _(cuStreamWriteValue32_v2) \
  _(cuStreamWaitValue64_v2)  \
  _(cuStreamWriteValue64_v2) \
  _(cuStreamBatchMemOp_v2)   \
  _(cuCtxGetDevice)

extern "C" {
typedef struct CUDA {
#define CREATE_MEMBER(name) decltype(&(name)) name;
  FLUX_FORALL_CUDA(CREATE_MEMBER)
#undef CREATE_MEMBER
} CUDA;
}

CUDA &cuda_stub();
namespace {
const char *
get_cu_error_string(CUresult statuse) {
  const char *msg;
  if (cuda_stub().cuGetErrorString(statuse, &msg) == CUDA_SUCCESS) {
    return msg;
  } else {
    return "unknown error";
  }
}
}  // namespace

#define CU_CHECK(status)                                                                       \
  do {                                                                                         \
    CUresult error = status;                                                                   \
    FLUX_CHECK(error == CUDA_SUCCESS) << "Got bad cuda status: " << get_cu_error_string(error) \
                                      << "(" << error << ") at " #status;                      \
  } while (0)

}  // namespace bytedance::flux
