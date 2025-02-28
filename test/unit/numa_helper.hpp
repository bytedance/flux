//===- numa_helper.hpp -------------------------------------------- C++ ---===//
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
#include <numa.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <errno.h>
#include "flux/flux.h"

#define LOG_ERR fprintf(stderr, "error %d got: %s\n", errno, strerror(errno))
#define CHECK_NUMA_ERROR(expr)                                                     \
  do {                                                                             \
    int rtn = expr;                                                                \
    if (rtn != 0) {                                                                \
      fprintf(stderr, "[%s:%d] `%s` error %d:\n", __FILE__, __LINE__, #expr, rtn); \
      LOG_ERR;                                                                     \
    }                                                                              \
  } while (0)

class NumaHelper {
 public:
  static int
  GetMaxNumaNodes() {
    return numa_available();
  }
};

class ScopedNumaMembind {
 public:
  ScopedNumaMembind() = delete;
  ScopedNumaMembind(const ScopedNumaMembind &) = delete;
  ScopedNumaMembind(ScopedNumaMembind &&) = delete;
  ScopedNumaMembind(int numa_node) : numa_node_(numa_get_membind_compat()) {
    FLUX_CHECK_GT(numa_num_configured_nodes(), 0);
    nodemask_t mask;
    nodemask_zero(&mask);
    if (numa_node == -1) {  // interleaved
      numa_set_membind(numa_all_nodes_ptr);
    } else {
      nodemask_set_compat(&mask, numa_node);
      numa_set_membind_compat(&mask);
    }
  }

  ~ScopedNumaMembind() { numa_set_membind_compat(&numa_node_); }

 private:
  nodemask_t numa_node_;
};

class ScopedNumaRunBind {
 public:
  ScopedNumaRunBind() = delete;
  ScopedNumaRunBind(const ScopedNumaRunBind &) = delete;
  ScopedNumaRunBind(ScopedNumaRunBind &&) = delete;
  // numa_node = -1 for all threads;
  ScopedNumaRunBind(int numa_node) {
    assert(NumaHelper::GetMaxNumaNodes() > 0);
    bitmask_ = numa_get_run_node_mask();
    if (numa_node == -1) {
      CHECK_NUMA_ERROR(numa_run_on_node_mask(numa_nodes_ptr));
    } else {
      CHECK_NUMA_ERROR(numa_run_on_node(numa_node));
    }
  }

  ~ScopedNumaRunBind() {
    CHECK_NUMA_ERROR(numa_run_on_node_mask(bitmask_));
    numa_bitmask_free(bitmask_);
  }

 private:
  bitmask *bitmask_ = nullptr;
};

class ScopedNumaBind {
 public:
  ScopedNumaBind() = delete;
  ScopedNumaBind(const ScopedNumaBind &) = delete;
  ScopedNumaBind(ScopedNumaBind &&) = delete;
  ScopedNumaBind(int numa_node) : mem_allocator_(numa_node), runner_(numa_node) {}

 private:
  ScopedNumaMembind mem_allocator_;
  ScopedNumaRunBind runner_;
};
