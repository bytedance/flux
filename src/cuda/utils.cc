//===- utils.cc -------------------------------------------------- C++ ---===//
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

#include "flux/utils.h"
#include <stdio.h>
#include <string>
#include <stdlib.h>

namespace bytedance::flux {

class NullBuffer : public std::streambuf {
 public:
  int
  overflow(int c) {
    return c;
  }
} null_buffer;
std::ostream null_stream(&null_buffer);

int
get_int_from_env(const char *env, int default_value) {
  auto *env_str = getenv(env);
  if (env_str == nullptr) {
#if defined(FLUX_DEBUG)
    fprintf(stderr, "env [%s] not set, using default_value: %d\n", env, default_value);
#endif
    return default_value;
  }
#if defined(FLUX_DEBUG)
  fprintf(stderr, "env [%s] set to %d\n", env, std::stoi(env_str));
#endif
  return std::stoi(env_str);
}

bool
get_bool_from_env(const char *env, bool default_value) {
  auto *env_str = getenv(env);
  if (env_str == nullptr) {
#if defined(FLUX_DEBUG)
    fprintf(stderr, "env [%s] not set, using default_value: %d\n", env, default_value);
#endif
    return default_value;
  }
  bool value = std::string(env_str) == "1" || std::string(env_str) == "ON";
#if defined(FLUX_DEBUG)
  fprintf(stderr, "env [%s] set to %d\n", env, value);
#endif
  return value;
}

namespace {
int
get_world_size_from_env_impl() {
  int tp_group_world_size = get_int_from_env("TP_GROUP_WORLD_SIZE", -1);
  if (tp_group_world_size != -1)
    return tp_group_world_size;
  return get_int_from_env("WORLD_SIZE", -1);
}

int
get_rank_from_env_impl() {
  int tp_group_rank = get_int_from_env("TP_GROUP_RANK", -1);
  if (tp_group_rank != -1)
    return tp_group_rank;
  return get_int_from_env("RANK", -1);
}
}  // namespace

int
get_world_size_from_env() {
  const static int kWorldSize = get_world_size_from_env_impl();
  return kWorldSize;
}
int
get_local_world_size_from_env() {
  const static int kLocalWorldSize = get_int_from_env("LOCAL_WORLD_SIZE", -1);
  return kLocalWorldSize;
}
int
get_rank_from_env() {
  const static int kRank = get_rank_from_env_impl();
  return kRank;
}
int
get_local_rank_from_env() {
  const static int kLocalRank = get_int_from_env("LOCAL_RANK", -1);
  return kLocalRank;
}

DistEnv::DistEnv(int32_t rank, int32_t world_size, int32_t nnodes)
    : rank(rank),
      world_size((world_size)),
      nnodes(nnodes),
      local_world_size(world_size / nnodes),
      local_rank(rank % local_world_size),
      node_idx(rank / local_world_size),
      divmod_local_world_size(local_world_size),
      divmod_world_size(world_size) {
  FLUX_CHECK(0 <= rank and rank < world_size)
      << "invalid DistEnv with rank:" << rank << ",world_size:" << world_size;
  FLUX_CHECK(local_world_size * nnodes == world_size)
      << "invalid DistEnv with world_size:" << world_size << " % nnodes:" << nnodes << " != 0";
  FLUX_CHECK(local_world_size <= kMaxLocalWorldSize)
      << local_world_size << " > " << kMaxLocalWorldSize;
}

}  // namespace bytedance::flux
