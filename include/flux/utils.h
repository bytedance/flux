#pragma once
#include <chrono>
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/fast_math.h"
#include "cute/int_tuple.hpp"
#include "flux/flux.h"

#include <iostream>

namespace bytedance::flux {
extern std::ostream null_stream;
}

#define FLUX_LOG(severity) std::cerr << severity << ": "
#define FLUX_LOG_FIRST_N(severity, N)                                \
  *([]() -> std::ostream * {                                         \
    static int counter = 0;                                          \
    counter++;                                                       \
    return counter > N ? &bytedance::flux::null_stream : &std::cerr; \
  }()) << __FILE__                                                   \
       << ":" << __LINE__ << ": "

namespace bytedance::flux {
int get_int_from_env(const char *env, int default_value);
bool get_bool_from_env(const char *env, bool default_value);

[[deprecated]]
int get_world_size_from_env();
[[deprecated]]
int get_local_world_size_from_env();
[[deprecated]]
int get_rank_from_env();
[[deprecated]]
int get_local_rank_from_env();

struct DistEnv {
 public:
  int32_t rank;
  int32_t world_size;
  int32_t nnodes;

  int32_t local_world_size;
  int32_t local_rank;
  int32_t node_idx;

 private:
  cutlass::FastDivmod divmod_local_world_size;
  cutlass::FastDivmod divmod_world_size;

 public:
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(DistEnv);
  DistEnv(int32_t rank, int32_t world_size, int32_t nnodes);

  CUTLASS_HOST_DEVICE
  int
  global_rank_to_local_rank(int global_rank) const {
    int node_idx, local_rank;
    divmod_local_world_size.fast_divmod(node_idx, local_rank, global_rank);
    return local_rank;
  }

  CUTLASS_HOST_DEVICE
  cute::tuple<int, int>
  global_rank_to_node_idx_local_rank(int global_rank) const {
    int node_idx, local_rank;
    divmod_local_world_size.fast_divmod(node_idx, local_rank, global_rank);
    return cute::make_tuple(node_idx, local_rank);
  }

  CUTLASS_HOST_DEVICE
  int
  local_rank_to_global_rank(int local_rank) const {
    return this->node_idx * local_world_size + local_rank;
  }

  CUTLASS_HOST_DEVICE
  int
  local_rank_to_global_rank(int local_rank, int node_idx) const {
    return node_idx * local_world_size + local_rank;
  }

  CUTLASS_HOST_DEVICE
  int
  rank_shift(int rank, int shift) const {
    int new_rank;
    int _ [[maybe_unused]] = divmod_world_size.divmod(new_rank, rank + shift);
    return new_rank;
  }
};

///////////////////////////////////////////////////////////////
// Timer
///////////////////////////////////////////////////////////////
class Timer {
 public:
  Timer() { start_point = std::chrono::steady_clock::now(); }

  void
  reset() {
    start_point = std::chrono::steady_clock::now();
  }

  double
  elapsed_millis() const {
    auto end_point = std::chrono::steady_clock::now();
    auto time = end_point - start_point;
    return std::chrono::duration<double, std::milli>(time).count();
  }

 private:
  std::chrono::steady_clock::time_point start_point;
};

}  // namespace bytedance::flux
