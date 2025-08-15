//===- quantization.cu ---------------------------- C++ ---===//
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

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cuda/barrier>
#include <type_traits>

#include "quantization.hpp"

#define TMA_HW_SUPPORTED

namespace bytedance {
namespace flux {

#ifdef TMA_HW_SUPPORTED
using barrier = cuda::barrier<cuda::thread_scope_block>;
// namespace cde = cuda::device::experimental;
#endif

constexpr size_t BLOCK_TILE_DIM = 128;
constexpr size_t WARP_TILE_DIM_X = 128;
constexpr size_t WARP_TILE_DIM_Y = 16;
constexpr size_t THREAD_TILE_DIM_X = 16;
constexpr size_t THREAD_TILE_DIM_Y = 4;
constexpr int kThreadsPerWarp = 32;

#ifdef TMA_HW_SUPPORTED
constexpr size_t NUM_BYTES_PER_BANK = 4;
constexpr size_t NUM_BANKS_PER_SHARED_ELEM = THREAD_TILE_DIM_Y / NUM_BYTES_PER_BANK;
constexpr size_t SHARED_BLOCK_TILE_DIM_Y = BLOCK_TILE_DIM;
constexpr size_t SHARED_BLOCK_TILE_DIM_X_BANKS =
    BLOCK_TILE_DIM / (NUM_BYTES_PER_BANK * NUM_BANKS_PER_SHARED_ELEM);
constexpr size_t NUM_BANKS_Y_IN_WARP = WARP_TILE_DIM_Y / NUM_BYTES_PER_BANK;
#endif

constexpr size_t ELE_PER_THREAD = THREAD_TILE_DIM_X * THREAD_TILE_DIM_Y;
constexpr size_t THREADS_PER_BLOCK = BLOCK_TILE_DIM * BLOCK_TILE_DIM / ELE_PER_THREAD;
constexpr size_t NUM_WARPS_X_IN_BLOCK = BLOCK_TILE_DIM / WARP_TILE_DIM_X;
constexpr size_t NUM_WARPS_Y_IN_BLOCK = BLOCK_TILE_DIM / WARP_TILE_DIM_Y;
constexpr size_t NUM_WARPS_IN_BLOCK = NUM_WARPS_X_IN_BLOCK * NUM_WARPS_Y_IN_BLOCK;

constexpr size_t NUM_THREADS_X_IN_WARP = WARP_TILE_DIM_X / THREAD_TILE_DIM_X;
constexpr size_t NUM_THREADS_Y_IN_WARP = kThreadsPerWarp / NUM_THREADS_X_IN_WARP;

// avoid looping over THREAD_TILE_DIM_X, need to spread out the workload using threads in dim Y
constexpr size_t NUM_ELTS_EACH_THREAD_GRAB_BY_X = THREAD_TILE_DIM_X / NUM_THREADS_Y_IN_WARP;

#define FLUX_DISPATCH_BOOL(condition, ConstName, ...) \
  {                                                   \
    if (condition) {                                  \
      constexpr bool ConstName = true;                \
      {                                               \
        __VA_ARGS__                                   \
      }                                               \
    } else {                                          \
      constexpr bool ConstName = false;               \
      {                                               \
        __VA_ARGS__                                   \
      }                                               \
    }                                                 \
  }

// Utilities for vectorized loads and stores
template <int kBytes>
struct BytesToType {};

template <>
struct BytesToType<1> {
  using type = uint8_t;
  static_assert(sizeof(type) == 1);
};

template <>
struct BytesToType<2> {
  using type = uint16_t;
  static_assert(sizeof(type) == 2);
};

template <>
struct BytesToType<4> {
  using type = uint32_t;
  static_assert(sizeof(type) == 4);
};

template <>
struct BytesToType<8> {
  using type = uint64_t;
  static_assert(sizeof(type) == 8);
};

template <>
struct BytesToType<16> {
  using type = uint4;
  static_assert(sizeof(type) == 16);
};

struct uint8 {
  uint4 u1, u2;
};

struct uint16 {
  uint4 u1, u2, u3, u4;
};

template <>
struct BytesToType<32> {
  using type = uint8;
  static_assert(sizeof(type) == 32);
};

template <>
struct BytesToType<64> {
  using type = uint16;
  static_assert(sizeof(type) == 64);
};

template <int kNumElems>
__device__ __forceinline__ float
WarpReduceMax(const float per_thread_max) {
  // kNumElems must be a power of 2 and <= 32
  static_assert(kNumElems <= kThreadsPerWarp, "kNumElems must be <= kThreadsPerWarp (32)");
  static_assert((kNumElems & (kNumElems - 1)) == 0, "kNumElems must be a power of 2");
  // reduction using warp shuffling
  float current_max = per_thread_max;
#pragma unroll
  for (int delta = kNumElems / 2; delta > 0; delta /= 2) {
    const float received_max = __shfl_down_sync(0xFFFFFFFF, current_max, delta);
    __builtin_assume(current_max >= 0.0f);
    __builtin_assume(received_max >= 0.0f);
    current_max = fmaxf(current_max, received_max);
  }
  return current_max;
}

// Struct for vectorized loads and stores
template <typename EleType, uint32_t kNumEle>
struct Vec {
  static constexpr int kBytes = kNumEle * sizeof(EleType);
  using VecType = typename BytesToType<kBytes>::type;

  // Union for vector or element-wise data access
  using DataType = union {
    VecType vec;
    EleType ele[kNumEle];
  };

  DataType data;

  // Vectorized load data from memory, interpreting the pointer as VecType.
  inline __device__ void
  VecLoadFrom(const void *base_ptr, int64_t idx = 0) {
    this->data.vec = static_cast<const VecType *>(base_ptr)[idx];
  }

  // Vectorized store data to memory, interpreting the pointer as VecType.
  inline __device__ void
  VecStoreTo(void *base_ptr, int64_t idx = 0) const {
    static_cast<VecType *>(base_ptr)[idx] = this->data.vec;
  }

  // If the pointer is unaligned or `num_ele` is less than `kNumEle`, load data element-wise from
  // memory. The remaining elements are set to zero.
  inline __device__ void
  EleLoadFromIfNeeded(const void *base_ptr, int64_t idx = 0, int num_ele = kNumEle) {
    const EleType *ele_ptr = static_cast<const EleType *>(base_ptr) + idx;
    bool is_unaligned = reinterpret_cast<uintptr_t>(ele_ptr) % kBytes != 0;
    // element-wise load
    if (is_unaligned || num_ele < kNumEle) {
#pragma unroll
      for (int i = 0; i < kNumEle; i++) {
        EleType val = (i < num_ele ? ele_ptr[i] : static_cast<EleType>(0.f));
        this->data.ele[i] = val;
      }
    } else {
      // vectorized load
      this->VecLoadFrom(ele_ptr);
    }
  }

  // If the pointer is unaligned or `num_ele` is less than `kNumEle`, store data element-wise to
  // memory.
  inline __device__ void
  EleStoreToIfNeeded(void *base_ptr, int64_t idx = 0, int num_ele = kNumEle) const {
    EleType *ele_ptr = static_cast<EleType *>(base_ptr) + idx;
    bool is_unaligned = reinterpret_cast<uintptr_t>(ele_ptr) % kBytes != 0;
    // element-wise store
    if (is_unaligned || num_ele < kNumEle) {
#pragma unroll
      for (int i = 0; i < kNumEle; i++) {
        if (i < num_ele) {
          ele_ptr[i] = this->data.ele[i];
        }
      }
    } else {
      // vectorized store
      this->VecStoreTo(ele_ptr);
    }
  }

  // Set all elements to zero
  inline __device__ void
  clear() {
#pragma unroll
    for (int i = 0; i < kNumEle; i++) {
      this->data.ele[i] = static_cast<EleType>(0.f);
    }
  }
};

#define MIN(a, b) (a < b ? a : b)

template <typename T>
struct F8LimitsTrait;

template <>
struct F8LimitsTrait<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
};

template <>
struct F8LimitsTrait<__nv_fp8_e5m2> {
  static constexpr float max = 57344.0f;
};

// Type trait to resolve the max finite value
// represented by a input type to quantization.
// Or to represent max representable power of 2
// finite value.
template <typename T, bool ForcePow2>
struct HighPrecisionFloatScaleLimitsTrait;

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, false> {
  static constexpr float max = std::numeric_limits<float>::max();
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, false> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, false> {
  // Hex float format of 1.(10 bits of 1) * 2 ^ 15
  static constexpr float max = 0x1.FFCp15;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, true> {
  // Hex float format of 1.0 * 2 ^ 15
  static constexpr float max = 0x1.0p15;
};

static __device__ constexpr unsigned int WARP_REDUCE_AMAX_BY_Y_GROUP_MASKS[8] = {
    0x01010101,
    0x02020202,
    0x04040404,
    0x08080808,
    0x10101010,
    0x20202020,
    0x40404040,
    0x80808080};

// max for every group_size elements in warp
template <int group_size, int shfl_down_stride>
__device__ __forceinline__ float
groupMax(float val, unsigned int groupMask) {
  for (int offset = group_size / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(groupMask, val, offset * shfl_down_stride));
  }
  return val;
}

template <typename IType, typename OType, bool Power2Scaling>
__device__ __forceinline__ float
ComputeScale(const float amax, const float eps) {
  constexpr float fp8_max = F8LimitsTrait<OType>::max;

  // Clamping amax to avoid division by small numbers
  float amax_mod = fmaxf(amax, eps);

  // Handle overflow cases for non-clamped amax (eps is 0 or very small)
  if (amax_mod == 0.f) {
    // If amax is 0, return 1
    return 1.f;
  }
  // Compute scale factor
  float scale = fp8_max / amax_mod;

  if (isinf(scale)) {
    // If scale is infinity, return max value of IType
    return HighPrecisionFloatScaleLimitsTrait<IType, Power2Scaling>::max;
  }
  if (scale == 0.0) {
    // Case that amax is "inf". The frexp, ldexp logic changes 0.0 scales.
    // Return 0.0 for 0.0 scale here is consistent with non-Power2Scaling model.
    // quantization will remove signal from the tensor,
    // this is bad for the model, but define pow2Scale behavior
    // as returning 0.0 scale. amax calculation can
    // improve the situation to avoid this by taking largest finite.
    return scale;
  }
  if constexpr (Power2Scaling) {
    // NOTE: using bit fiddling based on advice of Asit in this
    // thread: https://nvidia.slack.com/archives/C06EDT7LZEW/p1738274404153439

    uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
    // Scale must be positive, shift it
    uint8_t exp = scale_bits >> 23;

    // inf scales already early returned, as did nan scales.
    // The cases to consider here are normals, zero, and subnormals.
    // zero is not possible with current math as
    // 448.0 / float_max == 1.31655e-36, which is the smallest
    // possible scale given current dtypes. It is still in the normal
    // fp32 range with an exponent of -120, so subnormals are also
    // not possible.

    int32_t normal_biased_exp = static_cast<int32_t>(exp) - 127;
    __builtin_assume(exp != 0);
    // Normal numbers case.

    // TODO: When combining with URM, can be useful to avoid exponent==127 edge
    // cases where a random mantissa corresponds to a floating point special.
    // Consider how to avoid those pitfalls if/when adding URM.
    scale = ldexpf(1.0f, normal_biased_exp);
  }
  return scale;
}

PFN_cuTensorMapEncodeTiled
get_cuTensorMapEncodeTiled() {
  void *driver_ptr = nullptr;
  cudaDriverEntryPointQueryResult driver_status;
  cudaGetDriverEntryPoint(
      "cuTensorMapEncodeTiled", &driver_ptr, cudaEnableDefault, &driver_status);
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
}

template <typename OutputType>
CUtensorMap
get_tensor_map(void *tensor, size_t global_dim_x, size_t global_dim_y) {
  // example-begin create-tensor-map
  CUtensorMap tensor_map_output_trans{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {global_dim_x, global_dim_y};  // x, y
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {global_dim_x * sizeof(OutputType)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {BLOCK_TILE_DIM, BLOCK_TILE_DIM};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
  CUtensorMapDataType dataType;

  if constexpr (
      std::is_same_v<OutputType, __nv_fp8_e4m3> || std::is_same_v<OutputType, __nv_fp8_e5m2>) {
    dataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    FLUX_CHECK(false) << "Invalid Output type.";
  }

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
      &tensor_map_output_trans,  // CUtensorMap *tensorMap,
      dataType,
      rank,                                    // cuuint32_t tensorRank,
      reinterpret_cast<OutputType *>(tensor),  // void *globalAddress,
      size,                                    // const cuuint64_t *globalDim,
      stride,                                  // const cuuint64_t *globalStrides,
      box_size,                                // const cuuint32_t *boxDim,
      elem_stride,                             // const cuuint32_t *elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  return tensor_map_output_trans;
}

template <
    bool RETURN_TRANSPOSE,
    bool IS_E8_SCALING,
    bool permute_scale,
    typename CType,
    typename IType,
    typename OType>
__global__ void
__launch_bounds__(THREADS_PER_BLOCK) block_scaled_1d_cast_transpose_kernel_notaligned(
    const IType *const input,
    OType *const output_c,
    OType *const output_t,
    CType *const tile_scales_inv_c,
    CType *const tile_scales_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon) {
  static_assert(
      BLOCK_TILE_DIM == WARP_TILE_DIM_X,
      "Please make sure 1d block scaling kernel config BLOCK_TILE_DIM == WARP_TILE_DIM_X");

  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  // shared mem for amax reduction in entire block, each warp produces one amax, there are
  // NUM_WARPS_IN_BLOCK amax to reduce
  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];
  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = RETURN_TRANSPOSE ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;
  const int tid_in_block_x = warp_id_in_block_x * NUM_THREADS_X_IN_WARP + tid_in_warp_x;

  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_row_idx = tile_id_y * BLOCK_TILE_DIM;
  const size_t block_tile_start_col_idx = tile_id_x * BLOCK_TILE_DIM;
  const size_t block_tile_start_idx =
      block_tile_start_row_idx * row_length + block_tile_start_col_idx;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  // handle non-full tile
  // check for three cases: full thread tile, nonfull thread tile, empty thread tile
  // for empty thread tile, directly write zero to the transposed shared mem buffer
  // for nonfull thread tile, fill zero to thread tile and act as if it's full
  const size_t thread_tile_start_row_idx =
      tile_id_y * BLOCK_TILE_DIM + warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP +
      tid_in_warp_y * THREAD_TILE_DIM_Y;
  const size_t thread_tile_start_col_idx =
      tile_id_x * BLOCK_TILE_DIM + warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
      tid_in_warp_x * THREAD_TILE_DIM_X;

  const size_t thread_tile_end_row_idx = thread_tile_start_row_idx + THREAD_TILE_DIM_Y - 1;
  const size_t thread_tile_end_col_idx = thread_tile_start_col_idx + THREAD_TILE_DIM_X - 1;

  // bool full_block = (block_tile_start_row_idx + BLOCK_TILE_DIM <= num_rows) &&
  // (block_tile_start_col_idx + BLOCK_TILE_DIM <= row_length);
  bool full_thrd_tile =
      (thread_tile_end_row_idx < num_rows) && (thread_tile_end_col_idx < row_length);
  bool empty_thrd_tile =
      (thread_tile_start_row_idx >= num_rows) || (thread_tile_start_col_idx >= row_length);
  bool nonfull_thrd_tile = (!full_thrd_tile) && (!empty_thrd_tile);

  const size_t thread_tile_ncols =
      MIN(THREAD_TILE_DIM_X,
          (MIN(thread_tile_end_col_idx, row_length - 1) - thread_tile_start_col_idx + 1));
  const size_t thread_tile_nrows =
      MIN(THREAD_TILE_DIM_Y,
          (MIN(thread_tile_end_row_idx, num_rows - 1) - thread_tile_start_row_idx + 1));

  CType col_amax[THREAD_TILE_DIM_Y] = {0};
  CType row_amax[THREAD_TILE_DIM_X_] = {0};  // not used if not return transpose

  CType col_scale[THREAD_TILE_DIM_Y];
  CType row_scale[THREAD_TILE_DIM_X_];  // not used if not return transpose

  if (!empty_thrd_tile) {
    // Step 1: Load a block tile of input data into thread tiles on registers
    // Edge case: nonfull thread tile case, will use the partial load function here
    if (nonfull_thrd_tile) {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        if (i >= thread_tile_nrows) {
          thrd_tile_input[i].clear();
        } else {
          thrd_tile_input[i].EleLoadFromIfNeeded(
              input + thread_tile_start_idx + i * row_length, 0, thread_tile_ncols);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        thrd_tile_input[i].EleLoadFromIfNeeded(
            input + thread_tile_start_idx + i * row_length, 0, THREAD_TILE_DIM_X);
      }
    }

    // Step 2: calculate withint warp tile for amax and scale
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        // amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
        col_amax[i] =
            fmaxf(col_amax[i], fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
        if constexpr (RETURN_TRANSPOSE) {
          row_amax[j] =
              fmaxf(row_amax[j], fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
        }
      }
    }
  }

  // Step 2.1: warp level & block level reduction on col_amax and row_amax, make sure they are amax
  // values across entire block
  unsigned int groupMask =
      0xFF << (tid_in_warp_y * 8);  // Activate only the threads in the current group
  for (int _i = 0; _i < THREAD_TILE_DIM_Y; _i++) {
    float groupMaxVal = groupMax<NUM_THREADS_X_IN_WARP, 1>(col_amax[_i], groupMask);
    col_amax[_i] = __shfl_sync(groupMask, groupMaxVal, tid_in_warp_y * NUM_THREADS_X_IN_WARP);
  }
  // Step 2.1.2: reduction of row_amax in x direction
  if constexpr (RETURN_TRANSPOSE) {
    // warp level reduction along y
    groupMask = WARP_REDUCE_AMAX_BY_Y_GROUP_MASKS[tid_in_warp_x];
    for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
      float groupMaxVal =
          groupMax<NUM_THREADS_Y_IN_WARP, NUM_THREADS_X_IN_WARP>(row_amax[_i], groupMask);
      row_amax[_i] = __shfl_sync(groupMask, groupMaxVal, tid_in_warp_x);
    }
    // then block level reduction using 2D smem, then reduction along one dim, no atomics
    __shared__ CType block_tile_row_amax_shared[NUM_WARPS_Y_IN_BLOCK][BLOCK_TILE_DIM];
    if (tid_in_warp_y == 0) {
#pragma unroll
      for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
        block_tile_row_amax_shared[warp_id_in_block_y]
                                  [_i * NUM_THREADS_X_IN_WARP + tid_in_warp_x] = row_amax[_i];
      }
    }
    __syncthreads();
    if (threadIdx.x < BLOCK_TILE_DIM) {
      CType final_amax = 0.0f;
#pragma unroll
      for (int _i = 0; _i < NUM_WARPS_IN_BLOCK; _i++) {
        final_amax = fmaxf(final_amax, block_tile_row_amax_shared[_i][threadIdx.x]);
      }
      block_tile_row_amax_shared[0][threadIdx.x] = final_amax;
    }
    __syncthreads();
#pragma unroll
    for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
      row_amax[_i] = block_tile_row_amax_shared[0][_i * NUM_THREADS_X_IN_WARP + tid_in_warp_x];
    }
  }

  __syncthreads();

// Step 2.2: all threads calculate scale
#pragma unroll
  for (int _i = 0; _i < THREAD_TILE_DIM_Y; _i++) {
    col_scale[_i] = ComputeScale<IType, OType, IS_E8_SCALING>(col_amax[_i], epsilon);
  }
  if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
    for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
      row_scale[_i] = ComputeScale<IType, OType, IS_E8_SCALING>(row_amax[_i], epsilon);
    }
  }

  // Step 2.3: save scale_inv by a subset of threads
  if (warp_id_in_block_x == 0 && tid_in_warp_x == 0 && !empty_thrd_tile) {
    // first column of threads of entire block save scale_inv values
    static_assert(std::is_same<CType, float>::value);
    for (int _i = 0; _i < thread_tile_nrows; _i++) {
      size_t row_idx = thread_tile_start_row_idx + _i;
      size_t col_idx = tile_id_x;
      CType scale_inv = 1.0 / col_scale[_i];
      if constexpr (permute_scale) {
        size_t p_row = row_idx / BLOCK_TILE_DIM;
        size_t p_col = col_idx;
        size_t p_dep = row_idx % BLOCK_TILE_DIM;
        size_t p_2d_stride = BLOCK_TILE_DIM * scale_stride_y;
        tile_scales_inv_c[p_row * p_2d_stride + p_col * BLOCK_TILE_DIM + p_dep] = scale_inv;
      } else {
        tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
      }
    }
  }

  // Step 2.4: save transposed version of scale_inv by a subset of threads
  if constexpr (RETURN_TRANSPOSE) {
    if (warp_id_in_block_y == 0 && tid_in_warp_y == 0 && !empty_thrd_tile) {
      // first row of threads of entire block save scale_inv values
      for (int _i = 0; _i < thread_tile_ncols; _i++) {
        size_t row_idx = tile_id_x * BLOCK_TILE_DIM + tid_in_block_x * THREAD_TILE_DIM_X + _i;
        size_t col_idx = tile_id_y;
        CType scale_inv = 1.0 / row_scale[_i];
        if constexpr (permute_scale) {
          size_t p_row = row_idx / BLOCK_TILE_DIM;
          size_t p_col = col_idx;
          size_t p_dep = row_idx % BLOCK_TILE_DIM;
          size_t p_2d_stride = BLOCK_TILE_DIM * scale_t_stride_y;
          tile_scales_inv_t[p_row * p_2d_stride + p_col * BLOCK_TILE_DIM + p_dep] = scale_inv;
        } else {
          tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
        }
      }
    }
  }

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  // Edge case: in the non-full tile case, there are three subcases
  // for full thread tile, it's the same thing here
  // for nonfull thread tile, pay attention when saving tmp_output_c to global memory, cannot
  // VecStoreTo, but need to VecStoreTo_elts for empty tile, it should not enter this step, skip to
  // Step 4

  // set thrd_tile_out_trans to all zero
  if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      thrd_tile_out_trans[j].clear();
    }
  }

  if (!empty_thrd_tile) {
    OVecCast tmp_output_c;
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
      if (i >= thread_tile_nrows) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        // Step 3: Store cast output
        CType scale_data = col_scale[i];

        OType scaled_elt =
            static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.ele[j]) * scale_data);
        tmp_output_c.data.ele[j] = scaled_elt;
        // Step 4: do transpose within thread tile
        if constexpr (RETURN_TRANSPOSE) {
          CType scale_t_data = row_scale[j];
          OType scaled_t_elt = static_cast<OType>(
              static_cast<CType>(thrd_tile_input[i].data.ele[j]) * scale_t_data);
          thrd_tile_out_trans[j].data.ele[i] = scaled_t_elt;
        }
      }
      tmp_output_c.EleStoreToIfNeeded(
          output_c + thread_tile_start_idx + i * row_length, 0, thread_tile_ncols);
    }

    if constexpr (RETURN_TRANSPOSE) {
      const size_t block_tile_t_start_idx =
          tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
      const size_t warp_tile_t_start_idx =
          block_tile_t_start_idx +
          warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
          warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
      const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                             tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                             tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
      for (int i = 0; i < thread_tile_ncols; i++) {
        thrd_tile_out_trans[i].EleStoreToIfNeeded(
            output_t + thread_tile_t_start_idx + i * num_rows, 0, thread_tile_nrows);
      }
    }
  }
}

template <
    bool RETURN_TRANSPOSE,
    bool IS_E8_SCALING,
    bool permute_scale,
    typename CType,
    typename IType,
    typename OType>
__global__ void
__launch_bounds__(THREADS_PER_BLOCK) block_scaled_1d_cast_transpose_kernel(
    const IType *const input,
    OType *const output_c,
    OType *const output_t,
    CType *const tile_scales_inv_c,
    CType *const tile_scales_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon,
    const __grid_constant__ CUtensorMap tensor_map_output_t) {
  static_assert(
      BLOCK_TILE_DIM == WARP_TILE_DIM_X,
      "Please make sure 1d block scaling kernel config BLOCK_TILE_DIM == WARP_TILE_DIM_X");

  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = RETURN_TRANSPOSE ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;
  const int tid_in_block_x = warp_id_in_block_x * NUM_THREADS_X_IN_WARP + tid_in_warp_x;

  // This is ONLY true if the input is a full tile
  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_idx =
      tile_id_y * BLOCK_TILE_DIM * row_length + tile_id_x * BLOCK_TILE_DIM;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;
  const size_t thread_tile_start_row_idx =
      tile_id_y * BLOCK_TILE_DIM + warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP +
      tid_in_warp_y * THREAD_TILE_DIM_Y;
  // const size_t thread_tile_start_col_idx = tile_id_x * BLOCK_TILE_DIM + warp_id_in_block_x *
  // THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP + tid_in_warp_x * THREAD_TILE_DIM_X;

  CType col_amax[THREAD_TILE_DIM_Y] = {0};
  CType row_amax[THREAD_TILE_DIM_X_] = {0};  // not used if not return transpose

  CType col_scale[THREAD_TILE_DIM_Y];
  CType row_scale[THREAD_TILE_DIM_X_];  // not used if not return transpose

// Step 1: Load a block tile of input data into thread tiles on registers
#pragma unroll
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
    thrd_tile_input[i].VecLoadFrom(input + thread_tile_start_idx + i * row_length);
  }

  // Step 2: calculate withint warp tile for amax and scale
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      // amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
      col_amax[i] = fmaxf(col_amax[i], fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
      if constexpr (RETURN_TRANSPOSE) {
        row_amax[j] =
            fmaxf(row_amax[j], fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
      }
    }
  }

  // Step 2.1: warp level & block level reduction on col_amax and row_amax, make sure they are amax
  // values across entire block
  unsigned int groupMask =
      0xFF << (tid_in_warp_y * 8);  // Activate only the threads in the current group
  for (int _i = 0; _i < THREAD_TILE_DIM_Y; _i++) {
    float groupMaxVal = groupMax<NUM_THREADS_X_IN_WARP, 1>(col_amax[_i], groupMask);
    col_amax[_i] = __shfl_sync(groupMask, groupMaxVal, tid_in_warp_y * NUM_THREADS_X_IN_WARP);
  }
  // Step 2.1.2: reduction of row_amax in x direction
  if constexpr (RETURN_TRANSPOSE) {
    // warp level reduction along y
    groupMask = WARP_REDUCE_AMAX_BY_Y_GROUP_MASKS[tid_in_warp_x];
    for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
      float groupMaxVal =
          groupMax<NUM_THREADS_Y_IN_WARP, NUM_THREADS_X_IN_WARP>(row_amax[_i], groupMask);
      row_amax[_i] = __shfl_sync(groupMask, groupMaxVal, tid_in_warp_x);
    }
    // then block level reduction using 2D smem, then reduction along one dim, no atomics
    __shared__ CType block_tile_row_amax_shared[NUM_WARPS_Y_IN_BLOCK][BLOCK_TILE_DIM];
    if (tid_in_warp_y == 0) {
#pragma unroll
      for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
        block_tile_row_amax_shared[warp_id_in_block_y]
                                  [_i * NUM_THREADS_X_IN_WARP + tid_in_warp_x] = row_amax[_i];
      }
    }
    __syncthreads();
    if (threadIdx.x < BLOCK_TILE_DIM) {
      CType final_amax = 0.0f;
#pragma unroll
      for (int _i = 0; _i < NUM_WARPS_IN_BLOCK; _i++) {
        final_amax = fmaxf(final_amax, block_tile_row_amax_shared[_i][threadIdx.x]);
      }
      block_tile_row_amax_shared[0][threadIdx.x] = final_amax;
    }
    __syncthreads();
#pragma unroll
    for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
      row_amax[_i] = block_tile_row_amax_shared[0][_i * NUM_THREADS_X_IN_WARP + tid_in_warp_x];
    }
  }

  __syncthreads();

// Step 2.2: all threads calculate scale
#pragma unroll
  for (int _i = 0; _i < THREAD_TILE_DIM_Y; _i++) {
    col_scale[_i] = ComputeScale<IType, OType, IS_E8_SCALING>(col_amax[_i], epsilon);
  }
  if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
    for (int _i = 0; _i < THREAD_TILE_DIM_X; _i++) {
      row_scale[_i] = ComputeScale<IType, OType, IS_E8_SCALING>(row_amax[_i], epsilon);
    }
  }

  // Step 2.3: save scale_inv by a subset of threads
  if (warp_id_in_block_x == 0 && tid_in_warp_x < THREAD_TILE_DIM_Y) {
    static_assert(std::is_same<CType, float>::value);
    // _i as linear index to loop over col_scale
    int _i = tid_in_warp_x;
    size_t row_idx = thread_tile_start_row_idx + _i;
    size_t col_idx = tile_id_x;
    CType scale_inv = 1.0 / col_scale[_i];
    if constexpr (permute_scale) {
      size_t p_row = row_idx / BLOCK_TILE_DIM;
      size_t p_col = col_idx;
      size_t p_dep = row_idx % BLOCK_TILE_DIM;
      size_t p_2d_stride = BLOCK_TILE_DIM * scale_stride_y;
      tile_scales_inv_c[p_row * p_2d_stride + p_col * BLOCK_TILE_DIM + p_dep] = scale_inv;
    } else {
      tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;
    }
  }
  // Step 2.4: save transposed version of scale_inv by a subset of threads
  if constexpr (RETURN_TRANSPOSE) {
    size_t begin = tid_in_warp_y * NUM_ELTS_EACH_THREAD_GRAB_BY_X;
    if (warp_id_in_block_y < NUM_ELTS_EACH_THREAD_GRAB_BY_X && warp_id_in_block_x == 0) {
      // _i as linear index to loop over row_scale
      int _i = begin + warp_id_in_block_y;
      size_t row_idx = tile_id_x * BLOCK_TILE_DIM + tid_in_block_x * THREAD_TILE_DIM_X + _i;
      size_t col_idx = tile_id_y;
      CType scale_inv = 1.0 / row_scale[_i];
      if constexpr (permute_scale) {
        size_t p_row = row_idx / BLOCK_TILE_DIM;
        size_t p_col = col_idx;
        size_t p_dep = row_idx % BLOCK_TILE_DIM;
        size_t p_2d_stride = BLOCK_TILE_DIM * scale_t_stride_y;
        tile_scales_inv_t[p_row * p_2d_stride + p_col * BLOCK_TILE_DIM + p_dep] = scale_inv;
      } else {
        tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
      }
    }
  }

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  OVecCast tmp_output_c;

  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      // Step 3: Store cast output
      CType scale_data = col_scale[i];

      OType scaled_elt =
          static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.ele[j]) * scale_data);
      tmp_output_c.data.ele[j] = scaled_elt;
      // Step 4: do transpose within thread tile
      if constexpr (RETURN_TRANSPOSE) {
        CType scale_t_data = row_scale[j];
        OType scaled_t_elt =
            static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.ele[j]) * scale_t_data);
        thrd_tile_out_trans[j].data.ele[i] = scaled_t_elt;
      }
    }
    tmp_output_c.VecStoreTo(output_c + thread_tile_start_idx + i * row_length);
  }

  // Step 4: store transpose into shared memory
  if constexpr (RETURN_TRANSPOSE) {
    // #ifdef TMA_HW_SUPPORTED
    //     __shared__ alignas(128)
    //         OVecTrans
    //         block_tile_trans_shared[SHARED_BLOCK_TILE_DIM_Y][SHARED_BLOCK_TILE_DIM_X_BANKS];
    //     OType(*block_tile_trans_shared_otype_ptr)[BLOCK_TILE_DIM] =
    //         reinterpret_cast<OType(*)[BLOCK_TILE_DIM]>(block_tile_trans_shared);

    // #pragma unroll
    //     for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
    //       auto warp_id_in_block_x_ = warp_id_in_block_y;
    //       auto warp_id_in_block_y_ = warp_id_in_block_x;
    //       int row_idx = warp_id_in_block_y_ * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
    //           tid_in_warp_x * THREAD_TILE_DIM_X + i;
    //       int col_idx =
    //           warp_id_in_block_x_ * (NUM_BANKS_Y_IN_WARP / NUM_BANKS_PER_SHARED_ELEM) +
    //           tid_in_warp_y;
    //       block_tile_trans_shared[row_idx][col_idx] = thrd_tile_out_trans[i];
    //     }

    //     // Wait for shared memory writes to be visible to TMA engine.
    //     cde::fence_proxy_async_shared_cta();
    //     __syncthreads();
    //     // After syncthreads, writes by all threads are visible to TMA engine.

    //     // Step 5: store transpose output
    //     // Initiate TMA transfer to copy shared memory to global memory
    //     if (threadIdx.x == 0) {
    //       cde::cp_async_bulk_tensor_2d_shared_to_global(
    //           &tensor_map_output_t,
    //           tile_id_y * BLOCK_TILE_DIM,
    //           tile_id_x * BLOCK_TILE_DIM,
    //           block_tile_trans_shared_otype_ptr);
    //       // Wait for TMA transfer to have finished reading shared memory.
    //       // Create a "bulk async-group" out of the previous bulk copy operation.
    //       cde::cp_async_bulk_commit_group();
    //       // Wait for the group to have completed reading from shared memory.
    //       cde::cp_async_bulk_wait_group_read<0>();
    //     }
    // #else
    // Step 4 Alternative (when TMA is not available, skip writing to shared memory)
    const size_t block_tile_t_start_idx =
        tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
    const size_t warp_tile_t_start_idx =
        block_tile_t_start_idx +
        warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
        warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
    const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                           tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                           tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
    for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
      thrd_tile_out_trans[i].VecStoreTo(output_t + thread_tile_t_start_idx + i * num_rows);
    }
    // #endif
  }
}

template <
    bool RETURN_TRANSPOSE,
    bool IS_E8_SCALING,
    typename CType,
    typename IType,
    typename OType>
__global__ void
__launch_bounds__(THREADS_PER_BLOCK) block_scaled_cast_transpose_kernel_notaligned(
    const IType *const input,
    OType *const output_c,
    OType *const output_t,
    CType *const tile_scales_inv_c,
    CType *const tile_scales_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  // shared mem for amax reduction in entire block, each warp produces one amax, there are
  // NUM_WARPS_IN_BLOCK amax to reduce
  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = RETURN_TRANSPOSE ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_row_idx = tile_id_y * BLOCK_TILE_DIM;
  const size_t block_tile_start_col_idx = tile_id_x * BLOCK_TILE_DIM;
  const size_t block_tile_start_idx =
      block_tile_start_row_idx * row_length + block_tile_start_col_idx;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  // handle non-full tile
  // check for three cases: full thread tile, nonfull thread tile, empty thread tile
  // for empty thread tile, directly write zero to the transposed shared mem buffer
  // for nonfull thread tile, fill zero to thread tile and act as if it's full
  const size_t thread_tile_start_row_idx =
      tile_id_y * BLOCK_TILE_DIM + warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP +
      tid_in_warp_y * THREAD_TILE_DIM_Y;
  const size_t thread_tile_start_col_idx =
      tile_id_x * BLOCK_TILE_DIM + warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
      tid_in_warp_x * THREAD_TILE_DIM_X;

  const size_t thread_tile_end_row_idx = thread_tile_start_row_idx + THREAD_TILE_DIM_Y - 1;
  const size_t thread_tile_end_col_idx = thread_tile_start_col_idx + THREAD_TILE_DIM_X - 1;

  bool full_thrd_tile =
      (thread_tile_end_row_idx < num_rows) && (thread_tile_end_col_idx < row_length);
  bool empty_thrd_tile =
      (thread_tile_start_row_idx >= num_rows) || (thread_tile_start_col_idx >= row_length);
  bool nonfull_thrd_tile = (!full_thrd_tile) && (!empty_thrd_tile);

  const size_t thread_tile_ncols =
      MIN(THREAD_TILE_DIM_X,
          (MIN(thread_tile_end_col_idx, row_length - 1) - thread_tile_start_col_idx + 1));
  const size_t thread_tile_nrows =
      MIN(THREAD_TILE_DIM_Y,
          (MIN(thread_tile_end_row_idx, num_rows - 1) - thread_tile_start_row_idx + 1));

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

  if (!empty_thrd_tile) {
    // Step 1: Load a block tile of input data into thread tiles on registers
    // Edge case: nonfull thread tile case, will use the partial load function here
    if (nonfull_thrd_tile) {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        if (i >= thread_tile_nrows) {
          thrd_tile_input[i].clear();
        } else {
          thrd_tile_input[i].EleLoadFromIfNeeded(
              input + thread_tile_start_idx + i * row_length, 0, thread_tile_ncols);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        thrd_tile_input[i].EleLoadFromIfNeeded(
            input + thread_tile_start_idx + i * row_length, 0, THREAD_TILE_DIM_X);
      }
    }

    // Step 2: calculate block tile amax and scale
    // Calculate thread_tile amax
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
      }
    }
  }
  // Reduce amax in the warp (32x32 tile)
  warp_tile_amax = WarpReduceMax<kThreadsPerWarp>(amax);
  // broadcast the amax to all threads in a warp from the lane 0
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  // reduce warp_tile_amax across multiple warps in a thread block using shared mem
  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  // only 8 elements needs reduction, if using reduction tree, multiple _syncthreads will be
  // needed, instead we just let thread 0 do the job
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale = ComputeScale<IType, OType, IS_E8_SCALING>(block_tile_amax, epsilon);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    if constexpr (RETURN_TRANSPOSE) {
      row_idx = tile_id_x;
      col_idx = tile_id_y;
      tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
    }
  }

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  // Edge case: in the non-full tile case, there are three subcases
  // for full thread tile, it's the same thing here
  // for nonfull thread tile, pay attention when saving tmp_output_c to global
  // memory, cannot VecStoreTo, but need to EleStoreToIfNeeded for empty tile,
  // it should not enter this step, skip to Step 4

  // set thrd_tile_out_trans to all zero
  if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      thrd_tile_out_trans[j].clear();
    }
  }

  if (!empty_thrd_tile) {
    OVecCast tmp_output_c;
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
      if (i >= thread_tile_nrows) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        // Step 3: Store cast output
        CType scale_data = block_tile_scale;

        OType scaled_elt =
            static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.ele[j]) * scale_data);
        tmp_output_c.data.ele[j] = scaled_elt;
        // Step 4: do transpose within thread tile
        if constexpr (RETURN_TRANSPOSE) {
          thrd_tile_out_trans[j].data.ele[i] = scaled_elt;
        }
      }
      tmp_output_c.EleStoreToIfNeeded(
          output_c + thread_tile_start_idx + i * row_length, 0, thread_tile_ncols);
    }

    if constexpr (RETURN_TRANSPOSE) {
      const size_t block_tile_t_start_idx =
          tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
      const size_t warp_tile_t_start_idx =
          block_tile_t_start_idx +
          warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
          warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
      const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                             tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                             tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
      for (int i = 0; i < thread_tile_ncols; i++) {
        thrd_tile_out_trans[i].EleStoreToIfNeeded(
            output_t + thread_tile_t_start_idx + i * num_rows, 0, thread_tile_nrows);
      }
    }
  }
}

template <
    bool RETURN_TRANSPOSE,
    bool IS_E8_SCALING,
    typename CType,
    typename IType,
    typename OType>
__global__ void
__launch_bounds__(THREADS_PER_BLOCK) block_scaled_cast_transpose_kernel(
    const IType *const input,
    OType *const output_c,
    OType *const output_t,
    CType *const tile_scales_inv_c,
    CType *const tile_scales_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon,
    const __grid_constant__ CUtensorMap tensor_map_output_t) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  // shared mem for amax reduction in entire block, each warp produces one amax, there are
  // NUM_WARPS_IN_BLOCK amax to reduce
  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = RETURN_TRANSPOSE ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  // This is ONLY true if the input is a full tile
  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_idx =
      tile_id_y * BLOCK_TILE_DIM * row_length + tile_id_x * BLOCK_TILE_DIM;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

// Step 1: Load a block tile of input data into thread tiles on registers
#pragma unroll
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
    thrd_tile_input[i].VecLoadFrom(input + thread_tile_start_idx + i * row_length);
  }

  // Step 2: calculate block tile amax and scale
  // Calculate thread_tile amax
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.ele[j])));
    }
  }
  // Reduce amax in the warp (32x32 tile)
  warp_tile_amax = WarpReduceMax<kThreadsPerWarp>(amax);
  // broadcast the amax to all threads in a warp from the lane 0
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  // reduce warp_tile_amax across multiple warps in a thread block using shared mem
  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  // only 8 elements needs reduction, if using reduction tree, multiple _syncthreads will be
  // needed, instead we just let thread 0 do the job
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale = ComputeScale<IType, OType, IS_E8_SCALING>(block_tile_amax, epsilon);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    if constexpr (RETURN_TRANSPOSE) {
      row_idx = tile_id_x;
      col_idx = tile_id_y;
      tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
    }
  }

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  OVecCast tmp_output_c;

  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      // Step 3: Store cast output
      CType scale_data = block_tile_scale;

      OType scaled_elt =
          static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.ele[j]) * scale_data);
      tmp_output_c.data.ele[j] = scaled_elt;
      // Step 4: do transpose within thread tile
      if constexpr (RETURN_TRANSPOSE) {
        thrd_tile_out_trans[j].data.ele[i] = scaled_elt;
      }
    }
    tmp_output_c.VecStoreTo(output_c + thread_tile_start_idx + i * row_length);
  }

  // Step 4: store transpose into shared memory
  if constexpr (RETURN_TRANSPOSE) {
    // #ifdef TMA_HW_SUPPORTED
    //     __shared__ alignas(128)
    //         OVecTrans
    //         block_tile_trans_shared[SHARED_BLOCK_TILE_DIM_Y][SHARED_BLOCK_TILE_DIM_X_BANKS];
    //     OType(*block_tile_trans_shared_otype_ptr)[BLOCK_TILE_DIM] =
    //         reinterpret_cast<OType(*)[BLOCK_TILE_DIM]>(block_tile_trans_shared);

    // #pragma unroll
    //     for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
    //       auto warp_id_in_block_x_ = warp_id_in_block_y;
    //       auto warp_id_in_block_y_ = warp_id_in_block_x;
    //       int row_idx = warp_id_in_block_y_ * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
    //           tid_in_warp_x * THREAD_TILE_DIM_X + i;
    //       int col_idx =
    //           warp_id_in_block_x_ * (NUM_BANKS_Y_IN_WARP / NUM_BANKS_PER_SHARED_ELEM) +
    //           tid_in_warp_y;
    //       block_tile_trans_shared[row_idx][col_idx] = thrd_tile_out_trans[i];
    //     }

    //     // Wait for shared memory writes to be visible to TMA engine.
    //     cde::fence_proxy_async_shared_cta();
    //     __syncthreads();
    //     // After syncthreads, writes by all threads are visible to TMA engine.

    //     // Step 5: store transpose output
    //     // Initiate TMA transfer to copy shared memory to global memory
    //     if (threadIdx.x == 0) {
    //       cde::cp_async_bulk_tensor_2d_shared_to_global(
    //           &tensor_map_output_t,
    //           tile_id_y * BLOCK_TILE_DIM,
    //           tile_id_x * BLOCK_TILE_DIM,
    //           block_tile_trans_shared_otype_ptr);
    //       // Wait for TMA transfer to have finished reading shared memory.
    //       // Create a "bulk async-group" out of the previous bulk copy operation.
    //       cde::cp_async_bulk_commit_group();
    //       // Wait for the group to have completed reading from shared memory.
    //       cde::cp_async_bulk_wait_group_read<0>();
    //     }
    // #else
    // Step 4 Alternative (when TMA is not available, skip writing to shared memory)
    const size_t block_tile_t_start_idx =
        tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
    const size_t warp_tile_t_start_idx =
        block_tile_t_start_idx +
        warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
        warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
    const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                           tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                           tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
    for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
      thrd_tile_out_trans[i].VecStoreTo(output_t + thread_tile_t_start_idx + i * num_rows);
    }
    // #endif
  }
}

void
block_scaled_1d_cast_transpose_impl(
    void *input,
    void *output,
    void *output_t,
    void *scale_inv,
    void *scale_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon,
    dim3 grid,
    const bool return_transpose,
    cudaStream_t stream) {
  using InputType = nv_bfloat16;
  using OutputType = __nv_fp8_e4m3;
  constexpr bool kPow2Scale = false;
  constexpr bool kPermuteScale = true;

  const bool full_tile = row_length % BLOCK_TILE_DIM == 0 && num_rows % BLOCK_TILE_DIM == 0;
  FLUX_DISPATCH_BOOL(
      return_transpose,
      kReturnTranspose,

      if (full_tile) {
        CUtensorMap tensor_map_output_trans;
        if constexpr (kReturnTranspose) {
          tensor_map_output_trans = get_tensor_map<OutputType>(&output_t, num_rows, row_length);
        }

        block_scaled_1d_cast_transpose_kernel<
            kReturnTranspose,
            kPow2Scale,
            kPermuteScale,
            float,
            InputType,
            OutputType><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const InputType *>(input),
            reinterpret_cast<OutputType *>(output),
            reinterpret_cast<OutputType *>(output_t),
            reinterpret_cast<float *>(scale_inv),
            reinterpret_cast<float *>(scale_inv_t),
            row_length,
            num_rows,
            scale_stride_x,
            scale_stride_y,
            scale_t_stride_x,
            scale_t_stride_y,
            epsilon,
            tensor_map_output_trans);
      } else {
        block_scaled_1d_cast_transpose_kernel_notaligned<
            kReturnTranspose,
            kPow2Scale,
            kPermuteScale,
            float,
            InputType,
            OutputType><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const InputType *>(input),
            reinterpret_cast<OutputType *>(output),
            reinterpret_cast<OutputType *>(output_t),
            reinterpret_cast<float *>(scale_inv),
            reinterpret_cast<float *>(scale_inv_t),
            row_length,
            num_rows,
            scale_stride_x,
            scale_stride_y,
            scale_t_stride_x,
            scale_t_stride_y,
            epsilon);
      })
}

void
block_scaled_cast_transpose_impl(
    void *input,
    void *output,
    void *output_t,
    void *scale_inv,
    void *scale_inv_t,
    const size_t row_length,
    const size_t num_rows,
    const size_t scale_stride_x,
    const size_t scale_stride_y,
    const size_t scale_t_stride_x,
    const size_t scale_t_stride_y,
    const float epsilon,
    dim3 grid,
    const bool return_transpose,
    cudaStream_t stream) {
  using InputType = nv_bfloat16;
  using OutputType = __nv_fp8_e4m3;

  constexpr bool kPow2Scale = false;

  const bool full_tile = row_length % BLOCK_TILE_DIM == 0 && num_rows % BLOCK_TILE_DIM == 0;

  FLUX_DISPATCH_BOOL(
      return_transpose,
      kReturnTranspose,

      if (full_tile) {
        CUtensorMap tensor_map_output_trans;
        if constexpr (kReturnTranspose) {
          tensor_map_output_trans = get_tensor_map<OutputType>(&output_t, num_rows, row_length);
        }

        block_scaled_cast_transpose_kernel<
            kReturnTranspose,
            kPow2Scale,
            float,
            InputType,
            OutputType><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const InputType *>(input),
            reinterpret_cast<OutputType *>(output),
            reinterpret_cast<OutputType *>(output_t),
            reinterpret_cast<float *>(scale_inv),
            reinterpret_cast<float *>(scale_inv_t),
            row_length,
            num_rows,
            scale_stride_x,
            scale_stride_y,
            scale_t_stride_x,
            scale_t_stride_y,
            epsilon,
            tensor_map_output_trans);
      } else {
        block_scaled_cast_transpose_kernel_notaligned<
            kReturnTranspose,
            kPow2Scale,
            float,
            InputType,
            OutputType><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<const InputType *>(input),
            reinterpret_cast<OutputType *>(output),
            reinterpret_cast<OutputType *>(output_t),
            reinterpret_cast<float *>(scale_inv),
            reinterpret_cast<float *>(scale_inv_t),
            row_length,
            num_rows,
            scale_stride_x,
            scale_stride_y,
            scale_t_stride_x,
            scale_t_stride_y,
            epsilon);
      })
}
}  // namespace flux
}  // namespace bytedance
