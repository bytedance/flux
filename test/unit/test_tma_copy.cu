//===- test_tma_copy.cu ------------------------------------------- C++ ---===//
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

#include "flux/cuda/cuda_common.h"
#include "cutlass/cutlass.h"
#include "cute/config.hpp"
#include "cute/pointer.hpp"
#include "cute/util/type_traits.hpp"

#include "cute/pointer.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/barrier.h"

namespace bytedance::flux {
using namespace cute;

template <typename T>
CUTLASS_HOST_DEVICE void
debug_print(T const &val, char const *name) {
#if defined(FLUX_DEBUG)
  if (thread0()) {
    print("* %s: ", name);
    print(val);
    print("\n");
  }
#endif
}

struct BaseOperator {};

struct TmaCopyOperator {
  using Element = cutlass::half_t;
  using TileShape = Shape<_128, _256>;

  /// MMA related
  using AtomLayoutMNK = Layout<Shape<_2, _1, _1>>;
  using MMA_Op = SM90_64x256x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(MMA_Op{}, AtomLayoutMNK{}));

  /// Epilogue related
  using EpilogueTile = Shape<_128, _64>;
  using DispatchPolicy = cutlass::epilogue::Sm90TmaWarpSpecialized<4, 2, 32, false, false>;
  using CopyOpG2S = SM90_TMA_LOAD;
  using CopyOpS2G = SM90_TMA_STORE;
  using CopyOpR2S = SM90_U32x4_STSM_N;

  using StrideD = Shape<int64_t, _1, int64_t>;
  // using StrideD = Shape<_1, int64_t, int64_t>;
  constexpr static bool is_m_major = cutlass::epilogue::collective::detail::is_m_major<StrideD>();

  /// TMA specific
  using SmemShapeTma = decltype(make_shape(
      max_common_vector(make_layout(get<0>(EpilogueTile{})), make_layout(get<0>(EpilogueTile{}))),
      max_common_vector(
          make_layout(get<1>(EpilogueTile{})), make_layout(get<1>(EpilogueTile{})))));
  using SmemLayoutAtom = ComposedLayout<
      Swizzle<3, 4, 3>,
      smem_ptr_flag_bits<sizeof_bits_v<Element>>,
      cute::conditional_t<
          is_m_major,
          Layout<Shape<_64, _8>, Stride<_1, _64>>,
          Layout<Shape<_8, _64>, Stride<_64, _1>>>>;
  using SmemLayoutTma = decltype(tile_to_shape(
      SmemLayoutAtom{},
      SmemShapeTma{},
      cute::conditional_t<is_m_major, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutTma{},
      make_shape(
          size<0>(shape(EpilogueTile{})),
          size<1>(shape(EpilogueTile{})),
          Int<DispatchPolicy::StagesD>{}),
      cute::conditional_t<is_m_major, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));
  // reshape to optimize tma copy performance
  using TMA_STORE = decltype(make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(static_cast<Element *>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      SmemLayoutTma{}));

  /// Kernel launch related
  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  struct Arguments {
    int m;
    int n;
    Element *ptr_D;
  };

  struct Params {
    int m;
    int n;
    StrideD dD;
    Element *ptr_D;
    TMA_STORE tma_store;
  };

  struct SharedStorage {
    alignas(cutlass::detail::alignment_for_swizzle(SmemLayout{}))
        array_aligned<Element, size(SmemLayout{})> smem_d;
  };

  static Params
  to_underlying_arguments(Arguments const &args) {
    Params params;
    params.m = args.m;
    params.n = args.n;
    params.ptr_D = static_cast<Element *>(args.ptr_D);
    auto shape = make_shape(args.m, args.n, 1);
    auto dD = cutlass::make_cute_packed_stride(StrideD{}, shape);
    params.dD = dD;
    auto tensor = make_tensor(params.ptr_D, make_layout(make_shape(args.m, args.n, 1), dD));
    params.tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor, SmemLayoutTma{});
    return params;
  }

  CUTLASS_DEVICE void
  operator()(Params const &params, char *smem_buf) {
    int thread_idx = threadIdx.x;
    auto tiled_mma = TiledMma{};
    auto blk_shape = TileShape{};
    auto epi_tile = EpilogueTile{};
    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    // Allocate dummy accumulators
    Tensor accumulators =
        partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));  // (MMA,MMA_M,MMA_N)
    debug_print(accumulators, "accumulators");

    // Create shared memory tensor to store D in EpilogueTile
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(static_cast<Element *>(storage.smem_d.data())),
        SmemLayout{}));  // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
    debug_print(sD_epi, "sD_epi");

    // Create copy atom from register to shmem
    using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
    TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);
    debug_print(tiled_copy_C_atom, "tiled_copy_C_atom");

    Tensor mD = params.tma_store.get_tma_tensor(make_shape(params.m, params.n, 1));  // (M,N,L)
    debug_print(mD, "mD");
    int m_coord = blockIdx.x;
    int n_coord = blockIdx.y;

    Tensor gD = local_tile(
        mD, take<0, 2>(blk_shape), make_coord(m_coord, n_coord, 0));  // (TILE_M, TILE_N)
    debug_print(gD, "gD");
    Tensor gD_epi = flat_divide(gD, epi_tile);  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    debug_print(gD_epi, "gD_epi");

    // partition
    Tensor tC_gD = cutlass::epilogue::fusion::sm90_partition_for_epilogue<true>(
        gD,
        epi_tile,
        tiled_copy_C_atom,
        thread_idx);  // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    debug_print(tC_gD, "tC_gD");

    // Register to Shared Memory
    auto tiled_r2s = make_tiled_copy_S(Copy_Atom<CopyOpR2S, Element>{}, tiled_copy_C_atom);

    // Allocate D registers

    ThrCopy thread_r2s = tiled_r2s.get_slice(thread_idx);
    Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);  // ((R2S,R2S_V),MMA_M,MMA_N)
    debug_print(tRS_rAcc, "tRS_rAcc");
    Tensor tRS_sD = thread_r2s.partition_D(sD_epi);  // (R2S,R2S_M,R2S_N,PIPE_D)
    debug_print(tRS_sD, "tRS_sD");

    Layout tRS_rD_layout = make_layout(take<0, 3>(shape(thread_r2s.partition_S(sD_epi))));
    debug_print(tRS_rD_layout, "tRS_rD_layout");
    Tensor tRS_rD = make_tensor<Element>(tRS_rD_layout);  // (R2S,R2S_M,R2S_N)
    debug_print(tRS_rD, "tRS_rD");
    fill(tRS_rD, get_block_val());

    ThrCopy thrblk_s2g = params.tma_store.get_slice(_0{});

    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);  // (TMA,TMA_M,TMA_N,PIPE)
    debug_print(bSG_sD, "bSG_sD");
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);  // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)
    debug_print(bSG_gD, "bSG_gD");

    // registers => smem
    int store_pipe_index = 0;
    auto synchronize = [&]() { cutlass::arch::NamedBarrier::sync(size(TiledMma{}), 0); };
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m) {
        copy(tiled_r2s, tRS_rD, tRS_sD(_, _, _, store_pipe_index));
        synchronize();
        // smem => global
        if (cutlass::canonical_warp_idx() == 0) {
          copy(params.tma_store, bSG_sD(_, _, _, store_pipe_index), bSG_gD(_, _, _, epi_m, epi_n));
        }
      }
    }
  }

  CUTLASS_DEVICE static Element
  get_block_val() {
    int blk_idx = blockIdx.x + blockIdx.y * gridDim.x;
    return Element(blk_idx);
  }

  static dim3
  get_block_shape() {
    dim3 block;
    block.x = MaxThreadsPerBlock;
    return block;
  }

  static dim3
  get_grid_shape(int m_blocks, int n_blocks) {
    dim3 grid;
    grid.x = m_blocks;
    grid.y = n_blocks;
    return grid;
  }

  static constexpr int
  get_smem_size() {
    return sizeof(SharedStorage);
  }
};

template <typename Operator>
__global__ void
check_output(CUTLASS_GRID_CONSTANT typename Operator::Params const params) {
  using Element = typename Operator::Element;
  auto tile = typename Operator::TileShape{};
  auto [m_tile, n_tile] = tile;

  int m_coord = blockIdx.x;
  int n_coord = blockIdx.y;

  auto shape = make_shape(params.m, params.n, Int<1>{});
  auto mD = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(params.m, params.n, 1), params.dD);
  auto gD = local_tile(mD, tile, make_coord(m_coord, n_coord, 0));

  for (int i = 0; i < m_tile; ++i) {
    for (int j = 0; j < n_tile; ++j) {
      float expected = Operator::get_block_val();
      float val = gD(i, j);
      float diff = fabs(expected - val);
      if (diff > 0.001) {
        print("%d,%d,%d,%d:%f,%f\n", m_coord, n_coord, i, j, val, expected);
        CUTLASS_ASSERT(false);
      }
    }
  }
}

}  // namespace bytedance::flux

int
main() {
  using namespace bytedance::flux;
  using Operator = TmaCopyOperator;
  constexpr int m_blocks = 32;
  constexpr int n_blocks = 48;
  constexpr int cur_dev = 0;
  constexpr int dst_dev = 1;

  auto [m_tile, n_tile] = typename Operator::TileShape{};
  int m = m_blocks * m_tile;
  int n = n_blocks * n_tile;

  dim3 const block = Operator::get_block_shape();
  dim3 const grid = Operator::get_grid_shape(m_blocks, n_blocks);
  constexpr int smem_size = Operator::get_smem_size();

  cudaSetDevice(dst_dev);
  cutlass::DeviceAllocation<typename Operator::Element> block_D;
  block_D.reset(m * n);

  cudaSetDevice(cur_dev);
  if (cur_dev != dst_dev) {
    cudaDeviceEnablePeerAccess(dst_dev, 0);
  }

  typename Operator::Arguments args;
  args.m = m;
  args.n = n;
  args.ptr_D = static_cast<typename Operator::Element *>(block_D.get());
  typename Operator::Params params = Operator::to_underlying_arguments(args);
  cudaStream_t stream = nullptr;
#if defined(FLUX_DEBUG)
  cutlass::device_kernel<Operator><<<grid, block, smem_size, stream>>>(params);
  cudaDeviceSynchronize();
  return 0;
#endif  // FLUX_DEBUG
  GpuTimer timer;
  timer.start(stream);
  constexpr int iters = 50;
  for (int i = 0; i < iters; ++i) {
    cutlass::device_kernel<Operator><<<grid, block, smem_size, stream>>>(params);
  }
  timer.stop();
  {
    dim3 block;
    block.x = 1;
    dim3 const grid = Operator::get_grid_shape(m_blocks, n_blocks);
    check_output<Operator><<<grid, block, 0, stream>>>(params);
    cudaStreamSynchronize(stream);
  }
  float elapsed_ms = timer.elapsed_millis() / iters;
  print("time elapsed %fus\n", elapsed_ms * 1000);
  int bytes = m * n * sizeof_bits_v<typename Operator::Element> / 8;
  float Gbytes = bytes * 1.0 / 1024 / 1024 / 1024;
  float bandwidth = Gbytes * 1000 / elapsed_ms;
  print("bandwidth: %f GB/s\n", bandwidth);
  return 0;
}
