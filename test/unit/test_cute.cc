//===- test_cute.cc ----------------------------------------------- C++ ---===//
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

#include <iostream>
#include "cutlass/cutlass.h"
#include "cute/config.hpp"
#include "cute/util/type_traits.hpp"

#include "cute/pointer.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

namespace bytedance::flux {
using namespace cute;

template <typename T>
auto
print_v(T const &val, char const *name) {
  print("%s: ", name);
  print(val);
  print("\n");
}

void
test_cute() {
  print_v(
      composition(make_layout(make_shape(2, 4)), make_layout(make_shape(2, 4), make_stride(4, 1))),
      "compo");
  print("test_cute\n");
  using TileLayout = Layout<Shape<_128, _256>, Stride<_256, _1>>;
  using EpiTile = Shape<_64, _64>;
  auto tensor = make_counting_tensor(TileLayout{});
  auto epi_tile = EpiTile{};
  // print_v(zipped_divide(tensor, epi_tile), "zipped_div");
  // print_v(tiled_divide(tensor, epi_tile), "tiled_div");
  // print_v(logical_divide(tensor, epi_tile), "logical_div");

  auto layout = make_layout(make_shape(_32{}, _8{}), make_stride(_8{}, _1{}));
  auto composed = right_inverse(layout).compose(layout);
  print_v(composed.compose(composed, _), "right_inverse(layout).compose(layout)");
  print_v(right_inverse(layout).compose(layout), "right_inverse(layout).compose(layout)");
  // using Layout_TV = Layout<Shape<Shape<_16, _2>, _8>, Stride<Stride<_128, _1>, _16>>;
  // auto layout = Layout_TV{};
  // for (int i = 0; i < 64; ++i) {
  //   for (int j = 0; j < 8; ++j) {
  //     printf("%d,%d: %d\n", i, j, layout(i, j));
  //   }
  // }
  print_v(zip(layout), "zip(layout)");
  print_v(rank(zip(layout)), "rank(zip(layout))");
  constexpr int ts = tuple_size<Shape<_1, Shape<_1, _1>>>::value;
  print_v(ts, "ts");

  print_v(
      product_each(make_shape(make_shape(2, 3), make_shape(5))), "product_each(shape(layout))");
  print_v(make_basis_like(make_shape(make_shape(2, 3), make_shape(5))), "make_basis_like");

  print("\n");
}

void
test_sm80_tiled_mma() {
  print("test_sm80_tiled_mma\n");
  using MmaAtom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMma = TiledMMA<MmaAtom, Layout<Shape<_4, _2, _1>>, Tile<_64, _64, _16>>;
  auto tiled_mma = TiledMma{};
  print_v(tiled_mma, "tiled_mma");
  print_v(typename TiledMma::AtomLayoutC_TV{}, "typename TiledMma::AtomLayoutC_TV{}");
  print("\n");
}

void
test_sm80_tiled_copy_epilogue() {
  print("test_sm80_tiled_copy_epilogue\n");
  using TileShape = Shape<_256, _128, _32>;
  using TileShapeS2R = Shape<_16, _128>;
  // using Layout_TV = Layout<Shape<_32, _8>, Stride<_8, _1>>;
  using Layout_TV = Layout<Shape<Shape<_16, _2>, _8>, Stride<Stride<_128, _1>, _16>>;
  using Element = cutlass::half_t;
  using ElementAccumulator = float;
  using TiledCopyS2R =
      TiledCopy<Copy_Atom<DefaultCopy, ElementAccumulator>, Layout_TV, TileShapeS2R>;
  using SmemLayoutAtom =
      decltype(composition(Swizzle<3, 2, 3>{}, Layout<Shape<_8, _8>, Stride<_8, _1>>{}));
  using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<_256, _128>{}));

  auto [M, N, L] = make_tuple(512, 512, 1);
  auto [m_coord, n_coord, l_coord] = make_tuple(0, 0, 0);

  using LayoutD = cutlass::layout::RowMajor;
  using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, L));
  print_v(stride_D, "stride_D");
  auto mD_mnl = make_counting_tensor(make_layout(make_shape(M, N, L), stride_D));
  print_v(mD_mnl, "mD_mnl");
  using X = Underscore;
  Tensor gD_mnl = local_tile(mD_mnl, TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});
  print_v(gD_mnl, "gD_mnl");
  Tensor gD = gD_mnl(_, _, m_coord, n_coord, l_coord);
  print_v(gD, "gD");
  // Tensor sC = make_tensor(nullptr, SmemLayout{});
  // print_v(sC, "sC");
  auto tile = make_shape(size<0>(SmemLayout{}.shape()), size<1>(SmemLayout{}.shape()));
  print_v(tile, "tile");
  Tensor gDt = flat_divide(gD, tile);
  print_v(gDt, "gDt");
  auto tiled_s2r = TiledCopyS2R{};
  auto tD = tiled_s2r.get_thread_slice(32);
  print_v(tD, "tD");

  print_v(TiledCopyS2R::tidfrg_D(gDt.layout()), "TiledCopy::tidfrg_D(dtensor.layout())");
  {
    using Tiler_MN = typename TiledCopyS2R::Tiler_MN;
    using AtomLayoutRef = typename TiledCopyS2R::AtomLayoutRef;
    using AtomLayoutDst = typename TiledCopyS2R::AtomLayoutDst;
    auto tensor = zipped_divide(gDt, Tiler_MN{});
    print_v(tensor, "tensor");
    auto ref2trg = right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{});
    print_v(ref2trg, "ref2trg");
    auto atom_layout_TV = zipped_divide(
        typename TiledCopyS2R::TiledLayout_TV{},
        make_shape(typename TiledCopyS2R::AtomNumThr{}, typename TiledCopyS2R::AtomNumVal{}));
    print_v(atom_layout_TV, "atom_layout_TV");
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    print_v(trg_layout_TV, "trg_layout_TV");
    print_v(zip(trg_layout_TV), "zip(trg_layout_TV)");
    print_v(get<0>(zip(trg_layout_TV)), "get<0>(zip(trg_layout_TV))");
    print_v(rank(trg_layout_TV), "rank(trg_layout_TV)");
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1, Shape<_1, _1>>{});
    print_v(thrval2mn, "thrval2mn");
    print_v(tensor, "tensor");
    auto tv_tensor = tensor.compose(thrval2mn, _);
    print_v(tv_tensor, "tv_tensor");
    print_v(tv_tensor(make_coord(_, _), _), "tv_tensor(make_coord(_,_), _)");
    auto layout_0 = Layout<Shape<_16, _128>, Stride<_128, _1>>{};
    auto layout_1 = Layout<Shape<Shape<_16, _2>, _8>, Stride<Stride<_128, _1>, _16>>{};
    auto layout_compose = layout_0.compose(layout_1);
    print_v(layout_compose, "layout_compose");
    print_v(layout_compose(0, 0), "layout_compose(0, 0)");
    print_v(layout_compose(1, 0), "layout_compose(1, 0)");
    print_v(layout_compose(32, 0), "layout_compose(32, 0)");
  }
  {
    auto tD_1 = tiled_s2r.get_thread_slice(1);
    Tensor tDgD_1 = tD_1.partition_D(gDt);
    print_v(tDgD_1.data(), "tDgD_1.data()");
  }
  {
    auto tD_31 = tiled_s2r.get_thread_slice(31);
    Tensor tDgD_31 = tD_31.partition_D(gDt);
    print_v(tDgD_31.data(), "tDgD_31.data()");
  }

  Tensor tDgD = tD.partition_D(gDt);
  print_v(tDgD.data(), "tDgD.data()");
  print_v(tDgD, "tDgD");
}

void
test_sm90_tiled_copy() {}

void
test_epilogue_tile_contiguous() {
  print("test_epilogue_tile_contiguous\n");
  using TileShape = Shape<_128, _256>;
  using EpilogueTile = Shape<_128, _64>;
  int M = 4096;
  int N = 12288;
  using EpilogueTileLayout = Layout<EpilogueTile, Stride<_64, _1>>;
  auto [epi_tile_m, epi_tile_n] = EpilogueTile{};
  auto stride = make_stride(epi_tile_n, _1{}, epi_tile_m * N, epi_tile_m * epi_tile_n, M * N);
  print_v(stride, "stride");
  auto tma_shape = make_shape(epi_tile_m, epi_tile_n, M / epi_tile_m, N / epi_tile_n, 1);
  print_v(tma_shape, "tma_shape");
  Tensor mD_reshaped = make_counting_tensor(make_layout(tma_shape, stride));
  print_v(mD_reshaped, "mD_reshaped");
  auto mD_layout = make_layout(
      select<0, 2>(mD_reshaped.layout()),
      select<1, 3>(mD_reshaped.layout()),
      select<4>(mD_reshaped.layout()));
  print_v(mD_layout, "mD_layout");
  Tensor mD = make_tensor(mD_reshaped.data(), mD_layout);
  // Tensor mD = mD_reshaped.compose(mD_layout);
  print_v(mD, "mD");
  Tensor gD = local_tile(mD, TileShape{}, make_coord(0, 0, 0));
  print_v(gD, "gD");
  Tensor gD_epi = flat_divide(gD, EpilogueTile{});
  print_v(gD_epi, "gD_epi");
  print("\n");
}

void
test_smem_layout() {
  print("test_smem_layout:\n");
  using Element = cutlass::half_t;
  using SmemLayoutAtom = ComposedLayout<
      Swizzle<3, 4, 3>,
      smem_ptr_flag_bits<sizeof_bits_v<Element>>,
      Layout<Shape<_8, _64>, Stride<_64, _1>>>;
  using TileShape = Shape<_128, _64>;
  using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{}, TileShape{}, Step<_2, _1>{}));
  using SmemLayoutTmaReshape = decltype(SmemLayout{}.with_shape(Shape<_256, _32>{}));
  print_v(SmemLayout{}, "SmemLayout{}");
  print_v(SmemLayoutTmaReshape{}, "SmemLayoutTmaReshape{}");
  print("\n");
  int M = 48;
  int tile_M = 2;
  int nnodes = 2;
  int local_world_size = 4;
  int world_size = nnodes * local_world_size;
  int m_tile_per_rank = M / world_size;

  auto m_layout = make_layout(
      make_shape(m_tile_per_rank, nnodes, local_world_size),
      make_stride(1, local_world_size * m_tile_per_rank, m_tile_per_rank));

  auto swizzle = composition(Swizzle<3, 0, 3>{}, Layout<Shape<_8, _8>>{});
  auto squeeze = Layout<Shape<_8, _8>, Stride<_1, _0>>{};

  int rank = 1;
  for (int m_idx = 0; m_idx < M; ++m_idx) {
    auto [offset, node_idx, local_rank] = m_layout.get_hier_coord(m_idx);
    int local_rank_swizzle = squeeze(swizzle(local_rank, rank));
    int m_swizzle = m_layout(offset, node_idx, local_rank_swizzle);
    print(
        "m_idx:%d -> (%d,%d,%d->%d) -> %d\n",
        m_idx,
        offset,
        node_idx,
        local_rank,
        local_rank_swizzle,
        m_swizzle);
  }

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      print(squeeze(swizzle(j, i)));
      print(" ");
    }
    print("\n");
  }

  // auto tmp_layout = make_layout(make_shape(3, 4, 5, 6, 7));
  // print_v(group<3, 5>(tmp_layout)(_, _, _, 0), "group<3, 5>(tmp_layout)");
  print("\n");
}

void
test_partition() {
  print("test_partition:\n");
  using TileShape = Shape<_128, _256, _64>;
  auto cD = make_identity_tensor(take<0, 2>(TileShape{}));  // (TILE_M,TILE_N)
  constexpr int ThreadCount = 256;
  constexpr int Alignment = 8;
  int thread_idx = 0;
  using Element = cutlass::half_t;

  auto tiled_g2r = make_tiled_copy(
      Copy_Atom<DefaultCopy, Element>{},
      make_layout(make_shape(_8{}, _32{}), make_stride(_32{}, _1{})),
      make_layout(make_shape(_1{}, Int<Alignment>{}), make_stride(_0{}, _1{})));
  auto thread_g2r = tiled_g2r.get_slice(thread_idx);
  Tensor tDcD = thread_g2r.partition_S(cD);  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
  print_v(tiled_g2r, "tiled_g2r");
  print_v(tDcD, "tDcD");
  print("\n");
}

}  // namespace bytedance::flux

int
main() {
  using namespace bytedance::flux;
  test_cute();
  test_sm80_tiled_mma();
  test_sm80_tiled_copy_epilogue();
  test_epilogue_tile_contiguous();
  test_smem_layout();
  test_partition();
}
