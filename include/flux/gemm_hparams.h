//===- gemm_hparams.h --------------------------------------------- C++ ---===//
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
#include "cute/container/tuple.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "cute/layout.hpp"

namespace bytedance::flux {
using UnifiedTileShape = cute::tuple<int64_t, int64_t, int64_t>;
template <class T, __CUTE_REQUIRES(cute::is_tuple<T>::value)>
constexpr UnifiedTileShape
unify_type(T const &tile_shape) {
  static_assert(cute::tuple_size_v<T> == 3, "tile_shape requires tuple_size == 3");
  return cute::make_tuple(
      static_cast<int64_t>(cute::size<0>(tile_shape)),
      static_cast<int64_t>(cute::size<1>(tile_shape)),
      static_cast<int64_t>(cute::size<2>(tile_shape)));
}

/////////////////////////////////////////////////////
// Impl specific gemm hparams
/////////////////////////////////////////////////////
template <class... Ts>
struct GemmV2HParams : public FluxNamedTupleBase<GemmV2HParams, Ts...> {
  using Base = FluxNamedTupleBase<GemmV2HParams, Ts...>;
  using Base::Base;
  static constexpr char const *Name = "GemmV2HParams";
  static constexpr char const *LowerName = "gemm_v2_hparams";
  static constexpr std::array<char const *, 3> Fields = {
      "warp_shape", "instruction_shape", "streamk_mode"};

  FLUX_NAMED_TUPLE_DEFINE_FIELD(warp_shape, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(instruction_shape, 1)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(streamk_mode, 2)

  friend GemmV2HParams<UnifiedTileShape, UnifiedTileShape, GemmStreamkModeEnum>
  unify_type(GemmV2HParams const &obj) {
    return cute::make_tuple(
        unify_type(obj.warp_shape()),
        unify_type(obj.instruction_shape()),
        unify_type(obj.streamk_mode()));
  }
};

template <class WarpShape, class InstructionShape, class StreamkMode = _StreamkSK>
constexpr GemmV2HParams<WarpShape, InstructionShape, StreamkMode>
make_gemm_v2_hparams(
    WarpShape const &warp_shape,
    InstructionShape const &instruction_shape,
    StreamkMode const &streamk_mode = _StreamkSK{}) {
  return {cute::make_tuple(warp_shape, instruction_shape, streamk_mode)};
}

template <class... Ts>
constexpr GemmV2HParams<Ts...>
to_gemm_v2_hparams(cute::tuple<Ts...> const &tuple) {
  return {tuple};
}

template <class... Ts>
struct GemmV3HParams : public FluxNamedTupleBase<GemmV3HParams, Ts...> {
  using Base = FluxNamedTupleBase<GemmV3HParams, Ts...>;
  using Base::Base;
  static constexpr char const *Name = "GemmV3HParams";
  static constexpr char const *LowerName = "gemm_v3_hparams";
  static constexpr std::array<char const *, 2> Fields = {"cluster_shape", "kernel_schedule"};

  FLUX_NAMED_TUPLE_DEFINE_FIELD(cluster_shape, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(kernel_schedule, 1)

  friend GemmV3HParams<UnifiedTileShape, GemmKernelScheduleEnum>
  unify_type(GemmV3HParams const &obj) {
    return cute::make_tuple(unify_type(obj.cluster_shape()), unify_type(obj.kernel_schedule()));
  }
};

template <class ClusterShape, class KernelSchedule = _Cooperative>
constexpr GemmV3HParams<ClusterShape, KernelSchedule>
make_gemm_v3_hparams(
    ClusterShape const &cluster_shape, KernelSchedule const &kernel_schedule = _Cooperative{}) {
  return {cute::make_tuple(cluster_shape, kernel_schedule)};
}

template <class... Ts>
constexpr GemmV3HParams<Ts...>
to_gemm_v3_hparams(cute::tuple<Ts...> const &tuple) {
  return {tuple};
}

/////////////////////////////////////////////////////
// CommOp specific gemm hparams
/////////////////////////////////////////////////////
/// Tunable Hyper-Parameters

template <class... Ts>
struct GatherRSHParams : FluxNamedTupleBase<GatherRSHParams, Ts...> {
  using Base = FluxNamedTupleBase<GatherRSHParams, Ts...>;
  using Base::Base;

  static constexpr char const *Name = "GatherRSHParams";
  static constexpr char const *LowerName = "gather_rs_hparams";
  static constexpr std::array<char const *, 2> Fields = {"gather_rs_ctas", "n_dim_per_split"};

  FLUX_NAMED_TUPLE_DEFINE_FIELD(gather_rs_ctas, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(n_dim_per_split, 1)

  friend GatherRSHParams<int, int>
  unify_type(GatherRSHParams const &obj) {
    return cute::make_tuple(int(obj.gather_rs_ctas()), int(obj.n_dim_per_split()));
  }
};

template <class GatherRSCTAs = cute::Int<20>, class NDIM_PER_S = cute::_1024>
constexpr GatherRSHParams<GatherRSCTAs, NDIM_PER_S>
make_gather_rs_hparams(
    GatherRSCTAs const &nctas = cute::Int<20>{}, NDIM_PER_S ndim_per_s = cute::_1024{}) {
  return {cute::make_tuple(nctas, ndim_per_s)};
}

template <class... Ts>
constexpr GatherRSHParams<Ts...>
to_gather_rs_hparams(cute::tuple<Ts...> const &tup) {
  return {tup};
}

/////////////////////////////////////////////////////
// GemmHParams: params can change for better
// better performance
/////////////////////////////////////////////////////
using UnifiedImplHParams =
    std::variant<None, unified_type_t<GemmV2HParams>, unified_type_t<GemmV3HParams>>;
using UnifiedCommHParams = std::variant<None, unified_type_t<GatherRSHParams>>;

template <class... Ts>
struct GemmHParams : FluxNamedTupleBase<GemmHParams, Ts...> {
 public:
  using Base = FluxNamedTupleBase<GemmHParams, Ts...>;
  using Base::Base;
  static constexpr char const *Name = "GemmHParams";
  static constexpr char const *LowerName = "gemm_hparams";
  static constexpr std::array<char const *, 6> Fields = {
      "impl_spec", "comm_spec", "tile_shape", "gemm_kind", "mainloop_stage", "raster_order"};
  FLUX_NAMED_TUPLE_DEFINE_FIELD(impl_spec, 0)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(comm_spec, 1)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(tile_shape, 2)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(gemm_kind, 3)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(mainloop_stage, 4)
  FLUX_NAMED_TUPLE_DEFINE_FIELD(raster_order, 5)

  constexpr bool
  is_materialized() const noexcept {
    bool has_auto = is_auto_v<decltype(impl_spec())> or is_auto_v<decltype(comm_spec())> or
                    is_auto_v<decltype(tile_shape())> or is_auto_v<decltype(gemm_kind())> or
                    is_auto_v<decltype(mainloop_stage())> or is_auto_v<decltype(raster_order())>;
    return not has_auto;
  }

  using UnifiedGemmHParams = GemmHParams<
      UnifiedImplHParams,
      UnifiedCommHParams,
      UnifiedTileShape,
      GemmKindEnum,
      int,
      GemmRasterOrderEnum>;

  friend UnifiedGemmHParams
  unify_type(GemmHParams const &obj) {
    return cute::make_tuple(
        UnifiedImplHParams(unify_type(obj.impl_spec())),
        UnifiedCommHParams(unify_type(obj.comm_spec())),
        unify_type(obj.tile_shape()),
        unify_type(obj.gemm_kind()),
        int(obj.mainloop_stage()),
        unify_type(obj.raster_order()));
  }
};

using UnifiedGemmHParams = unified_type_t<GemmHParams>;

template <
    class ImplSpecific,
    class CommSpecific,
    class TileShape,
    class GemmKind = _GemmDefault,
    class MainloopStage = cute::_0,
    class RasterOrder = _RasterHeuristic>
constexpr GemmHParams<ImplSpecific, CommSpecific, TileShape, GemmKind, MainloopStage, RasterOrder>
make_gemm_hparams(
    ImplSpecific const &impl_spec = Auto{},
    CommSpecific const &comm_spec = Auto{},
    TileShape const &tile_shape = Auto{},
    GemmKind const &gemm_kind = _GemmDefault{},
    MainloopStage const &mainloop_stage = cute::_0{},
    RasterOrder const &raster_order = _RasterHeuristic{}) {
  return {
      cute::make_tuple(impl_spec, comm_spec, tile_shape, gemm_kind, mainloop_stage, raster_order)};
}

template <class... Ts>
constexpr GemmHParams<Ts...>
to_gemm_hparams(cute::tuple<Ts...> const &tup) {
  return {tup};
}

using _AutoHParams = GemmHParams<Auto, Auto, Auto, Auto, Auto, Auto>;

namespace detail {

template <class T>
struct is_gemm_hparams : std::false_type {};

template <class... Ts>
struct is_gemm_hparams<GemmHParams<Ts...>> : std::true_type {};
}  // namespace detail

template <class T>
inline constexpr bool is_gemm_hparams_v = detail::is_gemm_hparams<decay_and_strip_t<T>>::value;

// Create a tuple of GemmHParams by cartesian product
// of given sets of elements
template <
    class ImplSpecifics = cute::tuple<Auto>,
    class CommSpecifics = cute::tuple<Auto>,
    class TileShapes = cute::tuple<Auto>,
    class GemmKinds = cute::tuple<Auto>,
    class MainloopStages = cute::tuple<Auto>,
    class RasterOrders = cute::tuple<Auto>>
constexpr auto
make_space_gemm_hparams(
    ImplSpecifics const &impl_specs = cute::make_tuple(Auto{}),
    CommSpecifics const &comm_specifids = cute::make_tuple(Auto{}),
    TileShapes const &tile_shapes = cute::make_tuple(Auto{}),
    GemmKinds const &gemm_kinds = cute::make_tuple(Auto{}),
    MainloopStages const &mainloop_stages = cute::make_tuple(Auto{}),
    RasterOrders const &raster_orders = cute::make_tuple(Auto{})) {
  auto gemm_hparams_spaces = tuple_transform(
      tuple_cartesian_product(
          impl_specs, comm_specifids, tile_shapes, gemm_kinds, mainloop_stages, raster_orders),
      [](auto tup) { return to_gemm_hparams(tup); });
  return gemm_hparams_spaces;
}

/////////////////////////////////////////////////////
// Materialization of GemmHParams.
// Convert all Auto fields to specific values.
/////////////////////////////////////////////////////
namespace detail {
using namespace cute;

template <class... Ts>
constexpr auto
auto_impl_spec(GemmMeta<Ts...> meta) {
  if constexpr (meta.impl() == _GemmV2{} or meta.impl() == _GemmGroupedV2{}) {
    auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
    if constexpr (meta.arch() == _Sm89{} && dt_conf.is_input_fp8()) {
      return make_gemm_v2_hparams(Shape<_64, _32, _64>{}, Shape<_16, _8, _32>{});
    } else {
      return make_gemm_v2_hparams(Shape<_64, _64, _32>{}, Shape<_16, _8, _16>{});
    }
  } else if constexpr (meta.impl() == _GemmV3{} or meta.impl() == _GemmGroupedV3{}) {
    if constexpr (meta.arch() == _Sm80{}) {
      return make_gemm_v3_hparams(Shape<_1, _1, _1>{});
    } else if constexpr (meta.arch() == _Sm90{}) {
      return make_gemm_v3_hparams(Shape<_2, _1, _1>{});
    }
  } else {
    static_assert(cutlass::detail::dependent_false<decltype(meta.impl())>, "unsupported impl");
  }
}

template <class... Ts>
constexpr auto
auto_comm_spec(GemmMeta<Ts...> meta) {
  return None{};
}

template <class... Ts, class ImplHParams>
constexpr auto
auto_tile_shape(GemmMeta<Ts...> meta, ImplHParams impl_hparams) {
  if constexpr (meta.impl() == _GemmV2{} or meta.impl() == _GemmGroupedV2{}) {
    auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
    if constexpr (meta.arch() == _Sm89{} && dt_conf.is_input_fp8()) {
      return Shape<_128, _64, _64>{};
    } else {
      return Shape<_128, _128, _32>{};
    }
  } else if constexpr (meta.impl() == _GemmV3{} or meta.impl() == _GemmGroupedV3{}) {
    if constexpr (meta.arch() == _Sm80{}) {
      return Shape<_256, _128, _32>{};
    } else {
      auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
      using MDim = _128;
      using NDim = cute::conditional_t<
          (to_gemm_v3_meta(meta.impl_spec()).fast_accum()) or
              (to_gemm_v3_hparams(ImplHParams{}).kernel_schedule() == _PingPong{}),
          _128,
          _256>;
      using KDim =
          cute::Int<128 / cute::max(sizeof_dtype(dt_conf.a()), sizeof_dtype(dt_conf.b()))>;
      return Shape<MDim, NDim, KDim>{};
    }
  } else {
    static_assert(cutlass::detail::dependent_false<decltype(meta.impl())>, "unsupported impl");
  }
};

template <class... Ts>
constexpr auto
auto_gemm_kind(GemmMeta<Ts...> meta) {
  if constexpr (meta.impl() == _GemmV2{} or meta.impl() == _GemmGroupedV2{}) {
    return _GemmStreamK{};
  } else {
    return _GemmDefault{};
  }
};

template <class TileShape, class... Ts>
constexpr auto
auto_mainloop_stage(GemmMeta<Ts...> meta, TileShape const &) {
  if constexpr (
      (meta.impl() == _GemmV3{} or meta.impl() == _GemmGroupedV3{}) and meta.arch() == _Sm90{}) {
    return cute::_0{};  // Auto Stage Count
  } else if constexpr (meta.arch() == _Sm89{}) {
    auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
    if constexpr (dt_conf.is_input_fp8()) {
      return cute::_3{};
    } else {
      return cute::_4{};
    }
  } else {
    return cute::_4{};
  }
}

template <class... Ts>
constexpr auto
auto_raster_order(GemmMeta<Ts...> meta) {
  if constexpr (meta.arch() == _Sm89{}) {
    return _RasterAlongN{};
  } else {
    return _RasterHeuristic{};
  }
}

}  // namespace detail

template <class... Ts, class... Us>
constexpr auto
materialize_hparams(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  auto is_auto_or = [](auto v, auto w) {
    if constexpr (not is_auto_v<decltype(v)>) {
      return v;
    } else {
      return w;
    }
  };

  auto auto_impl_spec = detail::auto_impl_spec(meta);
  auto impl_spec = is_auto_or(hparams.impl_spec(), auto_impl_spec);
  auto auto_comm_spec = detail::auto_comm_spec(meta);
  auto auto_tile_shape = detail::auto_tile_shape(meta, impl_spec);
  auto tile_shape = is_auto_or(hparams.tile_shape(), auto_tile_shape);
  auto auto_gemm_kind = detail::auto_gemm_kind(meta);
  auto auto_mainloop_stage = detail::auto_mainloop_stage(meta, tile_shape);
  auto auto_raster_order = detail::auto_raster_order(meta);

  return make_gemm_hparams(
      impl_spec,
      is_auto_or(hparams.comm_spec(), auto_comm_spec),
      tile_shape,
      is_auto_or(hparams.gemm_kind(), auto_gemm_kind),
      is_auto_or(hparams.mainloop_stage(), auto_mainloop_stage),
      is_auto_or(hparams.raster_order(), auto_raster_order));
}

namespace detail {

template <class... Ts, class... Us>
constexpr bool
filter_arch(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
#if defined(__CUDACC__)
#if (__CUDACC_VER_MAJOR__ < 11)
  return false;
#endif
#if (__CUDACC_VER_MAJOR__ < 12)
  if (meta.arch() == _Sm90{} || meta.arch() == _Sm89{})
    return false;
#endif
#endif

  return true;
}

template <class... Ts, class... Us>
constexpr bool
filter_smem(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  auto [tile_m, tile_n, tile_k] = hparams.tile_shape();
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  int expect_min_smem = ((sizeof_dtype(dt_conf.a()) * tile_m * tile_k) +
                         (sizeof_dtype(dt_conf.b()) * tile_n * tile_k)) *
                        hparams.mainloop_stage();
  // print("!!!!!!!!!!!! expect min smem : %d\n", expect_min_smem);
  if (meta.arch() == _Sm80{} and expect_min_smem > 163 * 1024) {
    return false;
  }
  if (meta.arch() == _Sm89{} and expect_min_smem > 99 * 1024) {
    return false;
  }
  if (meta.arch() == _Sm90{} and expect_min_smem > 227 * 1024) {
    return false;
  }
  return true;
}

template <class... Ts, class... Us>
constexpr bool
filter_kernel_schedule(GemmMeta<Ts...> meta, GemmHParams<Us...> hparams) {
  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  if constexpr (meta.impl() == _GemmV3{} or meta.impl() == _GemmGroupedV3{}) {
    auto v3_hparams = to_gemm_v3_hparams(hparams.impl_spec());
    if (not dt_conf.is_input_fp8()) {
      return v3_hparams.kernel_schedule() == _Cooperative{};
    }
  }
  return true;
}

}  // namespace detail

// return tuple of (meta, materialized_hparams)
// and applying pre-defined filters
template <class... GemmMetaTs, class... GemmHParamsTs>
constexpr auto
make_space_meta_hparams_pair(
    cute::tuple<GemmMetaTs...> const &tup_meta, cute::tuple<GemmHParamsTs...> const &tup_hparams) {
  auto origin_space = tuple_cartesian_product(tup_meta, tup_hparams);
  auto materialized = tuple_transform(origin_space, [](auto const par) {
    auto [meta, hparams] = par;
    return cute::make_tuple(meta, materialize_hparams(meta, hparams));
  });

  return tuple_filter(materialized, [](auto const par) {
    auto meta = to_gemm_meta(cute::get<0>(par));
    auto hparams = materialize_hparams(meta, cute::get<1>(par));
    // apply predefined filters
    if constexpr (not detail::filter_arch(meta, hparams)) {
      return false;
    }
    if constexpr (not detail::filter_smem(meta, hparams)) {
      return false;
    }
    return true;
  });
}
}  // namespace bytedance::flux
