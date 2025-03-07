//===- test_flux_templates.cc ------------------------------------ C++ ---===//
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

#include <map>
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"

namespace bytedance::flux {
using namespace cute;

void
test_tuple_cartesian_product() {
  auto tup_empty = make_tuple();
  auto tuple_one = make_tuple(0);
  auto tuple_two = make_tuple(1, 2);
  auto tuple_tup = make_tuple(make_tuple(0, 1), make_tuple(2));
  auto tuple_mixed = make_tuple(3, make_tuple(2, 3));
  FLUX_CHECK(tuple_cartesian_product(tup_empty, tuple_one) == tup_empty);
  FLUX_CHECK(tuple_cartesian_product(tuple_one, tuple_one) == make_tuple(make_tuple(0, 0)));
  FLUX_CHECK(
      tuple_cartesian_product(tuple_one, tuple_two) ==
      make_tuple(make_tuple(0, 1), make_tuple(0, 2)));
  FLUX_CHECK(
      tuple_cartesian_product(tuple_two, tuple_tup) == make_tuple(
                                                           make_tuple(1, make_tuple(0, 1)),
                                                           make_tuple(1, make_tuple(2)),
                                                           make_tuple(2, make_tuple(0, 1)),
                                                           make_tuple(2, make_tuple(2))));
  FLUX_CHECK(
      tuple_cartesian_product(tuple_tup, tuple_mixed) ==
      make_tuple(
          make_tuple(make_tuple(0, 1), 3),
          make_tuple(make_tuple(0, 1), make_tuple(2, 3)),
          make_tuple(make_tuple(2), 3),
          make_tuple(make_tuple(2), make_tuple(2, 3))));
}

void
test_tuple_foreach() {
  auto all_nums = make_tuple(_1{}, _2{}, _3{}, _4{});
  int sum = 0;
  int prod = 1;
  tuple_for_each(all_nums, [&](auto val) { sum += val(); });
  tuple_for_each(all_nums, [&](auto val) { prod *= val(); });
  FLUX_CHECK(sum == 10);
  FLUX_CHECK(prod == 24);
}

void
test_tuple_filter() {
  auto all_nums = make_tuple(_1{}, _2{}, _3{});
  auto even_nums = tuple_filter(all_nums, [](auto val) { return val() % 2 == 0; });
  auto odd_nums = tuple_filter(all_nums, [](auto val) { return val() % 2 != 0; });
  FLUX_CHECK(even_nums == make_tuple(_2{}));
  FLUX_CHECK(odd_nums == make_tuple(_1{}, _3{}));
}

void
test_tuple_split_slice() {
  auto all_nums = make_tuple(_1{}, _2{}, _3{}, _4{}, _5{});
  auto slice_0 = tuple_split_slice<0, 3>(all_nums);
  auto slice_1 = tuple_split_slice<1, 3>(all_nums);
  auto slice_2 = tuple_split_slice<2, 3>(all_nums);
  FLUX_CHECK(slice_0 == make_tuple(_1{}, _2{}));
  FLUX_CHECK(slice_1 == make_tuple(_3{}, _4{}));
  FLUX_CHECK(slice_2 == make_tuple(_5{}));
}

void
test_return_if() {
  auto dtype = _FP16{}();

  auto get_size = [](DataTypeEnum dtype) {
    return tuple_return_if(
        cute::make_tuple(_BF16{}, _FP16{}, _FP32{}, _Void{}),
        /*pred=*/[dtype](auto c_dtype) { return dtype == c_dtype; },
        /*found=*/[](auto c_dtype) { return sizeof_dtype(decltype(c_dtype){}); },
        /*not_fount=*/[]() { return -1; });
  };
  FLUX_CHECK(get_size(_BF16{}()) == 2);
  FLUX_CHECK(get_size(_FP16{}()) == 2);
  FLUX_CHECK(get_size(_FP32{}()) == 4);
  FLUX_CHECK(get_size(_Void{}()) == 0);
  FLUX_CHECK(get_size(_E4M3{}()) == -1);  // not supported
}

void
test_tuple_has_elem() {
  auto tup = make_tuple(0, _1{}, _2{}, _FP16{}, _BF16{}, make_tuple(_3{}));
  static_assert(tuple_has_elem(tup, _1{}));
  static_assert(tuple_has_elem(tup, make_tuple(_3{})));
  FLUX_CHECK(tuple_has_elem(tup, 0));
  FLUX_CHECK(tuple_has_elem(tup, _1{}));
  FLUX_CHECK(tuple_has_elem(tup, _2{}));
  FLUX_CHECK(tuple_has_elem(tup, _FP16{}));
  FLUX_CHECK(tuple_has_elem(tup, _BF16{}));
  FLUX_CHECK(not tuple_has_elem(tup, _1{}()));  // type different
  FLUX_CHECK(not tuple_has_elem(tup, 2));
  FLUX_CHECK(not tuple_has_elem(tup, 0.0));
  FLUX_CHECK(not tuple_has_elem(make_tuple(), 0));
}

void
test_gemm_meta() {
  auto dt_conf = make_gemm_dtype_config(_FP16{});
  auto meta = make_gemm_meta(dt_conf, _Sm90{}, _CommNone{}, _RCR{}, _GemmV3{});
  auto unified_meta = unify_type(meta);
  FLUX_CHECK_EQ(unified_meta.dtype(), unify_type(dt_conf));
  FLUX_CHECK_EQ(unified_meta.arch(), _Sm90{}());
  FLUX_CHECK_EQ(unified_meta.comm_op(), _CommNone{}());
  FLUX_CHECK_EQ(unified_meta.gemm_layout(), _RCR{}());
  FLUX_CHECK_EQ(unified_meta.impl(), _GemmV3{}());
}

void
test_gemm_hparams() {
  auto dt_conf = make_gemm_dtype_config(_FP16{});
  auto meta = make_gemm_meta(dt_conf, _Sm90{}, _CommNone{}, _RCR{}, _GemmV3{});
  auto auto_hparams = _AutoHParams{};
  auto hparams = materialize_hparams(meta, auto_hparams);
  auto unified_hparams = unify_type(hparams);
  FLUX_CHECK_EQ(unified_hparams.comm_spec(), None{});
  FLUX_CHECK_EQ(unified_hparams.raster_order(), _RasterHeuristic{}());
  FLUX_CHECK_EQ(unified_hparams.mainloop_stage(), 0);
}
}  // namespace bytedance::flux

int
main() {
  using namespace bytedance::flux;
  test_tuple_cartesian_product();
  test_tuple_foreach();
  test_tuple_filter();
  test_tuple_split_slice();
  test_return_if();
  test_gemm_meta();
  test_gemm_hparams();
  test_tuple_has_elem();
  return 0;
}
