//===- flux.h ----------------------------------------------------- C++ ---===//
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
#include <assert.h>
#include <cstdint>
#include <chrono>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <type_traits>
#include <variant>
#include <tuple>
#include <string>
#include <iostream>
#include <algorithm>
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/config.hpp"
#include "cute/container/tuple.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/helper_macros.hpp"

#ifndef FLUX_LIKELY
#define FLUX_LIKELY(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef FLUX_UNLIKELY
#define FLUX_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

namespace bytedance {
namespace flux {

constexpr int kMaxLocalWorldSize = 8;
constexpr int kMaxWorldSize = 32;
using index_t = int64_t;

// Print a type in compiling error message
// Example:
// auto a = xxx();
// StaticPrintT<decltype<a>>{};
template <typename... T>
struct StaticPrintT {
  static_assert(cutlass::detail::dependent_false<T...>, "StaticPrintT");
};

// Print an int value in compiling error message
// Example:
// constexpr int x = xxx();
// StaticPrintInt<x>{};
template <int I>
struct StaticPrintInt {
  static_assert(!(I == I), "StaticPrintInt");
};

// If Cond is true, make compilation fail and print type pack Ts
template <bool Cond, class... Ts>
struct StaticPrintTypesIf {
  static_assert(not Cond, "StaticPrintTypesIf");
};

// Print expr with format #expr: expr
#define FLUX_DEBUG_PRINT(expr) \
  {                            \
    cute::print(#expr);        \
    cute::print(": ");         \
    cute::print((expr));       \
    cute::print("\n");         \
  }

namespace detail {
class CheckFail {
 public:
  CheckFail() = default;

  // Overload the stream insertion operator.
  template <typename T>
  CheckFail &
  operator<<(const T &value) {
    message_ << value;
    return *this;
  }

  // Destructor that throws an exception with the accumulated error message.
  ~CheckFail() noexcept(false) {
    std::cerr << message_.str() << std::endl;
    throw std::runtime_error(message_.str());
  }

 private:
  std::ostringstream message_;
};
}  // namespace detail

// Macro to check a condition and stream a custom error message if the check fails.
// Note that this can only be used in host code
#define FLUX_CHECK(condition)            \
  if (!(condition))                      \
  ::bytedance::flux::detail::CheckFail() \
      << __FILE__ << ":" << __LINE__ << " Check failed: " #condition ". "

#define FLUX_CHECK_BINOP(lhs, rhs, op)                                                            \
  if (auto x = (lhs), y = (decltype(x))(rhs); FLUX_UNLIKELY(!(x op y)))                           \
  ::bytedance::flux::detail::CheckFail() << __FILE__ << ":" << __LINE__ << " Check failed: " << x \
                                         << "(" #lhs ") " #op " " << y << "(" #rhs ")"

#define FLUX_CHECK_EQ(lhs, rhs) FLUX_CHECK_BINOP((lhs), (rhs), ==)
#define FLUX_CHECK_NE(lhs, rhs) FLUX_CHECK_BINOP((lhs), (rhs), !=)
#define FLUX_CHECK_LT(lhs, rhs) FLUX_CHECK_BINOP((lhs), (rhs), <)
#define FLUX_CHECK_GT(lhs, rhs) FLUX_CHECK_BINOP((lhs), (rhs), >)
#define FLUX_CHECK_LE(lhs, rhs) FLUX_CHECK_BINOP((lhs), (rhs), <=)
#define FLUX_CHECK_GE(lhs, rhs) FLUX_CHECK_BINOP((lhs), (rhs), >=)
#define FLUX_CHECK_DIV(lhs, rhs)                                                                  \
  if (auto x = (lhs), y = (decltype(x))(rhs); FLUX_UNLIKELY(x % y != 0))                          \
  ::bytedance::flux::detail::CheckFail() << __FILE__ << ":" << __LINE__ << " Check failed: " << x \
                                         << "(" #lhs ") % " << y << "(" #rhs ") != 0"

// Convert T&,T&&, const T&... to basic T
template <typename T>
using decay_and_strip_t = std::remove_cv_t<std::remove_reference_t<std::decay_t<T>>>;

namespace detail {
// Useful to create non-deduced context
template <typename T>
struct identity {
  using type = T;
};
}  // namespace detail

template <typename T>
using identity_t = typename detail::identity<T>::type;

/////////////////////////////////////////////////////
// Tuple Algorithms
/////////////////////////////////////////////////////
// Basic tuple and index sequence operations
namespace detail {
template <std::size_t Offset, std::size_t... Is>
constexpr auto
offset_index_sequence(std::index_sequence<Is...>) {
  return std::index_sequence<(Offset + Is)...>{};
}

template <std::size_t... Is, class Tuple>
constexpr auto
tuple_slice(std::index_sequence<Is...>, Tuple &&tuple) {
  return cute::make_tuple(cute::get<Is>(cute::forward<Tuple>(tuple))...);
}

// Split a tuple into (head_tuple, tail_tuple) at index `I`
// eg: tuple_split(1, <1,2,3>) -> <1>, <2,3>
template <std::size_t I, class Tuple>
constexpr auto
tuple_split(Tuple &&tuple) {
  constexpr int tail_idx_len = cute::tuple_size_v<std::remove_reference_t<Tuple>> - I;
  return std::make_pair(
      tuple_slice(std::make_index_sequence<I>{}, cute::forward<Tuple>(tuple)),
      tuple_slice(
          offset_index_sequence<I>(std::make_index_sequence<tail_idx_len>{}),
          cute::forward<Tuple>(tuple)));
}
}  // namespace detail

// works like a shorthand for cute::tuple_size_v<T> on the tuple object
// without calling decltype to get the tuple's type
template <class... Ts>
constexpr int
tuple_length(cute::tuple<Ts...> const &) {
  return sizeof...(Ts);
}

// Append an element to tuple in the tail
template <class T, class... Ts>
constexpr auto
tuple_append(cute::tuple<Ts...> const &tup, T e) {
  return cute::tuple_cat(tup, cute::make_tuple(e));
}

// replace an element of `tup` at `Idx` to `new_val`
//
// Example:
//   tuple_replace_item<0>(tuple(0, 1), 1.0) => tuple(1.0,1)
template <std::size_t Idx, class NewT, class... Ts>
auto
tuple_replace_item(cute::tuple<Ts...> const &tup, NewT &&new_val) {
  constexpr int tup_size = sizeof...(Ts);
  static_assert(0 <= Idx && Idx < tup_size, "Idx out of range");
  auto before_index = std::make_index_sequence<Idx>{};
  auto after_index =
      detail::offset_index_sequence<Idx + 1>(std::make_index_sequence<tup_size - 1 - Idx>{});

  return cute::tuple_cat(
      detail::tuple_slice(before_index, tup),
      cute::make_tuple(cute::forward<NewT>(new_val)),
      detail::tuple_slice(after_index, tup));
}

namespace detail {
template <class F, std::size_t... Is, class... Ts>
constexpr auto
tuple_transform_impl(std::index_sequence<Is...>, cute::tuple<Ts...> const &tup, F &&func) {
  return cute::make_tuple(func(cute::get<Is>(tup))...);
}
}  // namespace detail
// transform elements of a tuple by function `F`
// e.g. for tuple (A,B), returns (F(A),F(B))
template <class F, class... Ts>
constexpr auto
tuple_transform(cute::tuple<Ts...> const &tup, F &&func) {
  return detail::tuple_transform_impl(
      std::make_index_sequence<sizeof...(Ts)>{}, tup, cute::forward<F>(func));
}

namespace detail {
template <class F, std::size_t... Is, class... Ts>
constexpr void
tuple_for_each_impl(std::index_sequence<Is...>, cute::tuple<Ts...> const &tup, F &&func) {
  (func(cute::get<Is>(tup)), ...);
}
}  // namespace detail
// Call function F(e) for each element e in tuple
template <class F, class... Ts>
constexpr void
tuple_for_each(cute::tuple<Ts...> const &tup, F &&func) {
  detail::tuple_for_each_impl(
      std::make_index_sequence<sizeof...(Ts)>{}, tup, cute::forward<F>(func));
}

namespace detail {
template <class Predicate, class T>
constexpr auto
tuple_filter_impl_element(Predicate pred, T) {
  static_assert(cute::is_static_v<T>, "tuple filter requires static elem");
  constexpr bool is_pred = pred(T{});
  if constexpr (is_pred) {
    return cute::make_tuple(T{});
  } else {
    return cute::make_tuple();
  }
}

template <class Predicate, std::size_t... Is, class... Ts>
constexpr auto
tuple_filter_impl(std::index_sequence<Is...>, cute::tuple<Ts...> const &tup, Predicate pred) {
  return cute::tuple_cat(tuple_filter_impl_element(pred, cute::get<Is>(tup))...);
}
}  // namespace detail
// Create a new tuple with elements in `tup` that satisfy Predicate(e) == true
// all elements in `tup` must be static values
// Example:
// tuple_filter([](int x) { return x % 2 == 0; }, tuple(_0{},_1{},_2{}) => tuple(_0{},_2{})
template <class Predicate, class... Ts>
constexpr auto
tuple_filter(cute::tuple<Ts...> const &tup, Predicate pred) {
  return detail::tuple_filter_impl(std::make_index_sequence<sizeof...(Ts)>{}, tup, pred);
}

namespace detail {
template <bool Intermediate, class... Ts, class... Us>
constexpr auto
tuple_cartesian_product_two(cute::tuple<Ts...> const &tup0, cute::tuple<Us...> const &tup1) {
  if constexpr (sizeof...(Ts) == 0 or sizeof...(Us) == 0) {
    return cute::make_tuple();
  } else {
    // 1. Unpack out the head of the first tuple, cross product it with the full second tuple
    // 2. Recursively calculate the cartesian product by the ramaining elements of the first tuple
    //    and the full second tuple.
    // 3. Concat the two results.
    auto const &split_tup = detail::tuple_split<1>(tup0);
    return cute::tuple_cat(
        tuple_transform(
            tup1,
            [&](auto const &a) {
              if constexpr (Intermediate) {
                return tuple_append(cute::get<0>(split_tup.first), a);
              } else {
                return tuple_append(split_tup.first, a);
              }
            }),
        tuple_cartesian_product_two<Intermediate>(split_tup.second, tup1));
  }
}

template <bool Intermediate, class... Ts>
constexpr auto
tuple_cartesian_product_impl(cute::tuple<Ts...> const &&tup) {
  return tup;
}

template <bool Intermediate, class... Ts, class... Us, class... Tuples>
constexpr auto
tuple_cartesian_product_impl(
    cute::tuple<Ts...> const &tup0, cute::tuple<Us...> const &tup1, Tuples &&...other_tups) {
  return tuple_cartesian_product_impl</*Intermediate=*/true>(
      tuple_cartesian_product_two<Intermediate>(tup0, tup1), cute::forward<Tuples>(other_tups)...);
}
}  // namespace detail
// Cartesian product of tuples:
// Example:
// 1. (A,B) x (C,D) => ((A,C), (A,D), (B,C), (B,D))
// 2. (A,B) x (C,D) x (E) => ((A,C,E), (A,D,E), (B,C,E), (B,D,E))
template <class... Tuples>
constexpr auto
tuple_cartesian_product(Tuples &&...tups) {
  return detail::tuple_cartesian_product_impl</*Intermediate=*/false>(
      cute::forward<Tuples>(tups)...);
}

template <std::size_t... Is>
constexpr auto
make_index_seq_tuple(std::index_sequence<Is...>) {
  return cute::make_tuple(cute::Int<Is>{}...);
}

namespace detail {
template <int Idx, class Pred, class FoundFunc, class NotFoundFunc, class... Ts>
constexpr auto
tuple_return_if_impl(
    cute::tuple<Ts...> const &tup, Pred &&pred, FoundFunc &&found, NotFoundFunc &&not_found) {
  if constexpr (Idx >= sizeof...(Ts)) {
    return not_found();
  } else {
    using NotFoundRet = decltype(not_found());
    using FoundRet = decltype(found(cute::get<Idx>(tup)));
    if constexpr (not cute::is_same_v<FoundRet, NotFoundRet>) {
      StaticPrintTypesIf<true, FoundRet, NotFoundRet>{};
    } else {
      return pred(cute::get<Idx>(tup)) ? found(cute::get<Idx>(tup))
                                       : tuple_return_if_impl<Idx + 1>(
                                             tup,
                                             cute::forward<Pred>(pred),
                                             cute::forward<FoundFunc>(found),
                                             cute::forward<NotFoundFunc>(not_found));
    }
  }
}
}  // namespace detail
// Return `found`(e) for the first e that pred(e) == true,
// if no such element is found, return `not_found`()
template <class Pred, class FoundFunc, class NotFoundFunc, class... Ts>
constexpr auto
tuple_return_if(
    cute::tuple<Ts...> const &tup, Pred &&pred, FoundFunc &&found, NotFoundFunc &&not_found) {
  return detail::tuple_return_if_impl<0>(
      tup,
      cute::forward<Pred>(pred),
      cute::forward<FoundFunc>(found),
      cute::forward<NotFoundFunc>(not_found));
}

// Split `index_sequence` into `NSplits` and get the `SplitIdx`-th slice
template <int SplitIdx, int NSplits, std::size_t... Idx>
constexpr auto
index_seq_split_slice(std::index_sequence<Idx...>) {
  static_assert(0 <= SplitIdx and SplitIdx < NSplits, "SplitIdx out of bounds");
  constexpr int len = sizeof...(Idx);
  constexpr int len_per_split = cute::ceil_div(len, NSplits);
  constexpr int offset = len_per_split * SplitIdx;
  constexpr int split_len = cute::min(len_per_split, cute::max(len - offset, 0));
  constexpr auto index_seq =
      detail::offset_index_sequence<offset>(std::make_index_sequence<split_len>{});
  return index_seq;
}

// Split `tup` into `NSplits` and get the `SplitIdx`-th slice
// Example:
// tuple_split_slice<0, 2>(tuple(0,1,2)) => tuple(0,1)
// tuple_split_slice<1, 2>(tuple(0,1,2)) => tuple(2)
template <int SplitIdx, int NSplits, class... Ts>
constexpr auto
tuple_split_slice(cute::tuple<Ts...> const &tup) {
  constexpr auto index_seq =
      index_seq_split_slice<SplitIdx, NSplits>(std::make_index_sequence<sizeof...(Ts)>{});
  return detail::tuple_slice(index_seq, tup);
}

namespace detail {
template <class... Ts, std::size_t... Is>
constexpr auto
tuple_enumerate_impl(cute::tuple<Ts...> const &tup, std::index_sequence<Is...>) {
  return cute::make_tuple(std::make_pair(cute::Int<Is>{}, cute::get<Is>(tup))...);
}
}  // namespace detail
// works like Pythons's enumerate.
// e.g. for tuple (A,B), returns ((_0,A),(_1,B))
template <class... Ts>
constexpr auto
tuple_enumerate(cute::tuple<Ts...> const &tup) {
  return detail::tuple_enumerate_impl(tup, std::make_index_sequence<sizeof...(Ts)>{});
}

namespace detail {
template <class... Ts, std::size_t... Is>
static constexpr auto
tuple_unpack_cat_impl(cute::tuple<Ts...> const &tup, std::index_sequence<Is...>) {
  return cute::tuple_cat(cute::get<Is>(tup)...);
}
}  // namespace detail
// For a tuple of tuples, unpack the outmost layer of tuple
// and concat the inner tuples.
// e.g. ((A,B),(C,D)) => (A,B,C,D)
template <class... Ts>
static constexpr auto
tuple_unpack_cat(cute::tuple<Ts...> const &tup) {
  return detail::tuple_unpack_cat_impl(tup, std::make_index_sequence<sizeof...(Ts)>{});
}

namespace detail {
template <std::size_t Idx, class Elem, class... Ts>
static constexpr bool
tuple_has_elem_i(cute::tuple<Ts...> const &tup, Elem const &e) {
  if constexpr (Idx >= sizeof...(Ts)) {
    return false;
  } else {
    auto const &elem_i = cute::get<Idx>(tup);
    if constexpr (cute::is_same_v<decay_and_strip_t<decltype(elem_i)>, Elem>) {
      if (elem_i == e) {
        return true;
      }
    }
    return tuple_has_elem_i<Idx + 1>(tup, e);
  }
  return false;
}
}  // namespace detail
// checks if an element is in a tuple, requires both type and value to be equal
template <class Elem, class... Ts>
static constexpr bool
tuple_has_elem(cute::tuple<Ts...> const &tup, Elem const &e) {
  return detail::tuple_has_elem_i<0>(tup, e);
}

/////////////////////////////////////////////////////
// Enum classes
/////////////////////////////////////////////////////
enum class DataTypeEnum : int8_t { Void, FP16, BF16, FP32, E4M3, E5M2 };
enum class ArchEnum : int { Sm80 = 80, Sm89 = 89, Sm90 = 90 };
enum class CommOpEnum : int8_t {
  CommNone,       // gemm only, wo/ communication
  AllGather,      // tp allgather + gemm, comm not fused into gemm kernel
  ReduceScatter,  // gemm + tp reduce-scatter
  Gather,         // MoE tp, gemm + gather
  GatherStore,    // MoE tp, gemm + gather + store rs intermediates
  GatherRS,       // MoE tp, gemm + gather + rs
  AGKernel,       // tp allgather + gemm, comm fused into gemm kernel
  All2AllStore,   // MoE ep, gemm + store all2all intermediates
  AGScatter,      // MoE tp, ag + scatter + gemm
};
enum class GemmLayoutEnum : int8_t { RRR, RCR, RCC };
enum class ImplEnum : int8_t { GemmV2, GemmV3, GemmGroupedV2, GemmGroupedV3 };

enum class GemmKindEnum : int8_t { GemmDefault, GemmStreamK };
enum class CommKindEnum : int8_t { IntraNode, AcrossNode, IntraNodePcie };

enum class GemmStreamkModeEnum : int8_t { SK, DP };
enum class GemmRasterOrderEnum : int8_t { Heuristic, AlongM, AlongN };

enum class GemmKernelScheduleEnum : int8_t { Cooperative, PingPong };

/////////////////////////////////////////////////////
// Aliases for constant types
/////////////////////////////////////////////////////
using _CommNone = cute::C<CommOpEnum::CommNone>;
using _AllGather = cute::C<CommOpEnum::AllGather>;
using _ReduceScatter = cute::C<CommOpEnum::ReduceScatter>;
using _Gather = cute::C<CommOpEnum::Gather>;
using _GatherStore = cute::C<CommOpEnum::GatherStore>;
using _GatherRS = cute::C<CommOpEnum::GatherRS>;
using _AGKernel = cute::C<CommOpEnum::AGKernel>;
using _All2AllStore = cute::C<CommOpEnum::All2AllStore>;
using _AGScatter = cute::C<CommOpEnum::AGScatter>;

using _IntraNode = cute::C<CommKindEnum::IntraNode>;
using _AcrossNode = cute::C<CommKindEnum::AcrossNode>;
using _IntraNodePcie = cute::C<CommKindEnum::IntraNodePcie>;

using _GemmDefault = cute::C<GemmKindEnum::GemmDefault>;
using _GemmStreamK = cute::C<GemmKindEnum::GemmStreamK>;

using _Void = cute::C<DataTypeEnum::Void>;
using _FP16 = cute::C<DataTypeEnum::FP16>;
using _BF16 = cute::C<DataTypeEnum::BF16>;
using _FP32 = cute::C<DataTypeEnum::FP32>;
using _E4M3 = cute::C<DataTypeEnum::E4M3>;
using _E5M2 = cute::C<DataTypeEnum::E5M2>;

using _RRR = cute::C<GemmLayoutEnum::RRR>;
using _RCR = cute::C<GemmLayoutEnum::RCR>;
using _RCC = cute::C<GemmLayoutEnum::RCC>;

using _Sm80 = cute::C<ArchEnum::Sm80>;
using _Sm89 = cute::C<ArchEnum::Sm89>;
using _Sm90 = cute::C<ArchEnum::Sm90>;

using _GemmV2 = cute::C<ImplEnum::GemmV2>;
using _GemmV3 = cute::C<ImplEnum::GemmV3>;
using _GemmGroupedV2 = cute::C<ImplEnum::GemmGroupedV2>;
using _GemmGroupedV3 = cute::C<ImplEnum::GemmGroupedV3>;

using _True = cute::C<true>;
using _False = cute::C<false>;

using _StreamkSK = cute::C<GemmStreamkModeEnum::SK>;
using _StreamkDP = cute::C<GemmStreamkModeEnum::DP>;
using _RasterHeuristic = cute::C<GemmRasterOrderEnum::Heuristic>;
using _RasterAlongM = cute::C<GemmRasterOrderEnum::AlongM>;
using _RasterAlongN = cute::C<GemmRasterOrderEnum::AlongN>;

using _Cooperative = cute::C<GemmKernelScheduleEnum::Cooperative>;
using _PingPong = cute::C<GemmKernelScheduleEnum::PingPong>;

struct Auto : cute::tuple<> {};
struct None : cute::tuple<> {};

/////////////////////////////////////////////////////
// TypeTraits
/////////////////////////////////////////////////////
template <class T>
inline constexpr bool is_auto_v = std::is_same_v<decay_and_strip_t<T>, Auto>;

template <class T>
inline constexpr bool is_none_v = std::is_same_v<decay_and_strip_t<T>, None>;

namespace detail {
// check if a type is a cute::C<Enum> type
template <class T, class U>
struct is_const_of_type : std::false_type {};
template <auto v, class U>
struct is_const_of_type<cute::C<v>, U> : std::is_same<decay_and_strip_t<decltype(v)>, U> {};

template <class T>
struct is_flux_enum_type {
  using U = decay_and_strip_t<T>;
  static constexpr bool value =
      cute::is_same_v<U, CommOpEnum> or cute::is_same_v<U, CommKindEnum> or
      cute::is_same_v<U, GemmKindEnum> or cute::is_same_v<U, GemmLayoutEnum> or
      cute::is_same_v<U, DataTypeEnum> or cute::is_same_v<U, ArchEnum> or
      cute::is_same_v<U, ImplEnum> or cute::is_same_v<U, GemmStreamkModeEnum> or
      cute::is_same_v<U, GemmRasterOrderEnum> or cute::is_same_v<U, GemmKernelScheduleEnum>;
  using value_type = bool;
};

template <std::size_t... Is, class... Ts>
std::tuple<Ts...>
to_std_tuple_impl(std::index_sequence<Is...>, cute::tuple<Ts...> const &tup) {
  return std::make_tuple(cute::get<Is>(tup)...);
}
}  // namespace detail

template <class T, class U>
inline constexpr bool is_const_of_type_v =
    detail::is_const_of_type<decay_and_strip_t<T>, U>::value;

template <class T, class U>
inline constexpr bool is_of_type_v =
    is_const_of_type_v<T, U> or std::is_same_v<decay_and_strip_t<T>, U>;

template <class T>
inline constexpr bool is_flux_enum_v = detail::is_flux_enum_type<T>::value;

template <class Impl, __CUTE_REQUIRES(is_of_type_v<Impl, ImplEnum>)>
constexpr bool
is_grouped_gemm_impl(Impl const &impl) {
  return impl == _GemmGroupedV3{} or impl == _GemmGroupedV2{};
}

template <class DType, __CUTE_REQUIRES(is_of_type_v<DType, DataTypeEnum>)>
constexpr bool
is_fp8_dtype(DType const &dt) {
  return dt == _E4M3{} or dt == _E5M2{};
}

template <class DType, __CUTE_REQUIRES(is_of_type_v<DType, DataTypeEnum>)>
constexpr int
sizeof_dtype(DType const &dt) {
  // This function can be used in both compile time and runtime by
  // marking the return type with constexpr while use plain if-else
  // instead of if-constexpr in function body
  if (dt == _BF16{} or dt == _FP16{}) {
    return 2;
  } else if (dt == _FP32{}) {
    return 4;
  } else if (is_fp8_dtype(dt)) {
    return 1;
  } else if (dt == _Void{}) {
    return 0;
  } else {
    FLUX_CHECK(false) << "unsupported dtype:" << dt;
    return -1;
  }
}

// calculate void* ptr with offset of bytes
CUTLASS_HOST_DEVICE void *
ptr_offset(void *ptr, ptrdiff_t byte_offset) {
  return static_cast<char *>(ptr) + byte_offset;
}

CUTLASS_HOST_DEVICE void const *
ptr_offset(void const *ptr, ptrdiff_t byte_offset) {
  return static_cast<char const *>(ptr) + byte_offset;
}

// pad `sz` to the minimum multiple of `pad`
CUTLASS_HOST_DEVICE
int64_t
pad_to(int64_t sz, int64_t pad) {
  return (sz + pad - 1) / pad * pad;
}

template <class... Ts>
std::tuple<Ts...>
to_std_tuple(cute::tuple<Ts...> const &tup) {
  return detail::to_std_tuple_impl(std::make_index_sequence<sizeof...(Ts)>{}, tup);
}

//  Usage:
//    - used in type-selecting functions which returns an object of T but doesn't actually
//      create an instance of T.
//    - previously we return by *(T *)(nullptr), but it may cause compiler warnings. using this
//      function can avoid warnings and also force the type-selecting function only be valid
//      inside decltype since this function is incomplete.
//    - std::declval<T>() cannot be used because it has a static_assert check to forbid usage
//      inside functions.
//
//  Example:
//  template <bool Condition>
//  constexpr auto select_type() {
//    if constexpr(Condition) {
//      return make_declval<int>();
//    } else {
//      return make_declval<float>();
//    }
//  }
//  using T = decltype(select_type());
template <class T>
T make_declval();

// Wraps a type in value so that it can be used in functions,tuples...
//
// usually make_declval<T>() used with with decltype is enough to pass
// types as values, but make_declval<void>() cannot be used as function
// arguments (so it also cannot be used in make_tuple). in such case, we
// can use TypeWrapper<void>{} instead
template <class T>
struct TypeWrapper {
  using type = T;

  constexpr auto
  get() const {
    return make_declval<T>();
  }
};

///////////////////////////////////////////////////////////////
// Print
///////////////////////////////////////////////////////////////
inline char const *
enum_to_string(CommOpEnum comm_op) {
  switch (comm_op) {
    case CommOpEnum::CommNone: return "CommNone";
    case CommOpEnum::AllGather: return "AllGather";
    case CommOpEnum::AGKernel: return "AGKernel";
    case CommOpEnum::ReduceScatter: return "ReduceScatter";
    case CommOpEnum::Gather: return "Gather";
    case CommOpEnum::GatherStore: return "GatherStore";
    case CommOpEnum::GatherRS: return "GatherRS";
    case CommOpEnum::All2AllStore: return "All2AllStore";
    case CommOpEnum::AGScatter: return "AGScatter";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(CommKindEnum comm_t) {
  switch (comm_t) {
    case CommKindEnum::IntraNode: return "IntraNode";
    case CommKindEnum::AcrossNode: return "AcrossNode";
    case CommKindEnum::IntraNodePcie: return "IntraNodePcie";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(GemmKindEnum gemm_type) {
  switch (gemm_type) {
    case GemmKindEnum::GemmDefault: return "GemmDefault";
    case GemmKindEnum::GemmStreamK: return "GemmStreamK";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(DataTypeEnum dtype) {
  switch (dtype) {
    case DataTypeEnum::Void: return "Void";
    case DataTypeEnum::FP16: return "FP16";
    case DataTypeEnum::BF16: return "BF16";
    case DataTypeEnum::FP32: return "FP32";
    case DataTypeEnum::E4M3: return "E4M3";
    case DataTypeEnum::E5M2: return "E5M2";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(GemmLayoutEnum gemm_layout) {
  switch (gemm_layout) {
    case GemmLayoutEnum::RCR: return "RCR";
    case GemmLayoutEnum::RRR: return "RRR";
    case GemmLayoutEnum::RCC: return "RCC";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(ArchEnum arch) {
  switch (arch) {
    case ArchEnum::Sm80: return "Sm80";
    case ArchEnum::Sm89: return "Sm89";
    case ArchEnum::Sm90: return "Sm90";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(ImplEnum version) {
  switch (version) {
    case ImplEnum::GemmV2: return "GemmV2";
    case ImplEnum::GemmV3: return "GemmV3";
    case ImplEnum::GemmGroupedV2: return "GemmGroupedV2";
    case ImplEnum::GemmGroupedV3: return "GemmGroupedV3";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(GemmStreamkModeEnum streamk) {
  switch (streamk) {
    case GemmStreamkModeEnum::SK: return "StreamkSK";
    case GemmStreamkModeEnum::DP: return "StreamkDP";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(GemmRasterOrderEnum raster) {
  switch (raster) {
    case GemmRasterOrderEnum::Heuristic: return "RasterHeuristic";
    case GemmRasterOrderEnum::AlongM: return "RasterAlongM";
    case GemmRasterOrderEnum::AlongN: return "RasterAlongN";
    default: return "UNK";
  }
}

inline char const *
enum_to_string(GemmKernelScheduleEnum comm_op) {
  switch (comm_op) {
    case GemmKernelScheduleEnum::Cooperative: return "Cooperative";
    case GemmKernelScheduleEnum::PingPong: return "PingPong";
    default: return "UNK";
  }
}

template <class EnumT, __CUTE_REQUIRES(is_flux_enum_v<EnumT>)>
std::ostream &
operator<<(std::ostream &os, EnumT const &val) {
  os << enum_to_string(val);
  return os;
}

inline std::ostream &
operator<<(std::ostream &os, Auto const &v) {
  os << "Auto";
  return os;
}

inline std::ostream &
operator<<(std::ostream &os, None const &v) {
  os << "None";
  return os;
}

template <class... T>
std::enable_if_t<(sizeof...(T) > 0), std::ostream> &
operator<<(std::ostream &os, std::variant<T...> const &var) {
  ((std::holds_alternative<T>(var) ? void(os << std::get<T>(var)) : void(0)), ...);
  return os;
}

///////////////////////////////////////////////////////////////
// to_make_expr
//   translate an object to code that can create itself
///////////////////////////////////////////////////////////////
template <class EnumT, __CUTE_REQUIRES(is_flux_enum_v<EnumT>)>
std::string
to_make_expr(EnumT const &val) {
  std::ostringstream ss;
  ss << "_" << val << "{}()";
  return cute::move(ss).str();
}

template <auto v, __CUTE_REQUIRES(is_flux_enum_v<decltype(v)>)>
std::string
to_make_expr(cute::C<v> const &val) {
  std::ostringstream ss;
  ss << val << "{}";
  return cute::move(ss).str();
}

inline std::string
to_make_expr(bool val) {
  return val ? "true" : "false";
}

inline std::string
to_make_expr(int val) {
  return std::to_string(val);
}

inline std::string
to_make_expr(long val) {
  return std::to_string(val) + "l";
}

inline std::string
to_make_expr(Auto const &) {
  return "Auto{}";
}

inline std::string
to_make_expr(None const &) {
  return "None{}";
}

template <int v>
std::string
to_make_expr(cute::C<v> const &val) {
  std::ostringstream ss;
  ss << val;
  return cute::move(ss).str();
}

template <class... T>
std::enable_if_t<(sizeof...(T) > 0), std::string>
to_make_expr(std::variant<T...> const &var) {
  std::ostringstream ss;
  ((std::holds_alternative<T>(var) ? void(ss << to_make_expr(std::get<T>(var))) : void(0)), ...);
  return cute::move(ss).str();
}

namespace detail {
template <std::size_t... Is, class... Ts>
std::string
to_make_expression_tuple_impl(std::index_sequence<Is...>, cute::tuple<Ts...> const &tup) {
  std::ostringstream ss;
  ss << "cute::make_tuple(";
  ((ss << (Is == 0 ? "" : ",") << to_make_expr(cute::get<Is>(tup))), ...);
  ss << ")";
  return cute::move(ss).str();
}
}  // namespace detail

template <class... Ts>
std::string
to_make_expr(cute::tuple<Ts...> const &tup) {
  return detail::to_make_expression_tuple_impl(std::make_index_sequence<sizeof...(Ts)>{}, tup);
}

///////////////////////////////////////////////////////////////
// General comparator
///////////////////////////////////////////////////////////////
namespace detail {
template <class Tp, class Up, std::size_t I, std::size_t Size>
struct TupleCompare {
  static constexpr bool
  less(Tp const &t, Up const &u) {
    if constexpr (I == Size) {
      return false;
    } else {
      return bool(cute::get<I>(t) < cute::get<I>(u)) ||
             (!bool(cute::get<I>(u) < cute::get<I>(t)) &&
              TupleCompare<Tp, Up, I + 1, Size>::less(t, u));
    }
  }

  static constexpr bool
  eq(Tp const &t, Up const &u) {
    if constexpr (I == Size) {
      return true;
    } else {
      return bool(cute::get<I>(t) == cute::get<I>(u)) &&
             TupleCompare<Tp, Up, I + 1, Size>::eq(t, u);
    }
  }
};
}  // namespace detail

template <class... Ts>
constexpr bool
operator<(cute::tuple<Ts...> const &lhs, cute::tuple<Ts...> const &rhs) {
  using Compare = detail::TupleCompare<cute::tuple<Ts...>, cute::tuple<Ts...>, 0, sizeof...(Ts)>;
  return Compare::less(lhs, rhs);
}

template <class... Ts>
constexpr bool
operator==(cute::tuple<Ts...> const &lhs, cute::tuple<Ts...> const &rhs) {
  using Compare = detail::TupleCompare<cute::tuple<Ts...>, cute::tuple<Ts...>, 0, sizeof...(Ts)>;
  return Compare::eq(lhs, rhs);
}

///////////////////////////////////////////////////////////////
// FluxNamedTuple
///////////////////////////////////////////////////////////////
template <template <class...> class DerivedTpl, class... Ts>
struct FluxNamedTupleBase : cute::tuple<Ts...> {
  using Base = cute::tuple<Ts...>;
  using Derived = DerivedTpl<Ts...>;

 protected:
  static constexpr auto
  name() {
    return Derived::Name;
  }

  static constexpr auto
  lower_name() {
    return Derived::LowerName;
  }

  static constexpr int
  num_fields() {
    return Derived::Fields.size();
  }

  static constexpr auto
  fields() {
    return Derived::Fields;
  }

  static constexpr auto
  check_length() {
    static_assert(sizeof...(Ts) == num_fields(), "fields size mismatch");
  }

  template <class... Us>
  static constexpr DerivedTpl<Us...>
  from_tuple(cute::tuple<Us...> const &tup) {
    return {tup};
  }

  constexpr decltype(auto)
  as_tuple() const noexcept {
    return static_cast<Base const &>(*this);
  }

  constexpr decltype(auto)
  as_tuple() noexcept {
    return static_cast<Base &>(*this);
  }

  template <std::size_t... Is>
  void
  stream_operator_fields_impl(std::ostream &os, std::index_sequence<Is...>) const {
    static_assert(sizeof...(Is) == num_fields());
    ((os << (Is == 0 ? '(' : ',') << fields()[Is] << "=" << cute::get<Is>(as_tuple())), ...);
    os << ')';
  }

  template <std::size_t... Is>
  void
  to_make_expr_fields_impl(std::ostream &os, std::index_sequence<Is...>) const {
    static_assert(sizeof...(Is) == num_fields());
    ((os << (Is == 0 ? '(' : ',') << to_make_expr(cute::get<Is>(as_tuple()))), ...);
    os << ')';
  }

 public:
  constexpr FluxNamedTupleBase() {
    check_length();
    static_assert(cute::is_static_v<Base>, "default ctor requires static type");
  }

  constexpr FluxNamedTupleBase(cute::tuple<Ts...> const &base) : Base(base) { check_length(); }

  friend std::ostream &
  operator<<(std::ostream &os, FluxNamedTupleBase const &obj) {
    os << obj.name();
    obj.stream_operator_fields_impl(os, std::make_index_sequence<num_fields()>{});
    return os;
  }

  friend std::string
  to_make_expr(FluxNamedTupleBase const &obj) {
    std::stringstream ss;
    ss << "make_" << lower_name();
    obj.to_make_expr_fields_impl(ss, std::make_index_sequence<num_fields()>{});
    return std::move(ss).str();
  }
};

#define FLUX_NAMED_TUPLE_DEFINE_FIELD(FIELD, IDX)                                              \
  constexpr decltype(auto) FIELD() const noexcept { return cute::get<IDX>(this->as_tuple()); } \
  constexpr decltype(auto) FIELD() noexcept { return cute::get<IDX>(this->as_tuple()); }       \
  template <class NewType>                                                                     \
  constexpr auto FIELD(NewType const &val) const noexcept {                                    \
    return this->from_tuple(tuple_replace_item<IDX>(this->as_tuple(), val));                   \
  }

namespace detail {
template <class T>
struct is_flux_named_tuple : std::false_type {};

template <template <class...> class Tpl, class... Ts>
struct is_flux_named_tuple<Tpl<Ts...>>
    : std::is_base_of<FluxNamedTupleBase<Tpl, Ts...>, Tpl<Ts...>> {};
}  // namespace detail

template <class EnumT, __CUTE_REQUIRES(is_flux_enum_v<EnumT>)>
EnumT
unify_type(EnumT const &val) {
  return val;
}

template <auto v, __CUTE_REQUIRES(is_flux_enum_v<decltype(v)>)>
auto
unify_type(cute::C<v> const &val) -> decltype(v) {
  return val;
}

inline Auto
unify_type(Auto const &val) {
  return val;
}

inline None
unify_type(None const &val) {
  return val;
}

template <class... Ts>
auto
unify_type(std::variant<Ts...> const &val) {
  return val;
}

// for a given class template which derives from FluxNamedTuple,
// get the type of the return value of calling unify_type on it
template <
    template <class... Ts>
    class Tpl,
    __CUTE_REQUIRES(detail::is_flux_named_tuple<Tpl<>>::value)>
using unified_type_t = decltype(unify_type(make_declval<Tpl<>>()));

#define FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(CLS) \
  CLS() = default;                             \
  CLS(const CLS &) = default;                  \
  CLS(CLS &&) = default;                       \
  CLS &operator=(const CLS &) = default;       \
  CLS &operator=(CLS &&) = default;

}  // namespace flux
}  // namespace bytedance
