//===- utils.hpp -------------------------------------------------- C++ ---===//
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

#ifndef FLUX_TEST_UTIL_H_
#define FLUX_TEST_UTIL_H_
#include <string>
#include <memory>
#include <cxxabi.h>
#define FLUX_CONCAT_IMPL(x, y) x##y
#define FLUX_MACRO_CONCAT(x, y) FLUX_CONCAT_IMPL(x, y)

#if !defined(WITH_GFLAGS)
#include <sstream>
#include <iostream>
#include "flux/flux.h"
#include <set>
namespace {
class FakeGflagHelper {
 public:
  static FakeGflagHelper &
  instance() {
    static FakeGflagHelper _instance;
    return _instance;
  }

  template <typename T>
  static void
  add_flag(const std::string &name, const T &value, const std::string &desc) {
    instance().add_flag_internal(name, value, desc);
  }

  static std::string
  help_string() {
    return instance().help_str_;
  }

 private:
  template <typename T>
  void
  add_flag_internal(const std::string &name, const T &value, const std::string &desc) {
    FLUX_CHECK(items_.find(name) == items_.end()) << " duplicated flag: " << name << "\n";
    items_.insert(name);
    std::stringstream ss;
    ss << name << "=" << value;
    if (desc.empty()) {
      ss << "\n";
    } else {
      ss << "\n\tdesc:" << desc << "\n";
    }
    help_str_ += ss.str();
  }
  FakeGflagHelper() = default;
  std::set<std::string> items_;
  std::string help_str_;
};

class HelpStringRegister {
 public:
  template <typename T>
  HelpStringRegister(const std::string &name, const T &value, const std::string &desc) {
    FakeGflagHelper::add_flag(name, value, desc);
  }
};
}  // namespace

#define DEFINE_int32(name, val, desc)                                    \
  static int32_t FLAGS_##name = val;                                     \
  HelpStringRegister FLUX_MACRO_CONCAT(__flags_register__, __COUNTER__)( \
      "FLAGS_" #name, FLAGS_##name, desc)
#define DEFINE_int64(name, val, desc)                                    \
  static int64_t FLAGS_##name = val;                                     \
  HelpStringRegister FLUX_MACRO_CONCAT(__flags_register__, __COUNTER__)( \
      "FLAGS_" #name, FLAGS_##name, desc)
#define DEFINE_bool(name, val, desc)                                     \
  static bool FLAGS_##name = val;                                        \
  HelpStringRegister FLUX_MACRO_CONCAT(__flags_register__, __COUNTER__)( \
      "FLAGS_" #name, FLAGS_##name, desc)
#define DEFINE_string(name, val, desc)                                   \
  static std::string FLAGS_##name = val;                                 \
  HelpStringRegister FLUX_MACRO_CONCAT(__flags_register__, __COUNTER__)( \
      "FLAGS_" #name, FLAGS_##name, desc)
#else
#include <gflags/gflags.h>
#endif

inline void
init_flags(int *argc, char ***argv, bool remove_flags) {
#if defined(WITH_GFLAGS)
  gflags::ParseCommandLineFlags(argc, argv, remove_flags);
#else
  std::cerr << "WARNING: gflags is not defined. using default args: \n"
            << FakeGflagHelper::help_string() << "\n";
#endif
}

template <typename T>
std::string
type_name() {
  using TR = typename std::remove_reference<T>::type;
  std::unique_ptr<char, void (*)(void *)> own(
#ifndef _MSC_VER
      abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
#else
      nullptr,
#endif
      std::free);
  std::string r = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const<TR>::value)
    r += " const";
  if (std::is_volatile<TR>::value)
    r += " volatile";
  if (std::is_lvalue_reference<T>::value)
    r += "&";
  else if (std::is_rvalue_reference<T>::value)
    r += "&&";
  return r;
}

// for string delimiter
std::vector<std::string>
split(const std::string &s, const std::string &delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

#endif
