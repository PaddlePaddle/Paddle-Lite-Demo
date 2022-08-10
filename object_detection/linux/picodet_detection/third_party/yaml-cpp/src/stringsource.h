// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STRINGSOURCE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define STRINGSOURCE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <cstddef>

namespace YAML {
class StringCharSource {
 public:
  StringCharSource(const char* str, std::size_t size)
      : m_str(str), m_size(size), m_offset(0) {}

  operator bool() const { return m_offset < m_size; }
  char operator[](std::size_t i) const { return m_str[m_offset + i]; }
  bool operator!() const { return !static_cast<bool>(*this); }

  const StringCharSource operator+(int i) const {
    StringCharSource source(*this);
    if (static_cast<int>(source.m_offset) + i >= 0)
      source.m_offset += i;
    else
      source.m_offset = 0;
    return source;
  }

  StringCharSource& operator++() {
    ++m_offset;
    return *this;
  }

  StringCharSource& operator+=(std::size_t offset) {
    m_offset += offset;
    return *this;
  }

 private:
  const char* m_str;
  std::size_t m_size;
  std::size_t m_offset;
};
}

#endif  // STRINGSOURCE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
