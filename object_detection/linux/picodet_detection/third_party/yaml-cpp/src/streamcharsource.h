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

#ifndef STREAMCHARSOURCE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define STREAMCHARSOURCE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "stream.h"
#include "yaml-cpp/noexcept.h"
#include <cstddef>

namespace YAML {

class StreamCharSource {
 public:
  StreamCharSource(const Stream& stream) : m_offset(0), m_stream(stream) {}
  StreamCharSource(const StreamCharSource& source) = default;
  StreamCharSource(StreamCharSource&&) YAML_CPP_NOEXCEPT = default;
  StreamCharSource& operator=(const StreamCharSource&) = delete;
  StreamCharSource& operator=(StreamCharSource&&) = delete;
  ~StreamCharSource() = default;

  operator bool() const;
  char operator[](std::size_t i) const { return m_stream.CharAt(m_offset + i); }
  bool operator!() const { return !static_cast<bool>(*this); }

  const StreamCharSource operator+(int i) const;

 private:
  std::size_t m_offset;
  const Stream& m_stream;
};

inline StreamCharSource::operator bool() const {
  return m_stream.ReadAheadTo(m_offset);
}

inline const StreamCharSource StreamCharSource::operator+(int i) const {
  StreamCharSource source(*this);
  if (static_cast<int>(source.m_offset) + i >= 0)
    source.m_offset += static_cast<std::size_t>(i);
  else
    source.m_offset = 0;
  return source;
}
}  // namespace YAML

#endif  // STREAMCHARSOURCE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
