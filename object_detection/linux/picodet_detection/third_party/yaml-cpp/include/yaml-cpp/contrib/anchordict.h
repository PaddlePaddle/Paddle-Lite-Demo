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

#ifndef ANCHORDICT_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define ANCHORDICT_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <vector>

#include "../anchor.h"

namespace YAML {
/**
 * An object that stores and retrieves values correlating to {@link anchor_t}
 * values.
 *
 * <p>Efficient implementation that can make assumptions about how
 * {@code anchor_t} values are assigned by the {@link Parser} class.
 */
template <class T>
class AnchorDict {
 public:
  AnchorDict() : m_data{} {}
  void Register(anchor_t anchor, T value) {
    if (anchor > m_data.size()) {
      m_data.resize(anchor);
    }
    m_data[anchor - 1] = value;
  }

  T Get(anchor_t anchor) const { return m_data[anchor - 1]; }

 private:
  std::vector<T> m_data;
};
}  // namespace YAML

#endif  // ANCHORDICT_H_62B23520_7C8E_11DE_8A39_0800200C9A66
