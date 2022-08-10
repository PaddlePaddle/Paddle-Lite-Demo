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

#ifndef MARK_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define MARK_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/dll.h"

namespace YAML {
struct YAML_CPP_API Mark {
  Mark() : pos(0), line(0), column(0) {}

  static const Mark null_mark() { return Mark(-1, -1, -1); }

  bool is_null() const { return pos == -1 && line == -1 && column == -1; }

  int pos;
  int line, column;

 private:
  Mark(int pos_, int line_, int column_)
      : pos(pos_), line(line_), column(column_) {}
};
}

#endif  // MARK_H_62B23520_7C8E_11DE_8A39_0800200C9A66
