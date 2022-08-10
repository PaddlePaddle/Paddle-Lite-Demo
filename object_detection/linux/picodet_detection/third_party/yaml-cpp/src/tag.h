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

#ifndef TAG_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define TAG_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <string>

namespace YAML {
struct Directives;
struct Token;

struct Tag {
  enum TYPE {
    VERBATIM,
    PRIMARY_HANDLE,
    SECONDARY_HANDLE,
    NAMED_HANDLE,
    NON_SPECIFIC
  };

  Tag(const Token& token);
  const std::string Translate(const Directives& directives);

  TYPE type;
  std::string handle, value;
};
}

#endif  // TAG_H_62B23520_7C8E_11DE_8A39_0800200C9A66
