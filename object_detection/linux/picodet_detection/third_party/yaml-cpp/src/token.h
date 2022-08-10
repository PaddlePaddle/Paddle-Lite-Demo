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

#ifndef TOKEN_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define TOKEN_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include "yaml-cpp/mark.h"
#include <iostream>
#include <string>
#include <vector>

namespace YAML {
const std::string TokenNames[] = {
    "DIRECTIVE",        "DOC_START",      "DOC_END",       "BLOCK_SEQ_START",
    "BLOCK_MAP_START",  "BLOCK_SEQ_END",  "BLOCK_MAP_END", "BLOCK_ENTRY",
    "FLOW_SEQ_START",   "FLOW_MAP_START", "FLOW_SEQ_END",  "FLOW_MAP_END",
    "FLOW_MAP_COMPACT", "FLOW_ENTRY",     "KEY",           "VALUE",
    "ANCHOR",           "ALIAS",          "TAG",           "SCALAR"};

struct Token {
  // enums
  enum STATUS { VALID, INVALID, UNVERIFIED };
  enum TYPE {
    DIRECTIVE,
    DOC_START,
    DOC_END,
    BLOCK_SEQ_START,
    BLOCK_MAP_START,
    BLOCK_SEQ_END,
    BLOCK_MAP_END,
    BLOCK_ENTRY,
    FLOW_SEQ_START,
    FLOW_MAP_START,
    FLOW_SEQ_END,
    FLOW_MAP_END,
    FLOW_MAP_COMPACT,
    FLOW_ENTRY,
    KEY,
    VALUE,
    ANCHOR,
    ALIAS,
    TAG,
    PLAIN_SCALAR,
    NON_PLAIN_SCALAR
  };

  // data
  Token(TYPE type_, const Mark& mark_)
      : status(VALID), type(type_), mark(mark_), value{}, params{}, data(0) {}

  friend std::ostream& operator<<(std::ostream& out, const Token& token) {
    out << TokenNames[token.type] << std::string(": ") << token.value;
    for (const std::string& param : token.params)
      out << std::string(" ") << param;
    return out;
  }

  STATUS status;
  TYPE type;
  Mark mark;
  std::string value;
  std::vector<std::string> params;
  int data;
};
}  // namespace YAML

#endif  // TOKEN_H_62B23520_7C8E_11DE_8A39_0800200C9A66
