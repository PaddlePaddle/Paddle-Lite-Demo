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

#include <cassert>
#include <stdexcept>

#include "directives.h"  // IWYU pragma: keep
#include "tag.h"
#include "token.h"

namespace YAML {
Tag::Tag(const Token& token)
    : type(static_cast<TYPE>(token.data)), handle{}, value{} {
  switch (type) {
    case VERBATIM:
      value = token.value;
      break;
    case PRIMARY_HANDLE:
      value = token.value;
      break;
    case SECONDARY_HANDLE:
      value = token.value;
      break;
    case NAMED_HANDLE:
      handle = token.value;
      value = token.params[0];
      break;
    case NON_SPECIFIC:
      break;
    default:
      assert(false);
  }
}

const std::string Tag::Translate(const Directives& directives) {
  switch (type) {
    case VERBATIM:
      return value;
    case PRIMARY_HANDLE:
      return directives.TranslateTagHandle("!") + value;
    case SECONDARY_HANDLE:
      return directives.TranslateTagHandle("!!") + value;
    case NAMED_HANDLE:
      return directives.TranslateTagHandle("!" + handle + "!") + value;
    case NON_SPECIFIC:
      // TODO:
      return "!";
    default:
      assert(false);
  }
  throw std::runtime_error("yaml-cpp: internal error, bad tag type");
}
}  // namespace YAML
