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

#include "exp.h"
#include "regex_yaml.h"
#include "regeximpl.h"
#include "stream.h"
#include "yaml-cpp/exceptions.h"  // IWYU pragma: keep
#include "yaml-cpp/mark.h"

namespace YAML {
const std::string ScanVerbatimTag(Stream& INPUT) {
  std::string tag;

  // eat the start character
  INPUT.get();

  while (INPUT) {
    if (INPUT.peek() == Keys::VerbatimTagEnd) {
      // eat the end character
      INPUT.get();
      return tag;
    }

    int n = Exp::URI().Match(INPUT);
    if (n <= 0)
      break;

    tag += INPUT.get(n);
  }

  throw ParserException(INPUT.mark(), ErrorMsg::END_OF_VERBATIM_TAG);
}

const std::string ScanTagHandle(Stream& INPUT, bool& canBeHandle) {
  std::string tag;
  canBeHandle = true;
  Mark firstNonWordChar;

  while (INPUT) {
    if (INPUT.peek() == Keys::Tag) {
      if (!canBeHandle)
        throw ParserException(firstNonWordChar, ErrorMsg::CHAR_IN_TAG_HANDLE);
      break;
    }

    int n = 0;
    if (canBeHandle) {
      n = Exp::Word().Match(INPUT);
      if (n <= 0) {
        canBeHandle = false;
        firstNonWordChar = INPUT.mark();
      }
    }

    if (!canBeHandle)
      n = Exp::Tag().Match(INPUT);

    if (n <= 0)
      break;

    tag += INPUT.get(n);
  }

  return tag;
}

const std::string ScanTagSuffix(Stream& INPUT) {
  std::string tag;

  while (INPUT) {
    int n = Exp::Tag().Match(INPUT);
    if (n <= 0)
      break;

    tag += INPUT.get(n);
  }

  if (tag.empty())
    throw ParserException(INPUT.mark(), ErrorMsg::TAG_WITH_NO_SUFFIX);

  return tag;
}
}  // namespace YAML
