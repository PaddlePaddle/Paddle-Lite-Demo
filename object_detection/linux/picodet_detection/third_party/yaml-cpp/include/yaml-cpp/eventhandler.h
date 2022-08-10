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

#ifndef EVENTHANDLER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define EVENTHANDLER_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <string>

#include "yaml-cpp/anchor.h"
#include "yaml-cpp/emitterstyle.h"

namespace YAML {
struct Mark;

class EventHandler {
 public:
  virtual ~EventHandler() = default;

  virtual void OnDocumentStart(const Mark& mark) = 0;
  virtual void OnDocumentEnd() = 0;

  virtual void OnNull(const Mark& mark, anchor_t anchor) = 0;
  virtual void OnAlias(const Mark& mark, anchor_t anchor) = 0;
  virtual void OnScalar(const Mark& mark, const std::string& tag,
                        anchor_t anchor, const std::string& value) = 0;

  virtual void OnSequenceStart(const Mark& mark, const std::string& tag,
                               anchor_t anchor, EmitterStyle::value style) = 0;
  virtual void OnSequenceEnd() = 0;

  virtual void OnMapStart(const Mark& mark, const std::string& tag,
                          anchor_t anchor, EmitterStyle::value style) = 0;
  virtual void OnMapEnd() = 0;

  virtual void OnAnchor(const Mark& /*mark*/,
                        const std::string& /*anchor_name*/) {
    // empty default implementation for compatibility
  }
};
}  // namespace YAML

#endif  // EVENTHANDLER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
