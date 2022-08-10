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

#include <iostream>

#include "yaml-cpp/emitterstyle.h"
#include "yaml-cpp/eventhandler.h"
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

class NullEventHandler : public YAML::EventHandler {
 public:
  using Mark = YAML::Mark;
  using anchor_t = YAML::anchor_t;

  NullEventHandler() = default;

  void OnDocumentStart(const Mark&) override {}
  void OnDocumentEnd() override {}
  void OnNull(const Mark&, anchor_t) override {}
  void OnAlias(const Mark&, anchor_t) override {}
  void OnScalar(const Mark&, const std::string&, anchor_t,
                const std::string&) override {}
  void OnSequenceStart(const Mark&, const std::string&, anchor_t,
                       YAML::EmitterStyle::value style) override {}
  void OnSequenceEnd() override {}
  void OnMapStart(const Mark&, const std::string&, anchor_t,
                  YAML::EmitterStyle::value style) override {}
  void OnMapEnd() override {}
};

int main() {
  YAML::Node root;

  for (;;) {
    YAML::Node node;
    root = node;
  }
  return 0;
}
