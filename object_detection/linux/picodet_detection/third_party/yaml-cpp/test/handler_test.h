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

#include "mock_event_handler.h"
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::StrictMock;

namespace YAML {
class HandlerTest : public ::testing::Test {
 protected:
  void Parse(const std::string& example) {
    std::stringstream stream(example);
    Parser parser(stream);
    while (parser.HandleNextDocument(handler)) {
    }
  }

  void IgnoreParse(const std::string& example) {
    std::stringstream stream(example);
    Parser parser(stream);
    while (parser.HandleNextDocument(nice_handler)) {
    }
  }

  InSequence sequence;
  StrictMock<MockEventHandler> handler;
  NiceMock<MockEventHandler> nice_handler;
};
}
