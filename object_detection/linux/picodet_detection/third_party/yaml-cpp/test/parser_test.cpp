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

#include "yaml-cpp/parser.h"
#include "mock_event_handler.h"
#include "yaml-cpp/exceptions.h"
#include "gtest/gtest.h"
#include <yaml-cpp/depthguard.h>

using YAML::Parser;
using YAML::MockEventHandler;
using ::testing::NiceMock;
using ::testing::StrictMock;

TEST(ParserTest, Empty) {
  Parser parser;

  EXPECT_FALSE(parser);

  StrictMock<MockEventHandler> handler;
  EXPECT_FALSE(parser.HandleNextDocument(handler));
}

TEST(ParserTest, CVE_2017_5950) {
  std::string excessive_recursion;
  for (auto i = 0; i != 16384; ++i)
    excessive_recursion.push_back('[');
  std::istringstream input{excessive_recursion};
  Parser parser{input};

  NiceMock<MockEventHandler> handler;
  EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}

TEST(ParserTest, CVE_2018_20573) {
  std::string excessive_recursion;
  for (auto i = 0; i != 20535; ++i)
    excessive_recursion.push_back('{');
  std::istringstream input{excessive_recursion};
  Parser parser{input};

  NiceMock<MockEventHandler> handler;
  EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}

TEST(ParserTest, CVE_2018_20574) {
  std::string excessive_recursion;
  for (auto i = 0; i != 21989; ++i)
    excessive_recursion.push_back('{');
  std::istringstream input{excessive_recursion};
  Parser parser{input};

  NiceMock<MockEventHandler> handler;
  EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}

TEST(ParserTest, CVE_2019_6285) {
  std::string excessive_recursion;
  for (auto i = 0; i != 23100; ++i)
    excessive_recursion.push_back('[');
  excessive_recursion.push_back('f');
  std::istringstream input{excessive_recursion};
  Parser parser{input};

  NiceMock<MockEventHandler> handler;
  EXPECT_THROW(parser.HandleNextDocument(handler), YAML::DeepRecursion);
}
