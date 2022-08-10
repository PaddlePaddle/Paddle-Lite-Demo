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

#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

#include "gtest/gtest.h"

#define EXPECT_THROW_EXCEPTION(exception_type, statement, message) \
  ASSERT_THROW(statement, exception_type);                         \
  try {                                                            \
    statement;                                                     \
  } catch (const exception_type &e) {                              \
    EXPECT_EQ(e.msg, message);                                     \
  }

namespace YAML {
namespace {

TEST(ErrorMessageTest, BadSubscriptErrorMessage) {
  const char *example_yaml =
      "first:\n"
      "   second: 1\n"
      "   third: 2\n";

  Node doc = Load(example_yaml);

  // Test that printable key is part of error message
  EXPECT_THROW_EXCEPTION(YAML::BadSubscript, doc["first"]["second"]["fourth"],
                         "operator[] call on a scalar (key: \"fourth\")");

  EXPECT_THROW_EXCEPTION(YAML::BadSubscript, doc["first"]["second"][37],
                         "operator[] call on a scalar (key: \"37\")");

  // Non-printable key is not included in error message
  EXPECT_THROW_EXCEPTION(YAML::BadSubscript,
                         doc["first"]["second"][std::vector<int>()],
                         "operator[] call on a scalar");

  EXPECT_THROW_EXCEPTION(YAML::BadSubscript, doc["first"]["second"][Node()],
                         "operator[] call on a scalar");
}

TEST(ErrorMessageTest, Ex9_1_InvalidNodeErrorMessage) {
  const char *example_yaml =
      "first:\n"
      "   second: 1\n"
      "   third: 2\n";

  const Node doc = Load(example_yaml);

  // Test that printable key is part of error message
  EXPECT_THROW_EXCEPTION(YAML::InvalidNode, doc["first"]["fourth"].as<int>(),
                         "invalid node; first invalid key: \"fourth\"");

  EXPECT_THROW_EXCEPTION(YAML::InvalidNode, doc["first"][37].as<int>(),
                         "invalid node; first invalid key: \"37\"");

  // Non-printable key is not included in error message
  EXPECT_THROW_EXCEPTION(YAML::InvalidNode,
                         doc["first"][std::vector<int>()].as<int>(),
                         "invalid node; this may result from using a map "
                         "iterator as a sequence iterator, or vice-versa");
}
}
}
