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

#include "gtest/gtest.h"
#include <yaml-cpp/binary.h>

TEST(BinaryTest, DecodingSimple) {
  std::string input{90, 71, 86, 104, 90, 71, 74, 108, 90, 87, 89, 61};
  const std::vector<unsigned char> &result = YAML::DecodeBase64(input);
  EXPECT_EQ(std::string(result.begin(), result.end()), "deadbeef");
}

TEST(BinaryTest, DecodingNoCrashOnNegative) {
  std::string input{-58, -1, -99, 109};
  const std::vector<unsigned char> &result = YAML::DecodeBase64(input);
  EXPECT_TRUE(result.empty());
}
