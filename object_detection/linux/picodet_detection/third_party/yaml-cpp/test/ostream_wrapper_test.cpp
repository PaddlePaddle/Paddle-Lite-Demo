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

#include <cstddef>
#include <sstream>

#include "yaml-cpp/ostream_wrapper.h"
#include "gtest/gtest.h"

namespace {
TEST(OstreamWrapperTest, BufferNoWrite) {
  YAML::ostream_wrapper wrapper;
  EXPECT_STREQ("", wrapper.str());
}

TEST(OstreamWrapperTest, BufferWriteStr) {
  YAML::ostream_wrapper wrapper;
  wrapper.write(std::string("Hello, world"));
  EXPECT_STREQ("Hello, world", wrapper.str());
}

TEST(OstreamWrapperTest, BufferWriteCStr) {
  YAML::ostream_wrapper wrapper;
  wrapper.write("Hello, world");
  EXPECT_STREQ("Hello, world", wrapper.str());
}

TEST(OstreamWrapperTest, StreamNoWrite) {
  std::stringstream stream;
  YAML::ostream_wrapper wrapper(stream);
  EXPECT_STREQ(NULL, wrapper.str());
  EXPECT_EQ("", stream.str());
}

TEST(OstreamWrapperTest, StreamWriteStr) {
  std::stringstream stream;
  YAML::ostream_wrapper wrapper(stream);
  wrapper.write(std::string("Hello, world"));
  EXPECT_STREQ(NULL, wrapper.str());
  EXPECT_EQ("Hello, world", stream.str());
}

TEST(OstreamWrapperTest, StreamWriteCStr) {
  std::stringstream stream;
  YAML::ostream_wrapper wrapper(stream);
  wrapper.write("Hello, world");
  EXPECT_STREQ(NULL, wrapper.str());
  EXPECT_EQ("Hello, world", stream.str());
}

TEST(OstreamWrapperTest, Position) {
  YAML::ostream_wrapper wrapper;
  wrapper.write("Hello, world\n");
  EXPECT_EQ(1, wrapper.row());
  EXPECT_EQ(0, wrapper.col());
  EXPECT_EQ(13, wrapper.pos());
}

TEST(OstreamWrapperTest, Comment) {
  YAML::ostream_wrapper wrapper;
  wrapper.write("Hello, world ");
  wrapper.set_comment();
  EXPECT_TRUE(wrapper.comment());
  wrapper.write("foo");
  EXPECT_TRUE(wrapper.comment());
  wrapper.write("\n");
  EXPECT_FALSE(wrapper.comment());
}
}
