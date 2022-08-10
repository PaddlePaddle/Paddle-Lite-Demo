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

#include "handler_test.h"
#include "specexamples.h"   // IWYU pragma: keep
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;

#define EXPECT_THROW_PARSER_EXCEPTION(statement, message) \
  ASSERT_THROW(statement, ParserException);               \
  try {                                                   \
    statement;                                            \
  } catch (const ParserException& e) {                    \
    EXPECT_EQ(e.msg, message);                            \
  }

namespace YAML {
namespace {

TEST_F(HandlerTest, NoEndOfMapFlow) {
  EXPECT_THROW_PARSER_EXCEPTION(IgnoreParse("---{header: {id: 1"),
                                ErrorMsg::END_OF_MAP_FLOW);
}

TEST_F(HandlerTest, PlainScalarStartingWithQuestionMark) {
  EXPECT_CALL(handler, OnDocumentStart(_));
  EXPECT_CALL(handler, OnMapStart(_, "?", 0, EmitterStyle::Block));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "foo"));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "?bar"));
  EXPECT_CALL(handler, OnMapEnd());
  EXPECT_CALL(handler, OnDocumentEnd());
  Parse("foo: ?bar");
}

TEST_F(HandlerTest, NullStringScalar) {
  EXPECT_CALL(handler, OnDocumentStart(_));
  EXPECT_CALL(handler, OnMapStart(_, "?", 0, EmitterStyle::Block));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "foo"));
  EXPECT_CALL(handler, OnNull(_, 0));
  EXPECT_CALL(handler, OnMapEnd());
  EXPECT_CALL(handler, OnDocumentEnd());
  Parse("foo: null");
}

TEST_F(HandlerTest, CommentOnNewlineOfMapValueWithNoSpaces) {
  EXPECT_CALL(handler, OnDocumentStart(_));
  EXPECT_CALL(handler, OnMapStart(_, "?", 0, EmitterStyle::Block));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "key"));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "value"));
  EXPECT_CALL(handler, OnMapEnd());
  EXPECT_CALL(handler, OnDocumentEnd());
  Parse("key: value\n# comment");
}

TEST_F(HandlerTest, CommentOnNewlineOfMapValueWithOneSpace) {
  EXPECT_CALL(handler, OnDocumentStart(_));
  EXPECT_CALL(handler, OnMapStart(_, "?", 0, EmitterStyle::Block));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "key"));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "value"));
  EXPECT_CALL(handler, OnMapEnd());
  EXPECT_CALL(handler, OnDocumentEnd());
  Parse("key: value\n # comment");
}

TEST_F(HandlerTest, CommentOnNewlineOfMapValueWithManySpace) {
  EXPECT_CALL(handler, OnDocumentStart(_));
  EXPECT_CALL(handler, OnMapStart(_, "?", 0, EmitterStyle::Block));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "key"));
  EXPECT_CALL(handler, OnScalar(_, "?", 0, "value"));
  EXPECT_CALL(handler, OnMapEnd());
  EXPECT_CALL(handler, OnDocumentEnd());
  Parse("key: value\n    # comment");
}
}  // namespace
}  // namespace YAML
