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

#ifndef PARSER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define PARSER_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <ios>
#include <memory>

#include "yaml-cpp/dll.h"

namespace YAML {
class EventHandler;
class Node;
class Scanner;
struct Directives;
struct Token;

/**
 * A parser turns a stream of bytes into one stream of "events" per YAML
 * document in the input stream.
 */
class YAML_CPP_API Parser {
 public:
  /** Constructs an empty parser (with no input. */
  Parser();

  Parser(const Parser&) = delete;
  Parser(Parser&&) = delete;
  Parser& operator=(const Parser&) = delete;
  Parser& operator=(Parser&&) = delete;

  /**
   * Constructs a parser from the given input stream. The input stream must
   * live as long as the parser.
   */
  explicit Parser(std::istream& in);

  ~Parser();

  /** Evaluates to true if the parser has some valid input to be read. */
  explicit operator bool() const;

  /**
   * Resets the parser with the given input stream. Any existing state is
   * erased.
   */
  void Load(std::istream& in);

  /**
   * Handles the next document by calling events on the {@code eventHandler}.
   *
   * @throw a ParserException on error.
   * @return false if there are no more documents
   */
  bool HandleNextDocument(EventHandler& eventHandler);

  void PrintTokens(std::ostream& out);

 private:
  /**
   * Reads any directives that are next in the queue, setting the internal
   * {@code m_pDirectives} state.
   */
  void ParseDirectives();

  void HandleDirective(const Token& token);

  /**
   * Handles a "YAML" directive, which should be of the form 'major.minor' (like
   * a version number).
   */
  void HandleYamlDirective(const Token& token);

  /**
   * Handles a "TAG" directive, which should be of the form 'handle prefix',
   * where 'handle' is converted to 'prefix' in the file.
   */
  void HandleTagDirective(const Token& token);

 private:
  std::unique_ptr<Scanner> m_pScanner;
  std::unique_ptr<Directives> m_pDirectives;
};
}  // namespace YAML

#endif  // PARSER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
