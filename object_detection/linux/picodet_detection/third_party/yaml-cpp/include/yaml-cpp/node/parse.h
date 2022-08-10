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

#ifndef VALUE_PARSE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define VALUE_PARSE_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <iosfwd>
#include <string>
#include <vector>

#include "yaml-cpp/dll.h"

namespace YAML {
class Node;

/**
 * Loads the input string as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API Node Load(const std::string& input);

/**
 * Loads the input string as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API Node Load(const char* input);

/**
 * Loads the input stream as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API Node Load(std::istream& input);

/**
 * Loads the input file as a single YAML document.
 *
 * @throws {@link ParserException} if it is malformed.
 * @throws {@link BadFile} if the file cannot be loaded.
 */
YAML_CPP_API Node LoadFile(const std::string& filename);

/**
 * Loads the input string as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API std::vector<Node> LoadAll(const std::string& input);

/**
 * Loads the input string as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API std::vector<Node> LoadAll(const char* input);

/**
 * Loads the input stream as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 */
YAML_CPP_API std::vector<Node> LoadAll(std::istream& input);

/**
 * Loads the input file as a list of YAML documents.
 *
 * @throws {@link ParserException} if it is malformed.
 * @throws {@link BadFile} if the file cannot be loaded.
 */
YAML_CPP_API std::vector<Node> LoadAllFromFile(const std::string& filename);
}  // namespace YAML

#endif  // VALUE_PARSE_H_62B23520_7C8E_11DE_8A39_0800200C9A66
