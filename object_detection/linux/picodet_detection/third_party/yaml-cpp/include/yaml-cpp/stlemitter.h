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

#ifndef STLEMITTER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
#define STLEMITTER_H_62B23520_7C8E_11DE_8A39_0800200C9A66

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

#include <list>
#include <map>
#include <set>
#include <vector>

namespace YAML {
template <typename Seq>
inline Emitter& EmitSeq(Emitter& emitter, const Seq& seq) {
  emitter << BeginSeq;
  for (const auto& v : seq)
    emitter << v;
  emitter << EndSeq;
  return emitter;
}

template <typename T>
inline Emitter& operator<<(Emitter& emitter, const std::vector<T>& v) {
  return EmitSeq(emitter, v);
}

template <typename T>
inline Emitter& operator<<(Emitter& emitter, const std::list<T>& v) {
  return EmitSeq(emitter, v);
}

template <typename T>
inline Emitter& operator<<(Emitter& emitter, const std::set<T>& v) {
  return EmitSeq(emitter, v);
}

template <typename K, typename V>
inline Emitter& operator<<(Emitter& emitter, const std::map<K, V>& m) {
  emitter << BeginMap;
  for (const auto& v : m)
    emitter << Key << v.first << Value << v.second;
  emitter << EndMap;
  return emitter;
}
}

#endif  // STLEMITTER_H_62B23520_7C8E_11DE_8A39_0800200C9A66
