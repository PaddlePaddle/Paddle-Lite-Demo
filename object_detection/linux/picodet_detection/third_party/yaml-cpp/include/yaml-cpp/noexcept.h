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

#ifndef NOEXCEPT_H_768872DA_476C_11EA_88B8_90B11C0C0FF8
#define NOEXCEPT_H_768872DA_476C_11EA_88B8_90B11C0C0FF8

#if defined(_MSC_VER) ||                                            \
    (defined(__GNUC__) && (__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || \
     (__GNUC__ >= 4))  // GCC supports "pragma once" correctly since 3.4
#pragma once
#endif

// This is here for compatibility with older versions of Visual Studio
// which don't support noexcept.
#if defined(_MSC_VER) && _MSC_VER < 1900
#define YAML_CPP_NOEXCEPT _NOEXCEPT
#else
#define YAML_CPP_NOEXCEPT noexcept
#endif

#endif
