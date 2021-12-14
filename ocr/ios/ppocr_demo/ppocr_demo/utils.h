// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "paddle_api.h"

inline paddle::lite_api::PowerMode ParsePowerMode(std::string mode) {
  if (mode == "LITE_POWER_HIGH") {
    return paddle::lite_api::LITE_POWER_HIGH;
  } else if (mode == "LITE_POWER_LOW") {
    return paddle::lite_api::LITE_POWER_LOW;
  } else if (mode == "LITE_POWER_FULL") {
    return paddle::lite_api::LITE_POWER_FULL;
  } else if (mode == "LITE_POWER_RAND_HIGH") {
    return paddle::lite_api::LITE_POWER_RAND_HIGH;
  } else if (mode == "LITE_POWER_RAND_LOW") {
    return paddle::lite_api::LITE_POWER_RAND_LOW;
  }
  return paddle::lite_api::LITE_POWER_NO_BIND;
}

void NHWC3ToNC3HW(const float *src, float *dst, int size,
                  const std::vector<float> mean,
                  const std::vector<float> scale);

void NHWC1ToNC1HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height);
