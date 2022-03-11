// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "utils.h"
using namespace paddle::lite_api; // NOLINT

class ClsPredictor {
public:
  explicit ClsPredictor(const std::string &modelDir, const int cpuThreadNum,
                        const std::string &cpuPowerMode);

  cv::Mat Predict(const cv::Mat &rgbImage, double *preprocessTime,
                  double *predictTime, double *postprocessTime,
                  const float thresh);

private:
  void Preprocess(const cv::Mat &rgbaImage);
  cv::Mat Postprocess(const cv::Mat &img, const float thresh);

private:
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};
