// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "opencv2/core.hpp"      // NOLINT
#include "opencv2/imgcodecs.hpp" // NOLINT
#include "opencv2/imgproc.hpp"   // NOLINT
#include "paddle_api.h"          // NOLINT
#include "utils.h"               // NOLINT
#include <map>                   // NOLINT
#include <memory>                // NOLINT
#include <string>                // NOLINT
#include <vector>                // NOLINT

using namespace paddle::lite_api; // NOLINT

class DetPredictor {
public: // NOLINT
  explicit DetPredictor(const std::string &modelDir,
                        const int cpuThreadNum,           // NOLINT
                        const std::string &cpuPowerMode); // NOLINT

  std::vector<std::vector<std::vector<int>>>
  Predict(cv::Mat &rgbImage, std::map<std::string, double> Config, // NOLINT
          double *preprocessTime, double *predictTime,
          double *postprocessTime); // NOLINT

private:                                                       // NOLINT
  void Preprocess(const cv::Mat &img, const int max_side_len); // NOLINT
  std::vector<std::vector<std::vector<int>>>
  Postprocess(const cv::Mat srcimg,
              std::map<std::string, double> Config, // NOLINT
              int det_db_use_dilate);

private: // NOLINT
  std::vector<float> ratio_hw_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};
