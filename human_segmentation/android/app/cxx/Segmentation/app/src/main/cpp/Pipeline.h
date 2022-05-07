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

#include "Utils.h"
#include "paddle_api.h"                // NOLINT
#include <EGL/egl.h>                   // NOLINT
#include <GLES2/gl2.h>                 // NOLINT
#include <memory>                      // NOLINT
#include <opencv2/core.hpp>            // NOLINT
#include <opencv2/highgui/highgui.hpp> // NOLINT
#include <opencv2/imgcodecs.hpp>       // NOLINT
#include <opencv2/imgproc.hpp>         // NOLINT
#include <string>                      // NOLINT
#include <vector>                      // NOLINT

class Segmentation {
public: // NOLINT
  explicit Segmentation(const std::string &modelDir,
                        const std::string &labelPath, const int cpuThreadNum,
                        const std::string &cpuPowerMode,
                        const std::vector<int64_t> &inputShape,
                        const std::vector<float> &inputMean,
                        const std::vector<float> &inputStd);

  void Predict(const cv::Mat &rgbImage, std::vector<std::string> *results,
               double *preprocessTime, double *predictTime,
               double *postprocessTime);

  cv::Mat img_;

private: // NOLINT
  std::vector<std::string> LoadLabelList(const std::string &path);
  void Preprocess(const cv::Mat &img);
  cv::Mat Postprocess(const cv::Mat &img,
                      const std::vector<std::string> &labels);

private: // NOLINT
  std::vector<int64_t> inputShape_{1, 3, 513, 513};
  int warmup_{1};
  int repeats_{1};
  std::vector<float> inputMean_{0.f, 0.f, 0.f};
  std::vector<float> inputStd_{1.f, 1.f, 1.f};
  std::vector<std::string> labelList_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};

class Pipeline {
public: // NOLINT
  Pipeline(const std::string &modelDir, const std::string &labelPath,
           const int cpuThreadNum, const std::string &cpuPowerMode,
           const std::vector<int64_t> &inputShap,
           const std::vector<float> &inputMean,
           const std::vector<float> &inputStd);

  std::string Process(cv::Mat &rgbaImage); // NOLINT

  void GetOutImg(cv::Mat *output) { *output = image_; }

private: // NOLINT
  // Visualize the results to origin image
  void VisualizeResults(cv::Mat *rgbaImage);

  // Visualize the status(performance data) to origin image
  void VisualizeStatus(double preprocessTime, double predictTime,
                       double postprocessTime, cv::Mat *rgbaImage);

private: // NOLINT
  std::shared_ptr<Segmentation> seg_;
  cv::Mat image_;
};
