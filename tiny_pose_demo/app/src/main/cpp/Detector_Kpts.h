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
#include "paddle_api.h"
#include "Detector.h"
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

struct RESULT_KEYPOINT {
  std::string class_name;
  int num_joints = 17;
  std::vector<float> keypoints;
};

class Detector_KeyPoint {
public:
  explicit Detector_KeyPoint(
      const std::string &modelDir, const int accelerate_opencl,
      const int cpuThreadNum, const std::string &cpuPowerMode, int inputWidth,
      int inputHeight, const std::vector<float> &inputMean,
      const std::vector<float> &inputStd, float scoreThreshold);

  void Predict(const cv::Mat &rgbImage, std::vector<RESULT> *results,
               std::vector<RESULT_KEYPOINT> *results_kpts,
               double *preprocessTime, double *predictTime,
               double *postprocessTime, bool single);

  void CropImg(const cv::Mat &img, std::vector<cv::Mat> &crop_img,
               std::vector<RESULT> &results,
               std::vector<std::vector<float>> &center_bs,
               std::vector<std::vector<float>> &scale_bs,
               float expandratio = 0.2);
  void FindMaxRect(std::vector<RESULT> *results, std::vector<RESULT> &rect_buff, bool single);
  float get_threshold() { return scoreThreshold_; };

private:
  std::vector<cv::Scalar> GenerateColorMap(int numOfClasses);
  void Preprocess(std::vector<cv::Mat> &bs_images);
  void Postprocess(std::vector<RESULT_KEYPOINT> *results,
                   std::vector<std::vector<float>> &center_bs,
                   std::vector<std::vector<float>> &scale_bs);

private:
  int inputWidth_;
  int inputHeight_;
  bool use_dark = true;
  std::vector<float> inputMean_;
  std::vector<float> inputStd_;
  float scoreThreshold_;
  std::vector<cv::Scalar> colorMap_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_keypoint_;
};
