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
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

struct RESULT {
  std::string class_name;
  int class_id;
  cv::Scalar fill_color;
  float score;
  float x;
  float y;
  float w;
  float h;
};

class Detector {
public:
  explicit Detector(const std::string &modelDir, const int accelerate_opencl,
                    const int cpuThreadNum, const std::string &cpuPowerMode,
                    int inputWidth, int inputHeight,
                    const std::vector<float> &inputMean,
                    const std::vector<float> &inputStd, float scoreThreshold);

  void Predict(const cv::Mat &rgbImage, std::vector<RESULT> *results,
               double *preprocessTime, double *predictTime,
               double *postprocessTime);

private:
  std::vector<std::string> LoadLabelList(const std::string &path);
  std::vector<cv::Scalar> GenerateColorMap(int numOfClasses);
  void Preprocess(const cv::Mat &rgbaImage);
  void Postprocess(std::vector<RESULT> *results);
  RESULT disPred2Bbox(const float *&dfl_det, int label,
                      float score, int x, int y, int stride,
                      std::vector<float> im_shape, int reg_max);
  void PicoDetPostProcess(std::vector<RESULT> *results,
                          std::vector<const float *> outs,
                          std::vector<int> fpn_stride,
                          std::vector<float> im_shape,
                          std::vector<float> scale_factor,
                          float score_threshold, float nms_threshold,
                          int num_class, int reg_max);
private:
  int inputWidth_;
  int inputHeight_;
  std::vector<float> inputMean_;
  std::vector<float> inputStd_;
  float scoreThreshold_;
  std::vector<std::string> labelList_;
  std::vector<cv::Scalar> colorMap_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
  //
  std::string arch_ = "PicoDet";
  std::vector<int> fpn_stride_ = {8, 16, 32, 64};
  float score_threshold = 0.7;
  float nms_threshold = 0.5;
};



void nms(std::vector<RESULT> &input_boxes, float nms_threshold);
