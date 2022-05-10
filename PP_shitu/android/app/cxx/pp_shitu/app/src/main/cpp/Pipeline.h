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

#pragma once

#include "ObjectDetector.h" // NOLINT
#include "Recognition.h"    // NOLINT
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

class PipeLine {
public: // NOLINT
  explicit PipeLine(std::string det_model_path, std::string rec_model_path,
                    std::string label_path, std::vector<int> det_input_shape,
                    std::vector<int> rec_input_shape, int cpu_num_threads,
                    int warm_up, int repeats, int topk, std::string cpu_power);

  std::string run(std::vector<cv::Mat> &batch_imgs,      // NOLINT
                  std::vector<ObjectResult> &det_result, // NOLINT
                  int batch_size);

private: // NOLINT
  std::string det_model_path_;
  std::string rec_model_path_;
  std::string label_path_;
  std::vector<int> det_input_shape_;
  std::vector<int> rec_input_shape_;
  int cpu_num_threads_;
  int warm_up_;
  int repeats_;
  std::string cpu_pow_;
  std::shared_ptr<ObjectDetector> det_;
  std::shared_ptr<Recognition> rec_;
  int max_det_num_{3};
  float rec_nms_thresold_{0.05f};
  std::vector<double> times_{0, 0, 0, 0, 0, 0, 0};

  void DetPredictImage(const std::vector<cv::Mat> batch_imgs,
                       std::vector<ObjectResult> *im_result,
                       const int batch_size_det,
                       std::shared_ptr<ObjectDetector> det,
                       const int max_det_num = 3);
};
