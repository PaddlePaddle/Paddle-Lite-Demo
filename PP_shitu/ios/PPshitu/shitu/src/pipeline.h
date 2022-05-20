//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "object_detector.h" // NOLINT
#include "recognition.h"     // NOLINT
#include <algorithm>         // NOLINT
#include <memory>            // NOLINT
#include <string>            // NOLINT
#include <vector>            // NOLINT

class PipeLine {
public: // NOLINT
  explicit PipeLine(std::string det_model_path, std::string rec_model_path,
                    std::string label_path, std::vector<int> det_input_shape,
                    std::vector<int> rec_input_shape, int cpu_num_threads,
                    int warm_up, int repeats);

  std::vector<std::string>
  run(std::vector<cv::Mat> batch_imgs,
      std::vector<PPShiTu::ObjectResult> &det_result, // NOLINT
      cv::Mat *out_img, int batch_size);

  void print_time();

private: // NOLINT
  std::string det_model_path_;
  std::string rec_model_path_;
  std::string label_path_;
  std::vector<int> det_input_shape_;
  std::vector<int> rec_input_shape_;
  int cpu_num_threads_;
  int warm_up_;
  int repeats_;
  std::shared_ptr<PPShiTu::ObjectDetector> det_;
  std::shared_ptr<PPShiTu::Recognition> rec_;

  int max_det_num_{5};
  float rec_nms_thresold_{0.05f};
  std::vector<double> times_{0, 0, 0, 0, 0, 0, 0};

  void DetPredictImage(const std::vector<cv::Mat> batch_imgs,
                       std::vector<PPShiTu::ObjectResult> *im_result,
                       const int batch_size_det,
                       std::shared_ptr<PPShiTu::ObjectDetector> det,
                       const int max_det_num = 5);
};
