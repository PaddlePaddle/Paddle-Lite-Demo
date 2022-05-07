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

#include <algorithm>  // NOLINT
#include <ctime>      // NOLINT
#include <memory>     // NOLINT
#include <numeric>    // NOLINT
#include <string>     // NOLINT
#include <sys/time.h> // NOLINT
#include <time.h>     // NOLINT
#include <utility>    // NOLINT
#include <vector>     // NOLINT

namespace PPShiTu {

struct RectResult {
  std::string class_name;
  int class_id;
  float score;
};

// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
  // RecModel result
  std::vector<RectResult> rec_result;
};

// Object for storing all preprocessed data
class ImageBlob {
public: // NOLINT
  // image width and height
  std::vector<float> im_shape_;
  // Buffer for image data after preprocessing
  std::vector<float> im_data_;
  // in net data shape(after pad)
  std::vector<float> in_net_shape_;
  // Evaluation image width and height
  // std::vector<float>  eval_im_size_f_;
  // Scale factor for image size to origin image size
  std::vector<float> scale_factor_;
};

void nms(std::vector<ObjectResult> *input_boxes, float nms_threshold,
         bool rec_nms = false);
void neon_mean_scale(const float *din, float *dout, int size, float *mean,
                     float *scale);
void activation_function_softmax(const float *src, float *dst, int length);

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

} // NOLINT
