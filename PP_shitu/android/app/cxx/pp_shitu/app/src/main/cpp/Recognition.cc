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

#include "Recognition.h" // NOLINT
#include <utility>       // NOLINT

std::vector<RectResult>
Recognition::RunRecModel(const cv::Mat img, std::vector<double> &cost_time) {
  // Read img
  cv::Mat resize_image = ResizeImage(img);
  cv::Mat img_fp;
  resize_image.convertTo(img_fp, CV_32FC3, scale_);

  auto pre_cost0 = GetCurrentUS();
  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(
      std::move(this->predictor_->GetInput(0)));
  input_tensor->Resize({1, 3, img_fp.rows, img_fp.cols});
  auto *data0 = input_tensor->mutable_data<float>();
  const float *dimg = reinterpret_cast<const float *>(img_fp.data);
  neon_mean_scale(dimg, data0, img_fp.rows * img_fp.cols, mean_.data(),
                  std_.data());
  auto pre_cost1 = GetCurrentUS();

  // Run predictor
  for (int i = 0; i < warm_up_; i++)
    this->predictor_->Run();

  auto infer_cost0 = GetCurrentUS();
  for (int i = 0; i < repeats_; i++)
    this->predictor_->Run();
  auto infer_cost1 = GetCurrentUS();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(this->predictor_->GetOutput(1)));
  auto *output_data = output_tensor->data<float>();
  int output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }

  auto post_cost0 = GetCurrentUS();
  cv::Mat output_image;
  auto results = PostProcess(output_data, output_size, output_image);
  auto post_cost1 = GetCurrentUS();

  cost_time.push_back((pre_cost1 - pre_cost0) / 1000.f);
  cost_time.push_back((infer_cost1 - infer_cost0) / 1000.f / repeats_);
  cost_time.push_back((post_cost1 - post_cost0) / 1000.f);
  return results;
}

cv::Mat Recognition::ResizeImage(const cv::Mat img) {
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(this->size_, this->size_));
  return resize_img;
}

std::vector<RectResult> Recognition::PostProcess(const float *output_data,
                                                 int output_size,
                                                 cv::Mat &output_image) {
  std::vector<int> max_indices(this->topk_, 0);
  std::vector<double> max_scores(this->topk_, 0.f);
  for (int i = 0; i < this->topk_; i++) {
    max_indices[i] = 0;
    max_scores[i] = 0;
  }
  for (int i = 0; i < output_size; i++) {
    float score = output_data[i];
    int index = i;
    for (int j = 0; j < this->topk_; j++) {
      if (score > max_scores[j]) {
        index += max_indices[j];
        max_indices[j] = index - max_indices[j];
        index -= max_indices[j];
        score += max_scores[j];
        max_scores[j] = score - max_scores[j];
        score -= max_scores[j];
      }
    }
  }

  std::vector<RectResult> results(this->topk_);
  for (int i = 0; i < results.size(); i++) {
    results[i].class_name = "Unknown";
    if (max_indices[i] >= 0 && max_indices[i] < this->label_list_.size()) {
      results[i].class_name = this->label_list_[max_indices[i]];
    }
    results[i].score = max_scores[i];
    results[i].class_id = max_indices[i];
  }
  return results;
}
