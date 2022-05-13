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

#include "pipeline.h" // NOLINT
#include <algorithm>  // NOLINT
#include <fstream>    // NOLINT
#include <iostream>   // NOLINT
#include <math.h>     // NOLINT

void PrintResult(const std::string image_path,
                 std::vector<PPShiTu::ObjectResult> det_result) {
  printf("%s:\n", image_path.c_str());
  for (int i = 0; i < det_result.size(); ++i) {
    printf("\tresult%d: bbox[%d, %d, %d, %d], score: %f, label: %s\n", i,
           det_result[i].rect[0], det_result[i].rect[1], det_result[i].rect[2],
           det_result[i].rect[3], det_result[i].rec_result[0].score,
           det_result[i].rec_result[0].class_name.c_str());
  }
}

void VisualResult(cv::Mat *img, std::vector<PPShiTu::ObjectResult> results) {
  for (int i = 0; i < results.size(); i++) {
    int w = results[i].rect[2] - results[i].rect[0];
    int h = results[i].rect[3] - results[i].rect[1];
    cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
    cv::rectangle(*img, roi, cv::Scalar(i * 20, i * 50, 255), 4);
  }
  cv::cvtColor(*img, *img, cv::COLOR_BGR2RGB);
}

PipeLine::PipeLine(std::string det_model_path, std::string rec_model_path,
                   std::string label_path, std::vector<int> det_input_shape,
                   std::vector<int> rec_input_shape, int cpu_num_threads,
                   int warm_up, int repeats) {
  det_model_path_ = det_model_path;
  rec_model_path_ = rec_model_path;
  label_path_ = label_path;
  det_input_shape_ = det_input_shape;
  rec_input_shape_ = rec_input_shape;
  cpu_num_threads_ = cpu_num_threads;
  warm_up_ = warm_up;
  repeats_ = repeats;
  // create object detector
  det_ = std::make_shared<PPShiTu::ObjectDetector>(
      det_model_path_, det_input_shape_, cpu_num_threads_, warm_up_, repeats_);
  // create rec model
  rec_ = std::make_shared<PPShiTu::Recognition>(
      rec_model_path_, label_path_, rec_input_shape_, cpu_num_threads_,
      warm_up_, repeats_);
}

std::vector<std::string>
PipeLine::run(std::vector<cv::Mat> batch_imgs,
              std::vector<PPShiTu::ObjectResult> &det_result, // NOLINT
              cv::Mat *out_img, int batch_size) {
  std::fill(times_.begin(), times_.end(), 0);
  DetPredictImage(batch_imgs, &det_result, batch_size, det_, max_det_num_);
  // add the whole image for recognition to improve recall
  for (int i = 0; i < batch_imgs.size(); i++) {
    PPShiTu::ObjectResult result_whole_img = {
        {0, 0, batch_imgs[i].cols, batch_imgs[i].rows}, 0, 1.0};
    det_result.push_back(result_whole_img);
  }

  // get rec result
  for (int j = 0; j < det_result.size(); ++j) {
    std::vector<double> rec_time{};
    int w = det_result[j].rect[2] - det_result[j].rect[0];
    int h = det_result[j].rect[3] - det_result[j].rect[1];
    cv::Rect rect(det_result[j].rect[0], det_result[j].rect[1], w, h);
    cv::Mat crop_img = batch_imgs[0](rect);
    std::vector<PPShiTu::RectResult> result =
        rec_->RunRecModel(crop_img, rec_time);
    det_result[j].rec_result.assign(result.begin(), result.end());
    times_[3] += rec_time[0];
    times_[4] += rec_time[1];
    times_[5] += rec_time[2];
  }

  // rec nms
  auto nms_cost0 = PPShiTu::GetCurrentUS();
  PPShiTu::nms(&det_result, rec_nms_thresold_, true);
  auto nms_cost1 = PPShiTu::GetCurrentUS();
  times_[6] += (nms_cost1 - nms_cost0) / 1000.f;

  // results
  VisualResult(&batch_imgs[0], det_result);
  *out_img = batch_imgs[0];
  std::vector<std::string> res;
  res.emplace_back(std::to_string(times_[1] + times_[4]));
  for (int i = 0; i < det_result.size(); i++) {
    res.emplace_back(det_result[i].rec_result[0].class_name);
  }
  return res;
}

void PipeLine::print_time() {
  std::cout << "\n===================benchmark summary=================\n";
  std::cout << "ObjectDetect Preprocess:  " << times_[0] << "ms\n";
  std::cout << "ObjectDetect inference :  " << times_[1] << "ms\n";
  std::cout << "ObjectDetect Postprocess: " << times_[2] << "ms\n";
  std::cout << "Recongnise   Preprocess:  " << times_[3] << "ms\n";
  std::cout << "Recongnise   inference:   " << times_[4] << "ms\n";
  std::cout << "Recongnise   Postprocess: " << times_[5] << "ms\n";
  std::cout << "nms                     : " << times_[6] << "ms\n";
  std::cout << "=====================================================\n\n";
}

void PipeLine::DetPredictImage(const std::vector<cv::Mat> batch_imgs,
                               std::vector<PPShiTu::ObjectResult> *im_result,
                               const int batch_size_det,
                               std::shared_ptr<PPShiTu::ObjectDetector> det,
                               const int max_det_num) {
  int steps = ceil(batch_imgs.size() * 1.f / batch_size_det);
  for (int idx = 0; idx < steps; idx++) {
    int left_image_cnt = batch_imgs.size() - idx * batch_size_det;
    if (left_image_cnt > batch_size_det) {
      left_image_cnt = batch_size_det;
    }
    // Store all detected result
    std::vector<PPShiTu::ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;

    bool is_rbox = false;
    det->Predict(batch_imgs, &result, &bbox_num, &det_times);
    int item_start_idx = 0;
    for (int i = 0; i < left_image_cnt; i++) {
      cv::Mat im = batch_imgs[i];
      int detect_num = 0;
      for (int j = 0; j < min(bbox_num[i], max_det_num); j++) {
        PPShiTu::ObjectResult item = result[item_start_idx + j];
        if (item.class_id == -1) {
          continue;
        }
        detect_num += 1;
        im_result->push_back(item);
      }
      item_start_idx = item_start_idx + bbox_num[i];
    }
    times_[0] += det_times[0];
    times_[1] += det_times[1];
    times_[2] += det_times[2];
  }
}
