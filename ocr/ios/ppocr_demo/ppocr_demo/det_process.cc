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

#include "det_process.h"     // NOLINT
#include "db_post_process.h" // NOLINT
#include "timer.h"           // NOLINT
#include <map>               // NOLINT
#include <memory>            // NOLINT
#include <string>            // NOLINT
#include <utility>           // NOLINT
#include <vector>            // NOLINT

// resize image to a size multiple of 32 which is required by the network
cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,
                     std::vector<float> &ratio_hw) { // NOLINT
  int w = img.cols;
  int h = img.rows;
  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
    } else {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
    }
  }

  int resize_h = static_cast<int>(float(h) * ratio); // NOLINT
  int resize_w = static_cast<int>(float(w) * ratio); // NOLINT
  if (resize_h % 32 == 0)
    resize_h = resize_h;
  else if (resize_h / 32 < 1 + 1e-5)
    resize_h = 32;
  else
    resize_h = (resize_h / 32 - 1) * 32;

  if (resize_w % 32 == 0)
    resize_w = resize_w;
  else if (resize_w / 32 < 1 + 1e-5)
    resize_w = 32;
  else
    resize_w = (resize_w / 32 - 1) * 32;
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

  ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
  ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
  return resize_img;
}

DetPredictor::DetPredictor(const std::string &modelDir, const int cpuThreadNum,
                           const std::string &cpuPowerMode) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir);
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
}

void DetPredictor::Preprocess(const cv::Mat &srcimg, const int max_side_len) {
  cv::Mat img = DetResizeImg(srcimg, max_side_len, ratio_hw_);
  cv::Mat img_fp;
  img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor_->GetInput(0)));
  input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
  auto *data0 = input_tensor0->mutable_data<float>();

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  const float *dimg = reinterpret_cast<const float *>(img_fp.data);
  NHWC3ToNC3HW(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);
}

std::vector<std::vector<std::vector<int>>>
DetPredictor::Postprocess(const cv::Mat srcimg,
                          std::map<std::string, double> Config,
                          int det_db_use_dilate) {
  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor_->GetOutput(0)));
  auto *outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();

  int s2 = int(shape_out[2]); // NOLINT
  int s3 = int(shape_out[3]); // NOLINT
  cv::Mat pred_map = cv::Mat::zeros(s2, s3, CV_32F);
  memcpy(pred_map.data, outptr, s2 * s3 * sizeof(float));
  cv::Mat cbuf_map;
  pred_map.convertTo(cbuf_map, CV_8UC1, 255.0f);

  const double threshold = double(Config["det_db_thresh"]) * 255; // NOLINT
  const double max_value = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
  if (det_db_use_dilate == 1) {
    cv::Mat dilation_map;
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);
    bit_map = dilation_map;
  }
  auto boxes = BoxesFromBitmap(pred_map, bit_map, Config);

  std::vector<std::vector<std::vector<int>>> filter_boxes =
      FilterTagDetRes(boxes, ratio_hw_[0], ratio_hw_[1], srcimg);

  return filter_boxes;
}

std::vector<std::vector<std::vector<int>>>
DetPredictor::Predict(cv::Mat &img, std::map<std::string, double> Config,
                      double *preprocessTime, double *predictTime,
                      double *postprocessTime) {
  //  Timer tic;
  //  tic.start();
  cv::Mat srcimg;
  img.copyTo(srcimg);

  // Read img
  int max_side_len = int(Config["max_side_len"]);           // NOLINT
  int det_db_use_dilate = int(Config["det_db_use_dilate"]); // NOLINT

  Preprocess(img, max_side_len);
  // tic.end();
  // *preprocessTime = tic.get_average_ms();
  // std::cout << "det predictor preprocess costs" <<  *preprocessTime;

  // tic.start();
  // Run predictor
  predictor_->Run();
  //  tic.end();
  //  *predictTime = tic.get_average_ms();
  // std::cout << "det predictor predict costs" <<  *predictTime;

  //  tic.start();
  auto filter_boxes = Postprocess(srcimg, Config, det_db_use_dilate);
  //  tic.end();
  //  *predictTime = tic.get_average_ms();
  // std::cout << "det predictor postprocess costs" <<  *postprocessTime;
  return filter_boxes;
}
