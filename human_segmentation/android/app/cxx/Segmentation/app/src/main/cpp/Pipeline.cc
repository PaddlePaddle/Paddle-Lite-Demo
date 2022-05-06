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

#include "Pipeline.h"
#include <functional>
#include <utility>

Segmentation::Segmentation(const std::string &modelDir,
                           const std::string &labelPath, const int cpuThreadNum,
                           const std::string &cpuPowerMode,
                           const std::vector<int64_t> &inputShap,
                           const std::vector<float> &inputMean,
                           const std::vector<float> &inputStd)
    : inputShape_(inputShap), inputMean_(inputMean), inputStd_(inputStd) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  labelList_ = LoadLabelList(labelPath);
}

std::vector<std::string> Segmentation::LoadLabelList(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    LOGE("Load input label file error.");
    exit(1);
  }
  std::string line;
  while (getline(ifs, line)) {
    labels.emplace_back(line);
  }
  ifs.close();
  return labels;
}

void Segmentation::Preprocess(const cv::Mat &img) {
  // Prepare input data from image
  auto inputTensor = predictor_->GetInput(0);
  inputTensor->Resize(inputShape_);
  auto data = inputTensor->mutable_data<float>();
  // read img and pre-process
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(inputShape_[3], inputShape_[2]), 0.f,
             0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1.f);
  const float *dimg = reinterpret_cast<const float *>(imgf.data);
  NHWC3ToNC3HW(dimg, data, inputMean_.data(), inputStd_.data(), inputShape_[3],
               inputShape_[2]);
}

cv::Mat Segmentation::Postprocess(const cv::Mat &img,
                                  const std::vector<std::string> &labels) {
  auto output_tensor = predictor_->GetOutput(0);
  auto output_data = output_tensor->data<int64_t>();
  auto output_shape = output_tensor->shape();
  cv::Mat rgb_img;
  cv::resize(img, rgb_img, cv::Size(output_shape[2], output_shape[1]), 0.f,
             0.f);
  if ("background" == labels[0]) {
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        int tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[0] +
                      output_data[i * output_shape[2] + j] * 150;
        rgb_img.at<cv::Vec3b>(i, j)[0] = tmp_pix > 255 ? 255 : tmp_pix;
        tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[1] +
                  output_data[i * output_shape[2] + j] * 150;
        rgb_img.at<cv::Vec3b>(i, j)[1] = tmp_pix > 255 ? 255 : tmp_pix;
      }
    }
  } else {
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        int mask_pix = output_data[i * output_shape[2] + j] ^ 0;
        int tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[0] + mask_pix * 150;
        rgb_img.at<cv::Vec3b>(i, j)[0] = tmp_pix > 255 ? 255 : tmp_pix;
        tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[1] + mask_pix * 150;
        rgb_img.at<cv::Vec3b>(i, j)[1] = tmp_pix > 255 ? 255 : tmp_pix;
      }
    }
  }
  return rgb_img;
}

void Segmentation::Predict(const cv::Mat &rgbaImage,
                           std::vector<std::string> *results,
                           double *preprocessTime, double *predictTime,
                           double *postprocessTime) {
  auto t = GetCurrentTime();
  t = GetCurrentTime();
  cv::Mat rgbImage;
  cv::cvtColor(rgbaImage, rgbImage, cv::COLOR_RGBA2RGB);
  Preprocess(rgbImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Detector preprocess costs %f ms", *preprocessTime);

  double averageTime = 0.f;
  // warmup
  for (int i = 0; i < warmup_; i++) {
    predictor_->Run();
  }
  for (int i = 0; i < repeats_; i++) {
    t = GetCurrentTime();
    predictor_->Run();
    *predictTime = GetElapsedTime(t);
    averageTime += *predictTime;
  }
  averageTime = averageTime / repeats_;
  LOGD("warmup:%d, repeat:%d, Detector predict costs %f ms", warmup_, repeats_,
       *predictTime);

  t = GetCurrentTime();
  std::string infer_time = std::to_string(averageTime);
  results->push_back(infer_time);
  img_ = Postprocess(rgbImage, labelList_);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   const std::vector<int64_t> &intputShape,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd) {
  seg_.reset(new Segmentation(modelDir, labelPath, cpuThreadNum, cpuPowerMode,
                              intputShape, inputMean, inputStd));
}

std::string Pipeline::Process(cv::Mat &rgbaImage) {
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;

  // Feed the image, run inference and parse the results
  std::vector<std::string> results;
  seg_->Predict(rgbaImage, &results, &preprocessTime, &predictTime,
                &postprocessTime);
  image_ = seg_->img_;
  std::string res = "";
  for (int i = 0; i < results.size(); i++) {
    res = res + results[i] + "\n";
  }
  return res;
}
