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

#include "Pipeline.h"
#include <functional>
#include <utility>

Classifier::Classifier(const std::string &modelDir,
                       const std::string &labelPath, const int cpuThreadNum,
                       const std::string &cpuPowerMode,
                       const std::vector<int64_t> &inputShap,
                       const std::vector<float> &inputMean,
                       const std::vector<float> &inputStd, const int topk)
    : inputShape_(inputShap), inputMean_(inputMean), inputStd_(inputStd),
      topk_(topk) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  labelList_ = LoadLabelList(labelPath);
}

std::vector<std::string>
Classifier::LoadLabelList(const std::string &labelPath) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(labelPath);
  while (file) {
    std::string line;
    std::getline(file, line);
    if (line.length() > 0) {
      for (int i = 0; i < line.length(); i++) {
        if (line[i] == ' ') {
          std::string res_val = line.substr(i, line.length() - i - 1);
          labels.push_back(res_val);
          break;
        }
      }
    }
  }
  file.clear();
  file.close();
  return labels;
}

void Classifier::Preprocess(const cv::Mat &rgbaImage) {
  // Feed the input tensor with the data of the preprocessed image
  auto inputTensor = predictor_->GetInput(0);
  inputTensor->Resize(inputShape_);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage,
             cv::Size(inputShape_[3], inputShape_[2]));
  cv::Mat resizedRGBImage;
  cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);
  resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0 / 255.0f);
  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data), inputData,
               inputMean_.data(), inputStd_.data(), inputShape_[3],
               inputShape_[2]);
}
void Classifier::Postprocess(const int topk,
                             const std::vector<std::string> &labels,
                             std::vector<std::string> *results) {
  auto outputTensor = predictor_->GetOutput(0);
  auto scores = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int size = ShapeProduction(outputShape);
  std::vector<std::pair<float, int>> vec;
  vec.resize(size);
  for (int i = 0; i < size; i++) {
    vec[i] = std::make_pair(scores[i], i);
  }

  std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                    std::greater<std::pair<float, int>>());

  // print topk and score
  for (int i = 0; i < topk; i++) {
    float score = vec[i].first;
    int index = vec[i].second;
    std::ostringstream buff1;
    buff1 << "score: " << score;
    std::string top_i_str = "class: " + labels[index] + buff1.str();
    results->push_back(top_i_str);
    printf("i: %d,  index: %d,  name: %s,  score: %f \n", i, index,
           labels[index].c_str(), score);
  }
}

void Classifier::Predict(const cv::Mat &rgbaImage,
                         std::vector<std::string> *results,
                         double *preprocessTime, double *predictTime,
                         double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *preprocessTime);

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

  LOGD("Detector predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  std::string infer_time = std::to_string(averageTime);
  results->push_back(infer_time);
  Postprocess(topk_, labelList_, results);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   const std::vector<int64_t> &intputShape,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, const int topk) {
  classifier_.reset(new Classifier(modelDir, labelPath, cpuThreadNum,
                                   cpuPowerMode, intputShape, inputMean,
                                   inputStd, topk));
}

std::string Pipeline::Process(cv::Mat &rgbaImage) {
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;

  // Feed the image, run inference and parse the results
  std::vector<std::string> results;
  classifier_->Predict(rgbaImage, &results, &preprocessTime, &predictTime,
                       &postprocessTime);

  std::string res = "";
  for (int i = 0; i < results.size(); i++) {
    res = res + results[i] + "\n";
  }
  return res;
}
