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

Detector::Detector(const std::string &modelDir,
                   const std::string &labelPath, const int cpuThreadNum,
                   const std::string &cpuPowerMode,
                   const std::vector<int64_t> &inputShap,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd)
        : inputShape_(inputShap), inputMean_(inputMean), inputStd_(inputStd){
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
Detector::LoadLabelList(const std::string &labelPath) {
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

void Detector::Preprocess(const cv::Mat &rgbaImage) {
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

std::vector<float> Detector::Postprocess() {
  int topk = 100;
  float nmsParamNmsThreshold = 0.5;
  float confidenceThreshold = 0.5;
  std :: vector<std::pair<float,std::vector<float>>> vec;

  auto scoresTensor = predictor_->GetOutput(0);
  auto boxsTensor = predictor_->GetOutput(1);
  auto scores_Shape = scoresTensor->shape();
  int scoresSize = ShapeProduction(scores_Shape);

  auto *scores = scoresTensor->data<float>();
  auto *boxs = boxsTensor->data<float>();

  for (int i = 0, j = 0; i < scoresSize; i += 2, j+=4) {
    float rawLeft = boxs[j];
    float rawTop = boxs[j + 1];
    float rawRight = boxs[j + 2];
    float rawBottom = boxs[j + 3];
    float clampedLeft = fmax(fmin(rawLeft, 1.f), 0.f);
    float clampedTop = fmax(fmin(rawTop, 1.f), 0.f);
    float clampedRight = fmax(fmin(rawRight, 1.f), 0.f);
    float clampedBottom = fmax(fmin(rawBottom, 1.f), 0.f);
    std::vector<float> box;
    box.push_back(clampedLeft * imgWidth);
    box.push_back(clampedTop * imgHeight);
    box.push_back(clampedRight * imgWidth);
    box.push_back(clampedBottom * imgHeight);
    vec.push_back(std::make_pair(scores[i + 1], box));
  }

  std::sort(vec.begin(), vec.end(),
            std::greater<std::pair<float, std::vector<float>>>());

  std::vector<int> outputIndex;
  auto computeOverlapAreaRate = [](std::vector<float>anchor1, std::vector<float>anchor2) -> float {
      float xx1 = anchor1[0]>anchor2[0]?anchor1[0]:anchor2[0];
      float yy1 = anchor1[1]>anchor2[1]?anchor1[1]:anchor2[1];
      float xx2 = anchor1[2]<anchor2[2]?anchor1[2]:anchor2[2];
      float yy2 = anchor1[3]<anchor2[3]?anchor1[3]:anchor2[3];
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if(w<0||h<0){
        return 0;
      }
      float inter = w * h;
      float anchor1_area1 = (anchor1[2] - anchor1[0] + 1)*(anchor1[3] - anchor1[1] + 1);
      float anchor2_area1 = (anchor2[2] - anchor2[0] + 1)*(anchor2[3] - anchor2[1] + 1);
      return inter / (anchor1_area1 + anchor2_area1 - inter);
  };

  int count = 0;
  float INVALID_ANCHOR = -10000.0f;
  for(int i=0;i<vec.size();i++){
    if(fabs(vec[i].first-INVALID_ANCHOR) < 1e-5){
      continue;
    }
    if (++count >= topk) {
      break;
    }
    for(int j=i+1;j<vec.size();j++){
      if(fabs(vec[j].first-INVALID_ANCHOR) > 1e-5) {
        if (computeOverlapAreaRate(vec[i].second, vec[j].second) > nmsParamNmsThreshold) {
          vec[j].first = INVALID_ANCHOR;
        }
      }
    }
  }
  for(int i=0;i<vec.size() && count>0;i++){
    if(fabs(vec[i].first-INVALID_ANCHOR) > 1e-5){
      outputIndex.push_back(i);
      count--;
    }
  }
  std::vector<float> boxAndScores;
  if (outputIndex.size() > 0) {
    for (auto id:outputIndex) {
      if(vec[id].first < confidenceThreshold) continue;
      if(isnan(vec[id].first)){//skip the NaN score, maybe not correct
        continue;
      }
      for(int k=0;k<4;k++)
        boxAndScores.push_back((vec[id].second)[k]);//x1,y1,x2,y2
      boxAndScores.push_back((vec[id].first));  //possibility
    }
  }
  return boxAndScores;
}

std::vector<float> Detector::Predict(const cv::Mat &rgbaImage) {
  float preprocessTime,predictTime,postprocessTime;
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  preprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", preprocessTime);

  float averageTime = 0.f;
  // warmup
  for (int i = 0; i < warmup_; i++) {
    predictor_->Run();
  }
  for (int i = 0; i < repeats_; i++) {
    t = GetCurrentTime();
    predictor_->Run();
    predictTime = GetElapsedTime(t);
    averageTime += predictTime;
  }
  averageTime = averageTime / repeats_;

  LOGD("Detector predict costs %f ms", predictTime);

  t = GetCurrentTime();
  auto boxs_scores = Postprocess();
  postprocessTime = GetElapsedTime(t);
  boxs_scores.push_back(averageTime);
  LOGD("Detector postprocess costs %f ms", postprocessTime);
  return boxs_scores;
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   const std::vector<int64_t> &intputShape,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd) {
  detector_.reset(new Detector(modelDir, labelPath, cpuThreadNum,
                               cpuPowerMode, intputShape, inputMean,
                               inputStd));
}

std::vector<float> Pipeline::Process(cv::Mat &rgbaImage, int height, int width) {
  // Feed the image, run inference and parse the results
  detector_->set_imgHeight(height);
  detector_->set_imgWidth(width);
  return detector_->Predict(rgbaImage);
}
