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

Classifier::Classifier(const std::string &modelDir,
                       const std::string &labelPath, const int cpuThreadNum,
                       const std::string &cpuPowerMode, int inputWidth,
                       int inputHeight, const std::vector<float> &inputMean,
                       const std::vector<float> &inputStd)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
      inputStd_(inputStd) {
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
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos);
    }
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

void Classifier::Preprocess(const cv::Mat &rgbaImage) {
  // Feed the input tensor with the data of the preprocessed image
  auto inputTensor = predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage, cv::Size(inputShape[3], inputShape[2]));
  cv::Mat resizedRGBImage;
  cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);
  resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0 / 255.0f);
  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data), inputData,
               inputMean_.data(), inputStd_.data(), inputShape[3],
               inputShape[2]);
}

bool topk_compare_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

void Classifier::Postprocess(std::vector<RESULT> *results) {
  auto outputTensor = predictor_->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  const int TOPK = 3;
  std::vector<std::pair<float, int>> vec;
  for (int i = 0; i < outputSize; i++) {
    vec.push_back(std::make_pair(outputData[i], i));
  }
  std::partial_sort(vec.begin(), vec.begin() + TOPK, vec.end(),
                    topk_compare_func);
  results->resize(TOPK);
  for (int i = 0; i < TOPK; i++) {
    (*results)[i].score = vec[i].first;
    (*results)[i].class_id = vec[i].second;
    (*results)[i].class_name = "Unknown";
    if ((*results)[i].class_id >= 0 &&
        (*results)[i].class_id < labelList_.size()) {
      (*results)[i].class_name = labelList_[(*results)[i].class_id];
    }
  }
}

void Classifier::Predict(const cv::Mat &rgbaImage, std::vector<RESULT> *results,
                         double *preprocessTime, double *predictTime,
                         double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Classifier postprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Classifier predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(results);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Classifier postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd) {
  classifier_.reset(new Classifier(modelDir, labelPath, cpuThreadNum,
                                   cpuPowerMode, inputWidth, inputHeight,
                                   inputMean, inputStd));
}

void Pipeline::VisualizeResults(const std::vector<RESULT> &results,
                                cv::Mat *rgbaImage) {
  int w = rgbaImage->cols;
  int h = rgbaImage->rows;
  int cx = w / 2;
  int offsetY = h / 2;
  for (int i = 0; i < results.size(); i++) {
    cv::Scalar font_color = cv::Scalar(128, 128, 128);
    int font_face = cv::FONT_HERSHEY_PLAIN;
    double font_scale = 2.f;
    float font_thickness = 2;
    if (i == 0) { // Top 1
      font_color = cv::Scalar(232, 155, 0);
    }
    std::string text = "Top" + std::to_string(i + 1) + "." +
                       results[i].class_name + ":" +
                       std::to_string(results[i].score);
    cv::Size size =
        cv::getTextSize(text, font_face, font_scale, font_thickness, nullptr);
    cv::putText(*rgbaImage, text, cv::Point2d(cx - size.width / 2, offsetY),
                cv::FONT_HERSHEY_PLAIN, font_scale, font_color, font_thickness);
    offsetY += size.height * 1.5f;
  }
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double preprocessTime, double predictTime,
                               double postprocessTime, cv::Mat *rgbaImage) {
  char text[255];
  cv::Scalar fontColor = cv::Scalar(255, 255, 255);
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1.f;
  float fontThickness = 1;
  sprintf(text, "Read GLFBO time: %.1f ms", readGLFBOTime);
  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
  textSize.height *= 1.25f;
  cv::Point2d offset(10, textSize.height + 15);
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Write GLTexture time: %.1f ms", writeGLTextureTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Preprocess time: %.1f ms", preprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Predict time: %.1f ms", predictTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Postprocess time: %.1f ms", postprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
}

bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  static double readGLFBOTime = 0, writeGLTextureTime = 0;
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;

  // Read pixels from FBO texture to CV image
  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // Feed the image, run inference and parse the results
  std::vector<RESULT> results;
  classifier_->Predict(rgbaImage, &results, &preprocessTime, &predictTime,
                       &postprocessTime);

  // Visualize the objects to the origin image
  VisualizeResults(results, &rgbaImage);

  // Visualize the status(performance data) to the origin image
  VisualizeStatus(readGLFBOTime, writeGLTextureTime, preprocessTime,
                  predictTime, postprocessTime, &rgbaImage);

  // Dump modified image if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  // Write back to texture2D
  WriteRGBAImageBackToGLTexture(rgbaImage, outTextureId, &writeGLTextureTime);
  return true;
}
