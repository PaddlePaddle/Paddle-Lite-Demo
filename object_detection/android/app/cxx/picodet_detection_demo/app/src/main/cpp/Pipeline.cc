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
Detector::Detector(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
      inputStd_(inputStd), scoreThreshold_(scoreThreshold) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  labelList_ = LoadLabelList(labelPath);
  colorMap_ = GenerateColorMap(labelList_.size());
}

std::vector<std::string> Detector::LoadLabelList(const std::string &labelPath) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(labelPath);
  while (file) {
    std::string line;
    std::getline(file, line);
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

std::vector<cv::Scalar> Detector::GenerateColorMap(int numOfClasses) {
  std::vector<cv::Scalar> colorMap = std::vector<cv::Scalar>(numOfClasses);
  for (int i = 0; i < numOfClasses; i++) {
    int j = 0;
    int label = i;
    int R = 0, G = 0, B = 0;
    while (label) {
      R |= (((label >> 0) & 1) << (7 - j));
      G |= (((label >> 1) & 1) << (7 - j));
      B |= (((label >> 2) & 1) << (7 - j));
      j++;
      label >>= 3;
    }
    colorMap[i] = cv::Scalar(R, G, B);
  }
  return colorMap;
}

void Detector::Preprocess(const cv::Mat &rgbaImage) {
  // Feed the input tensor with the data of the preprocessed image
  auto input_tensor_scale = predictor_->GetInput(1);
  input_tensor_scale->Resize({1, 2});
  auto *scale_data = input_tensor_scale->mutable_data<float>();
  scale_data[0] =
      static_cast<float>(inputHeight_) / static_cast<float>(rgbaImage.rows);
  scale_data[1] =
      static_cast<float>(inputWidth_) / static_cast<float>(rgbaImage.cols);

  auto inputTensor = predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage,
             cv::Size(inputShape[3], inputShape[2]));
  cv::Mat resizedRGBImage;
  cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);
  resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0 / 255.0f);
  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data), inputData,
               inputMean_.data(), inputStd_.data(), inputShape[3],
               inputShape[2]);
}

void Detector::Postprocess(std::vector<Object> *results) {
  auto outputTensor = predictor_->GetOutput(0);
  auto output_bbox_tensor = predictor_->GetOutput(1);
  auto outputData = outputTensor->data<float>();
  auto *bbox_num = output_bbox_tensor->data<int>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);

  for (int i = 0; i < bbox_num[0]; i++) {
    // Class id
    auto class_id = static_cast<int>(round(outputData[i * 6]));
    // Confidence score
    auto score = outputData[1 + i * 6];
    int xmin = static_cast<int>(outputData[2 + i * 6]);
    int ymin = static_cast<int>(outputData[3 + i * 6]);
    int xmax = static_cast<int>(outputData[4 + i * 6]);
    int ymax = static_cast<int>(outputData[5 + i * 6]);
    int w = xmax - xmin;
    int h = ymax - ymin;
    if (score < scoreThreshold_)
      continue;

    Object object;
    object.class_name = class_id >= 0 && class_id < labelList_.size()
                            ? labelList_[class_id]
                            : "Unknow";
    object.fill_color = class_id >= 0 && class_id < colorMap_.size()
                            ? colorMap_[class_id]
                            : cv::Scalar(0, 0, 0);
    object.prob = score;
    object.x = xmin;
    object.y = ymin;
    object.w = w;
    object.h = h;
    results->push_back(object);
  }
}

void Detector::Predict(const cv::Mat &rgbaImage, std::vector<Object> *results,
                       double *preprocessTime, double *predictTime,
                       double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Detector predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(results);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold) {
  detector_.reset(new Detector(modelDir, labelPath, cpuThreadNum, cpuPowerMode,
                               inputWidth, inputHeight, inputMean, inputStd,
                               scoreThreshold));
}

void Pipeline::VisualizeResults(const std::vector<Object> &results,
                                cv::Mat *rgbaImage) {
  int oriw = rgbaImage->cols;
  int orih = rgbaImage->rows;
  for (int i = 0; i < results.size(); i++) {
    Object object = results[i];
    cv::Rect boundingBox = cv::Rect(object.x, object.y, object.w, object.h) &
                           cv::Rect(0, 0, oriw - 1, orih - 1);
    // Configure text size
    std::string text = object.class_name + ": ";
    std::string str_prob = std::to_string(object.prob);
    text += str_prob.substr(0, str_prob.find(".") + 4);
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1.5f;
    float fontThickness = 1.0f;
    cv::Size textSize =
        cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
    // Draw roi object, text, and background
    cv::rectangle(*rgbaImage, boundingBox, object.fill_color, 2);
    cv::rectangle(*rgbaImage,
                  cv::Point2d(boundingBox.x,
                              boundingBox.y - round(textSize.height * 1.25f)),
                  cv::Point2d(boundingBox.x + boundingBox.width, boundingBox.y),
                  object.fill_color, -1);
    cv::putText(*rgbaImage, text, cv::Point2d(boundingBox.x, boundingBox.y),
                fontFace, fontScale, cv::Scalar(255, 255, 255), fontThickness);
  }
}

void Pipeline::VisualizeStatus(double preprocessTime, double predictTime,
                               double postprocessTime, cv::Mat *rgbaImage) {
  char text[255];
  cv::Scalar fontColor = cv::Scalar(255, 255, 255);
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1.f;
  float fontThickness = 1;
  sprintf(text, "Preprocess time: %.1f ms", preprocessTime); // NOLINT
  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
  textSize.height *= 1.25f;
  cv::Point2d offset(10, textSize.height + 15);
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Predict time: %.1f ms", predictTime); // NOLINT
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Postprocess time: %.3f ms", postprocessTime); // NOLINT
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
}

bool Pipeline::Process(cv::Mat &rgbaImage, std::string savedImagePath) {
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;

  // Feed the image, run inference and parse the results
  std::vector<Object> results;
  detector_->Predict(rgbaImage, &results, &preprocessTime, &predictTime,
                     &postprocessTime);

  // Visualize the objects to the origin image
  VisualizeResults(results, &rgbaImage);

  // Visualize the status(performance data) to the origin image
  VisualizeStatus(preprocessTime, predictTime, postprocessTime, &rgbaImage);

  // Dump modified image if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  return true;
}
