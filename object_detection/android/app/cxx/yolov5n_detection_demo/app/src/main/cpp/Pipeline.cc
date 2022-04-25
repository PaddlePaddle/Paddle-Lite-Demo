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

  LOGD("--->model path: %s", modelDir.c_str());

  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  labelList_ = LoadLabelList(labelPath);
  colorMap_ = GenerateColorMap(labelList_.size());
  channelLength_ = inputWidth_ * inputHeight_;
  isInited_ = false;
}

void Detector::InitParams(const int& width, const int& height) {
    if (isInited_)
        return;

    float r_w = inputWidth_ / (width * 1.0);
    float r_h = inputHeight_ / (height * 1.0);
    if (r_h > r_w) {
        inputW = inputWidth_;
        inputH = r_w * height;
        inputX = 0;
        inputY = (inputHeight_ - inputH) / 2;
        ratio_ = r_w;
    } else {
        inputW = r_h * width;
        inputH = inputHeight_;
        inputX = (inputWidth_ - inputW) / 2;
        inputY = 0;
        ratio_ = r_h;
    }
    isInited_ = true;
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
  InitParams(rgbaImage.cols, rgbaImage.rows);

  // Feed the input tensor with the data of the preprocessed image
  auto inputTensor = predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();

  cv::Mat img;
  cv::cvtColor(rgbaImage, img, cv::COLOR_BGRA2RGB);
  cv::Mat re(inputH, inputW, CV_8UC3);
  cv::resize(img, re, cv::Size(inputW, inputH));
  cv::Mat out(inputHeight_, inputWidth_, CV_8UC3, cv::Scalar(128, 128, 128));
  re.copyTo(out(cv::Rect(inputX, inputY, re.cols, re.rows)));

  //split channels
  out.convertTo(out, CV_32FC3, 1. / 255.);
  cv::Mat input_channels[3];
  cv::split(out, input_channels);
  for (int j = 0; j < 3; j++) {
      memcpy(inputData + channelLength_ * j, input_channels[j].data,
             channelLength_ * sizeof(float));
  }
}

void Detector::ExtractBoxes(int seq_id, const float* in, std::map<int, std::vector<Object>>& outs,
    std::vector<int64_t>& shape) {
    int cls_num = shape[3] - 5;
    int xdim = int(inputWidth_ / strides_[seq_id]);
    for(int c=0; c < shape[1]; c++) {
        int step = c * shape[2] * shape[3];
        for (int r=0; r <  shape[2]; r++){
            int offset = step + r * shape[3];
            float score = in[offset + 4];
            if (score < confThresh_)
                continue;

            int max_cls_id = 0;
            float max_cls_val = 0;
            for (int i=0; i<cls_num; i++) {
                if (in[offset+5+i] > max_cls_val) {
                    max_cls_val = in[offset+5+i];
                    max_cls_id = i;
                }
            }

            score *= max_cls_val;
            if(score < confThresh_)
                continue;

            Object obj;
            int y = int(r / xdim);
            int x = int(r % xdim);
            int cx = static_cast<int>(((in[offset] * 2 - 0.5 + x) *
                    strides_[seq_id] - inputX) / ratio_);
            int cy = static_cast<int>(((in[offset+1] * 2 - 0.5 + y) *
                    strides_[seq_id] - inputY) / ratio_);
            int w = static_cast<int>(pow(in[offset+2] * 2, 2) *
                    anchors_[seq_id][2 * c] / ratio_);
            int h = static_cast<int>(pow(in[offset+3] * 2, 2) *
                    anchors_[seq_id][2 * c + 1] /ratio_);
            int left = cx - w / 2.0;
            int top = cy - h / 2.0;

            obj.rec = cv::Rect(left, top, w, h); //xywh
            obj.prob = score;
            obj.class_id = max_cls_id;

            if (outs.count(obj.class_id) == 0) outs.emplace(obj.class_id, std::vector<Object>());
            outs[obj.class_id].emplace_back(obj);
        }
    }
}

static float iou_calc(const cv::Rect& rec_a, const cv::Rect& rec_b) {
    cv::Rect u = rec_a | rec_b;
    cv::Rect s = rec_a & rec_b;
    float s_area = s.area();
    if (s_area < 20)
        return 0.f;
    return u.area() * 1.0 / s_area;
}

static bool cmp(const Object& a, const Object& b) {
    return a.prob > b.prob;
}

void Detector::Nms(std::map<int, std::vector<Object>>& src, std::vector<Object>* res) {
    for (auto it = src.begin(); it != src.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            item.class_name = item.class_id >= 0 && item.class_id < labelList_.size()
                                      ? labelList_[item.class_id]
                                      : "Unknow";
            item.fill_color = item.class_id >= 0 && item.class_id < colorMap_.size()
                                ? colorMap_[item.class_id]
                                : cv::Scalar(0, 0, 0);
            res->push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou_calc(item.rec, dets[n].rec) > nmsThresh_) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

void Detector::Postprocess(std::vector<Object> *results) {
  std::map<int, std::vector<Object>> raw_outputs;
  for (int k = 0; k < 3; k++) {
      std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(std::move(predictor_->GetOutput(k)));
      auto* outptr = output_tensor->data<float>();
      auto shape_out = output_tensor->shape();
      ExtractBoxes(k, outptr, raw_outputs, shape_out);
  }
  Nms(raw_outputs, results);
}

void Detector::Predict(const cv::Mat &rgbaImage, std::vector<Object> *results,
                       double *preprocessTime, double *predictTime,
                       double *postprocessTime) {
  auto t = GetCurrentTime();
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
    cv::Rect boundingBox = object.rec & cv::Rect(0, 0, oriw - 1, orih - 1);
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
