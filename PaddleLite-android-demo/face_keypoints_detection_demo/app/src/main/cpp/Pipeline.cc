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

FaceDetector::FaceDetector(const std::string &modelDir, const int cpuThreadNum,
                           const std::string &cpuPowerMode, float inputScale,
                           const std::vector<float> &inputMean,
                           const std::vector<float> &inputStd,
                           float scoreThreshold)
    : inputScale_(inputScale), inputMean_(inputMean), inputStd_(inputStd),
      scoreThreshold_(scoreThreshold) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
}

void FaceDetector::Preprocess(const cv::Mat &rgbaImage) {
  auto t = GetCurrentTime();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage, cv::Size(640,480));

  cv::Mat resizedBGRImage;
  cv::cvtColor(resizedRGBAImage, resizedBGRImage, cv::COLOR_RGBA2BGR);
  resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1.0 / 255.0f);
  std::vector<int64_t> inputShape = {1, 3, resizedBGRImage.rows,
                                     resizedBGRImage.cols};
  // Prepare input tensor
  auto inputTensor = predictor_->GetInput(0);
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedBGRImage.data), inputData,
               inputMean_.data(), inputStd_.data(), inputShape[3],
               inputShape[2]);
}

void FaceDetector::HardNms(std::vector<Face> *input, std::vector<Face> *output, float iou_threshold) {
  std::sort(input->begin(), input->end(), [](const Face &a, const Face &b) { return a.score > b.score; });
  int box_num = input->size();
  std::vector<int> merged(box_num, 0);
  for (int i = 0; i < box_num; i++) {
    if (merged[i])
      continue;
    std::vector<Face> buf;
    buf.push_back(input->at(i));
    merged[i] = 1;

    float h0 = input->at(i).roi.height;
    float w0 = input->at(i).roi.width;

    float area0 = h0 * w0;

    for (int j = i + 1; j < box_num; j++) {
      if (merged[j])
        continue;

      float inner_x0 =
              input->at(i).roi.x > input->at(j).roi.x ? input->at(i).roi.x : input->at(j).roi.x;
      float inner_y0 =
              input->at(i).roi.y > input->at(j).roi.y ? input->at(i).roi.y : input->at(j).roi.y;

      float inputi_x1 = input->at(i).roi.x + input->at(i).roi.width;
      float inputi_y1 = input->at(i).roi.y + input->at(i).roi.height;
      float inputj_x1 = input->at(j).roi.x + input->at(j).roi.width;
      float inputj_y1 = input->at(j).roi.y + input->at(j).roi.height;
      float inner_x1 = inputi_x1 < inputj_x1 ? inputi_x1 : inputj_x1;
      float inner_y1 = inputi_y1 < inputj_y1 ? inputi_y1 : inputj_y1;

      float inner_h = inner_y1 - inner_y0 + 1;
      float inner_w = inner_x1 - inner_x0 + 1;
      if (inner_h <= 0 || inner_w <= 0)
        continue;
      float inner_area = inner_h * inner_w;

      float h1 = input->at(j).roi.height;
      float w1 = input->at(j).roi.width;
      float area1 = h1 * w1;

      float score;
      score = inner_area / (area0 + area1 - inner_area);
      if (score > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input->at(j));
      }
    }
    output->push_back(buf[0]);
  }
}

void FaceDetector::Postprocess(const cv::Mat &rgbaImage,
                               std::vector<Face> *faces) {
  int imageWidth = rgbaImage.cols;
  int imageHeight = rgbaImage.rows;
  // Get output tensor
  auto outputTensor = predictor_->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);

  auto outputTensor1 = predictor_->GetOutput(1);
  auto outputData1 = outputTensor1->data<float>();

  faces->clear();
  std::vector<Face> faces_tmp;
  for (int i = 0; i < outputSize; i += 2) {
    // Class id
    float class_id = outputData[i];
    // Confidence score
    float score = outputData[i + 1];
    int left = outputData1[2* i] * imageWidth;
    int top = outputData1[2*i + 1] * imageHeight;
    int right = outputData1[2*i + 2] * imageWidth;
    int bottom = outputData1[2*i + 3] * imageHeight;
    int width = right - left;
    int height = bottom - top;
    if (score > scoreThreshold_ && score < 1) {
      Face face;
      face.roi = cv::Rect(left, top, width, height) &
                 cv::Rect(0, 0, imageWidth - 1, imageHeight - 1);
      face.score = score;
      faces_tmp.push_back(face);
    }
  }
  HardNms(&faces_tmp, faces, 0.5);
}

void FaceDetector::Predict(const cv::Mat &rgbaImage, std::vector<Face> *faces,
                           double *preprocessTime, double *predictTime,
                           double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Face detector postprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Face detector predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(rgbaImage, faces);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Face detector postprocess costs %f ms", *postprocessTime);
}

FaceKeypointsDetector::FaceKeypointsDetector(const std::string &modelDir,
                                             const int cpuThreadNum,
                                             const std::string &cpuPowerMode,
                                             int inputWidth, int inputHeight)
    : inputWidth_(inputWidth), inputHeight_(inputHeight) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
}

void FaceKeypointsDetector::Preprocess(
    const cv::Mat &rgbaImage, const std::vector<Face> &faces,
    std::vector<cv::Rect> *adjustedFaceROIs) {
  // Prepare input tensor
  auto inputTensor = predictor_->GetInput(0);
  int batchSize = faces.size();
  std::vector<int64_t> inputShape = {batchSize, 1, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  for (int i = 0; i < batchSize; i++) {
    // Adjust the face region to improve the accuracy according to the aspect
    // ratio of input image of the target model
    int cx = faces[i].roi.x + faces[i].roi.width / 2.0f;
    int cy = faces[i].roi.y + faces[i].roi.height / 2.0f;
    int w = faces[i].roi.width;
    int h = faces[i].roi.height;
    float roiAspectRatio =
        static_cast<float>(faces[i].roi.width) / faces[i].roi.height;
    float inputAspectRatio = static_cast<float>(inputShape[3]) / inputShape[2];
    if (fabs(roiAspectRatio - inputAspectRatio) > 1e-5) {
      float widthRatio = static_cast<float>(faces[i].roi.width) / inputShape[3];
      float heightRatio =
          static_cast<float>(faces[i].roi.height) / inputShape[2];
      if (widthRatio > heightRatio) {
        h = w / inputAspectRatio;
      } else {
        w = h * inputAspectRatio;
      }
    }
    // Update the face region with adjusted roi
    (*adjustedFaceROIs)[i] =
        cv::Rect(cx - w / 2, cy - h / 2, w, h) &
        cv::Rect(0, 0, rgbaImage.cols - 1, rgbaImage.rows - 1);
    // Crop and obtain the face image
    cv::Mat resizedRGBAImage(rgbaImage, (*adjustedFaceROIs)[i]);
    cv::resize(resizedRGBAImage, resizedRGBAImage, cv::Size(inputShape[3], inputShape[2]));
    cv::Mat resizedGRAYImage;
    cv::cvtColor(resizedRGBAImage, resizedGRAYImage, cv::COLOR_RGBA2GRAY);
    resizedGRAYImage.convertTo(resizedGRAYImage, CV_32FC1);
    cv::Mat mean, std;
    cv::meanStdDev(resizedGRAYImage, mean, std);
    float inputMean = static_cast<float>(mean.at<double>(0, 0));
    float inputStd = static_cast<float>(std.at<double>(0, 0)) + 0.000001f;
    NHWC1ToNC1HW(reinterpret_cast<const float *>(resizedGRAYImage.data),
                 inputData, &inputMean, &inputStd, inputShape[3],
                 inputShape[2]);
    inputData += inputShape[1] * inputShape[2] * inputShape[3];
  }
}

void FaceKeypointsDetector::Postprocess(
    const std::vector<cv::Rect> &adjustedFaceROIs, std::vector<Face> *faces) {
  auto outputTensor = predictor_->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  int batchSize = faces->size();
  int keypointsNum = outputSize / batchSize;
  assert(batchSize == adjustedFaceROIs.size());
  assert(keypointsNum == 136); // 68 x 2
  for (int i = 0; i < batchSize; i++) {
    // Face keypoints with coordinates (x, y)
    for (int j = 0; j < keypointsNum; j += 2) {
      (*faces)[i].keypoints.push_back(cv::Point2d(
          adjustedFaceROIs[i].x + outputData[j] * adjustedFaceROIs[i].width,
          adjustedFaceROIs[i].y +
              outputData[j + 1] * adjustedFaceROIs[i].height));
    }
    outputData += keypointsNum;
  }
}

void FaceKeypointsDetector::Predict(const cv::Mat &rgbImage,
                                    std::vector<Face> *faces,
                                    double *preprocessTime, double *predictTime,
                                    double *postprocessTime) {
  auto t = GetCurrentTime();
  std::vector<cv::Rect> adjustedFaceROIs(faces->size());
  Preprocess(rgbImage, *faces, &adjustedFaceROIs);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Face keypoints detector postprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Face keypoints detector predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(adjustedFaceROIs, faces);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Face keypoints detector postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &fdtModelDir, const int fdtCPUThreadNum,
                   const std::string &fdtCPUPowerMode, float fdtInputScale,
                   const std::vector<float> &fdtInputMean,
                   const std::vector<float> &fdtInputStd,
                   float fdtScoreThreshold, const std::string &fkpModelDir,
                   const int fkpCPUThreadNum,
                   const std::string &fkpCPUPowerMode, int fkpInputWidth,
                   int fkpInputHeight) {
  faceDetector_.reset(new FaceDetector(
      fdtModelDir, fdtCPUThreadNum, fdtCPUPowerMode, fdtInputScale,
      fdtInputMean, fdtInputStd, fdtScoreThreshold));
  faceKeypointsDetector_.reset(
      new FaceKeypointsDetector(fkpModelDir, fkpCPUThreadNum, fkpCPUPowerMode,
                                fkpInputWidth, fkpInputHeight));
}

void Pipeline::VisualizeResults(const std::vector<Face> &faces,
                                cv::Mat *rgbaImage,
                                double *visualizeResultsTime) {
  auto t = GetCurrentTime();
  for (int i = 0; i < faces.size(); i++) {
    auto roi = faces[i].roi;
    // Configure color
    cv::Scalar color = cv::Scalar(255, 0, 0);
    // Draw roi object
    cv::rectangle(*rgbaImage, faces[i].roi, color, 2);
    // Draw face keypoints
    for (int j = 0; j < faces[i].keypoints.size(); j++) {
      cv::circle(*rgbaImage, faces[i].keypoints[j], 1, cv::Scalar(0, 255, 0),
                 2); //在图像中画出特征点，1是圆的半径
    }
    // 美白效果
    // cv::Mat tmp;
    // cv::Mat rgbImage;
    // cv::cvtColor(*rgbaImage, rgbImage, cv::COLOR_RGBA2RGB);
    // rgbImage = whitening(rgbImage);
    // cv::cvtColor(rgbImage, *rgbaImage, cv::COLOR_RGB2RGBA);
  }
  *visualizeResultsTime = GetElapsedTime(t);
  LOGD("VisualizeResults costs %f ms", *visualizeResultsTime);
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double fdtPreprocessTime, double fdtPredictTime,
                               double fdtPostprocessTime,
                               double fkpPreprocessTime, double fkpPredictTime,
                               double fkpPostprocessTime,
                               double visualizeResultsTime,
                               cv::Mat *rgbaImage) {
  char text[255];
  cv::Scalar color = cv::Scalar(255, 255, 255);
  int font_face = cv::FONT_HERSHEY_PLAIN;
  double font_scale = 1.f;
  float thickness = 1;
  sprintf(text, "Read GLFBO time: %.1f ms", readGLFBOTime);
  cv::Size text_size =
      cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
  text_size.height *= 1.25f;
  cv::Point2d offset(10, text_size.height + 15);
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  sprintf(text, "Write GLTexture time: %.1f ms", writeGLTextureTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  // Face detector
  sprintf(text, "FDT preprocess time: %.1f ms", fdtPreprocessTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  sprintf(text, "FDT predict time: %.1f ms", fdtPredictTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  sprintf(text, "FDT postprocess time: %.1f ms", fdtPostprocessTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  // Face keypoints detector
  sprintf(text, "FKP preprocess time: %.1f ms", fkpPreprocessTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  sprintf(text, "FKP predict time: %.1f ms", fkpPredictTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  sprintf(text, "FKP postprocess time: %.1f ms", fkpPostprocessTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  // Visualize results
  sprintf(text, "Visualize results time: %.1f ms", visualizeResultsTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
}

bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  double readGLFBOTime = 0, writeGLTextureTime = 0;
  double fdtPreprocessTime = 0, fdtPredictTime = 0, fdtPostprocessTime = 0;
  double fkpPreprocessTime = 0, fkpPredictTime = 0, fkpPostprocessTime = 0;
  double visualizeResultsTime = 0;

  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // Stage1: Face detection
  std::vector<Face> faces;
  faceDetector_->Predict(rgbaImage, &faces, &fdtPreprocessTime, &fdtPredictTime,
                         &fdtPostprocessTime);
  if (faces.size() > 0) {
    // Stage2: FaceKeypoint detection
    faceKeypointsDetector_->Predict(rgbaImage, &faces, &fkpPreprocessTime,
                                    &fkpPredictTime, &fkpPostprocessTime);
    // Stage3: Visualize results
    VisualizeResults(faces, &rgbaImage, &visualizeResultsTime);
  }

  // Visualize the status(performace data) to origin image
  VisualizeStatus(readGLFBOTime, writeGLTextureTime, fdtPreprocessTime,
                  fdtPredictTime, fdtPostprocessTime, fkpPreprocessTime,
                  fkpPredictTime, fkpPostprocessTime, visualizeResultsTime,
                  &rgbaImage);

  // Dump modified image if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  WriteRGBAImageBackToGLTexture(rgbaImage, outTextureId, &writeGLTextureTime);
  return true;
}
