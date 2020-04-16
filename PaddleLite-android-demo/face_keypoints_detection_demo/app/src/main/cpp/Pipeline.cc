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
  cv::resize(rgbaImage, resizedRGBAImage, cv::Size(), inputScale_, inputScale_,
             cv::INTER_CUBIC);
  cv::Mat resizedBGRImage;
  cv::cvtColor(resizedRGBAImage, resizedBGRImage, cv::COLOR_RGBA2BGR);
  resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1.0 / 255.0f);
  std::vector<int64_t> inputShape = {1, 3, resizedBGRImage.rows,
                                     resizedBGRImage.cols};
  // Prepare input tensor
  auto inputTensor = predictor_->GetInput(0);
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  NHWC2NCHW(reinterpret_cast<const float *>(resizedBGRImage.data), inputData,
            inputMean_.data(), inputStd_.data(), inputShape[3], inputShape[2]);
}

void FaceDetector::Postprocess(const cv::Mat &rgbaImage,
                               std::vector<Face> *faces) {
  int imageWidth = rgbaImage.cols;
  int imageHeight = rgbaImage.rows;
  // Get output tensor
  auto outputTensor = predictor_->GetOutput(2);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  faces->clear();
  for (int i = 0; i < outputSize; i += 6) {
    // Class id
    float class_id = outputData[i];
    // Confidence score
    float score = outputData[i + 1];
    int left = outputData[i + 2] * imageWidth;
    int top = outputData[i + 3] * imageHeight;
    int right = outputData[i + 4] * imageWidth;
    int bottom = outputData[i + 5] * imageHeight;
    int width = right - left;
    int height = bottom - top;
    if (score > scoreThreshold_) {
      Face face;
      face.roi = cv::Rect(left, top, width, height) &
                 cv::Rect(0, 0, imageWidth - 1, imageHeight - 1);
      faces->push_back(face);
    }
  }
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

FaceKeypointsDetector::FaceKeypointsDetector(const std::string &modelDir, const int cpuThreadNum,
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
}

void FaceKeypointsDetector::Preprocess(const cv::Mat &rgbaImage,
                                      const std::vector<Face> &faces) {
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
      float heightRatio = static_cast<float>(faces[i].roi.height) / inputShape[2];
      if (widthRatio > heightRatio) {
        h = w / inputAspectRatio;
      } else {
        w = h * inputAspectRatio;
      }
    }
    cv::Mat resizedRGBAImage(
                rgbaImage, cv::Rect(cx - w / 2, cy - h / 2, w, h) &
                           cv::Rect(0, 0, rgbaImage.cols - 1, rgbaImage.rows - 1));

    cv::resize(resizedRGBAImage, resizedRGBAImage,
               cv::Size(inputShape[3], inputShape[2]), 0.0f, 0.0f,
               cv::INTER_CUBIC);
    cv::Mat resizedGRAYImage;
    cv::cvtColor(resizedRGBAImage, resizedGRAYImage, cv::COLOR_RGBA2GRAY);
    resizedGRAYImage.convertTo(resizedGRAYImage, CV_32FC3, 1.0 / 255.0f);
    NHWC2NCHW_GRAY(reinterpret_cast<const float *>(resizedGRAYImage.data), inputData,
                   inputMean_.data(), inputStd_.data(), inputShape[3],
                   inputShape[2]);
    inputData += inputShape[1] * inputShape[2] * inputShape[3];
  }
}

void FaceKeypointsDetector::Postprocess(std::vector<Face> *faces,
                                       std::vector<cv::Point2d> *points) {
  auto outputTensor = predictor_->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  int batchSize = faces->size();
  int keypointsNum = outputSize / batchSize;
  assert(keypointsNum==136);
  for (int i = 0; i < batchSize; i++) {
    // get keypoints 68 x 2
    for (int j = 0; j < keypointsNum; j += 2) {
      points->push_back(cv::Point2d(outputData[j], outputData[j + 1]));
    }
    outputData += keypointsNum;
  }
}

void FaceKeypointsDetector::Predict(const cv::Mat &rgbImage, std::vector<Face> *faces,
                                   std::vector<cv::Point2d> *points, double *preprocessTime,
                                   double *predictTime, double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbImage, *faces);
  *preprocessTime = GetElapsedTime(t);
  LOGD("FaceKeypoint classifier postprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("FaceKeypoint classifier predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(faces, points);
  *postprocessTime = GetElapsedTime(t);
  LOGD("FaceKeypoint classifier postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &fdtModelDir, const int fdtCPUThreadNum,
                   const std::string &fdtCPUPowerMode, float fdtInputScale,
                   const std::vector<float> &fdtInputMean,
                   const std::vector<float> &fdtInputStd,
                   float detScoreThreshold, const std::string &fkpModelDir,
                   const int fkpCPUThreadNum,
                   const std::string &fkpCPUPowerMode, int fkpInputWidth,
                   int fkpInputHeight, const std::vector<float> &fkpInputMean,
                   const std::vector<float> &fkpInputStd) {
  faceDetector_.reset(new FaceDetector(
      fdtModelDir, fdtCPUThreadNum, fdtCPUPowerMode, fdtInputScale,
      fdtInputMean, fdtInputStd, detScoreThreshold));
  facKeypointsDetector_.reset(new FaceKeypointsDetector(
      fkpModelDir, fkpCPUThreadNum, fkpCPUPowerMode, fkpInputWidth,
      fkpInputHeight, fkpInputMean, fkpInputStd));
}

void Pipeline::VisualizeResults(const std::vector<Face> &faces,
                                         const std::vector<cv::Point2d> &points,
                                         cv::Mat *rgbaImage) {
  int keypointSize = 68;
  int rows = rgbaImage->rows;
  int cols = rgbaImage->cols;

  for (int i = 0; i < faces.size(); i++) {
    auto roi = faces[i].roi;
    // Configure color
    cv::Scalar color =  cv::Scalar(255, 0, 0);
    // Draw roi object
    cv::rectangle(*rgbaImage, faces[i].roi, color, 2);

    // Draw points
    int offset = i * keypointSize;
    std::vector<cv::Point2d> face_landmark;
    for (int k = 0; k < keypointSize; k++){
      cv::Point2d res = points[offset + k];
      cv::Point2d pp;
      pp.x = faces[i].roi.x + res.x * faces[i].roi.width;
      pp.y = faces[i].roi.y + res.y * faces[i].roi.height;

      cv::circle(*rgbaImage, pp, 1, cv::Scalar(0, 255, 0), 2); //在图像中画出特征点，1是圆的半径
      face_landmark.push_back(pp);
    }
    // 美白效果
//    cv::Mat tmp;
//    cv::Mat rgbImage;
//    cv::cvtColor(*rgbaImage, rgbImage, cv::COLOR_RGBA2RGB);
//    rgbImage = whitening(rgbImage);
//    cv::cvtColor(rgbImage, *rgbaImage, cv::COLOR_RGB2RGBA);
  }
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double fdtPreprocessTime, double fdtPredictTime,
                               double fdtPostprocessTime,
                               double fkpPreprocessTime, double fkpPredictTime,
                               double fkpPostprocessTime, cv::Mat *rgbaImage) {
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
  // FaceKeypointsDetector
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
}

bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  double readGLFBOTime = 0, writeGLTextureTime = 0;
  double fdtPreprocessTime = 0, fdtPredictTime = 0, fdtPostprocessTime = 0;
  double fkpPreprocessTime = 0, fkpPredictTime = 0, fkpPostprocessTime = 0;

  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // Stage1: Face detection
  std::vector<Face> faces;
  faceDetector_->Predict(rgbaImage, &faces, &fdtPreprocessTime, &fdtPredictTime,
                         &fdtPostprocessTime);
  if (faces.size() > 0) {
    // Stage2: FaceKeypoint detection
      std::vector<cv::Point2d> points;
      facKeypointsDetector_->Predict(rgbaImage, &faces, &points, &fkpPreprocessTime,
                             &fkpPredictTime, &fkpPostprocessTime);
    // Stage3: Visualize results
    auto t = GetCurrentTime();
    VisualizeResults(faces, points, &rgbaImage);
    double time = GetElapsedTime(t);
    LOGI("--visual time: %f ms", time);
  }

  // Visualize the status(performace data) to origin image
  VisualizeStatus(readGLFBOTime, writeGLTextureTime, fdtPreprocessTime,
                  fdtPredictTime, fdtPostprocessTime, fkpPreprocessTime,
                  fkpPredictTime, fkpPostprocessTime, &rgbaImage);

  // Dump modified image if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  WriteRGBAImageBackToGLTexture(rgbaImage, outTextureId, &writeGLTextureTime);
  return true;
}
