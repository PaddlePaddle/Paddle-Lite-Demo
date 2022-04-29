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
// #include "pose_action.h"

// Pipeline

Pipeline::Pipeline(const std::string &modelDir, int accelerate_opencl,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold) {
  detector_.reset(new Detector(modelDir, accelerate_opencl, cpuThreadNum, cpuPowerMode,
                               inputWidth, inputHeight, inputMean, inputStd,
                               scoreThreshold));
  detector_keypoint_.reset(
      new Detector_KeyPoint(modelDir, accelerate_opencl, cpuThreadNum, cpuPowerMode,
                            192, 256, inputMean, inputStd, 0.2));
}

void Pipeline::VisualizeResults(const std::vector<RESULT> &results,
                                cv::Mat *rgbaImage) {
  int w = rgbaImage->cols;
  int h = rgbaImage->rows;
  for (int i = 0; i < results.size(); i++) {
    RESULT object = results[i];
    cv::Rect boundingBox =
        cv::Rect(object.x * w, object.y * h, object.w * w, object.h * h) &
        cv::Rect(0, 0, w - 1, h - 1);
    // Configure text size
    std::string text = object.class_name + " ";
    text += std::to_string(static_cast<int>(object.score * 100)) + "%";
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

void Pipeline::VisualizeKptsResults(
    const std::vector<RESULT> &results,
    const std::vector<RESULT_KEYPOINT> &results_kpts, cv::Mat *rgbaImage, bool vis_rect) {
  if (vis_rect) {
    int w = rgbaImage->cols;
    int h = rgbaImage->rows;
    for (int i = 0; i < results.size(); i++) {
      RESULT object = results[i];
      cv::Rect boundingBox =
          cv::Rect(object.x * w, object.y * h, object.w * w, object.h * h) &
          cv::Rect(0, 0, w - 1, h - 1);
      // Configure text size
      std::string text = object.class_name + " ";
      text += std::to_string(static_cast<int>(object.score * 100)) + "%";
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

  const int edge[][2] = {{0, 1},   {0, 2},  {1, 3},   {2, 4},   {3, 5},
                         {4, 6},   {5, 7},  {6, 8},   {7, 9},   {8, 10},
                         {5, 11},  {6, 12}, {11, 13}, {12, 14}, {13, 15},
                         {14, 16}, {11, 12}};
  float kpts_threshold = detector_keypoint_->get_threshold();
  for (int batchid = 0; batchid < results_kpts.size(); batchid++) {
    for (int i = 0; i < results_kpts[batchid].num_joints; i++) {
      if (results_kpts[batchid].keypoints[i * 3] > kpts_threshold) {
        int x_coord = int(results_kpts[batchid].keypoints[i * 3 + 1]);
        int y_coord = int(results_kpts[batchid].keypoints[i * 3 + 2]);
        cv::circle(*rgbaImage, cv::Point2d(x_coord, y_coord), 2,
                   cv::Scalar(0, 255, 255), 2);
      }
    }
    for (int i = 0; i < results_kpts[batchid].num_joints; i++) {
      if (results_kpts[batchid].keypoints[edge[i][0] * 3] < kpts_threshold ||
          results_kpts[batchid].keypoints[edge[i][1] * 3] < kpts_threshold)
        continue;
      int x_start = int(results_kpts[batchid].keypoints[edge[i][0] * 3 + 1]);
      int y_start = int(results_kpts[batchid].keypoints[edge[i][0] * 3 + 2]);
      int x_end = int(results_kpts[batchid].keypoints[edge[i][1] * 3 + 1]);
      int y_end = int(results_kpts[batchid].keypoints[edge[i][1] * 3 + 2]);
      cv::line(*rgbaImage, cv::Point2d(x_start, y_start),
               cv::Point2d(x_end, y_end), cv::Scalar(0, 255, 255), 2);
    }
  }
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double preprocessTime, double predictTime,
                               double postprocessTime, cv::Mat *rgbaImage,
                               std::vector<RESULT_KEYPOINT> &results_kpts,
                               std::vector<RESULT> &results) {
  char text[255];
  cv::Scalar fontColor = cv::Scalar(255, 0, 0);
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 2.f;
  float fontThickness = 2;
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

// void Pipeline::Action_Process(cv::Mat *rgbaImage,
//                               std::vector<RESULT_KEYPOINT> &results_kpts,
//                               std::vector<RESULT> &results,
//                               int actionid,
//                               bool single_person,
//                               int imgw) {
//   if (single_person) {
//     int action_count = get_action_count(0);
//     //1: check_lateral_raise
//     //2: check_stand_press
//     //3: check_deep_down
//     if (!results.empty()) {
//       action_count = single_action_check(results_kpts[0].keypoints, results[0].h*rgbaImage->rows, actionid, 0);
//     }
//   }
//   else {
//     double_action_check(results_kpts, results, actionid, imgw);
//   }
// }

static std::vector<RESULT> results;
static int idx = 0;
bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  static double readGLFBOTime = 0, writeGLTextureTime = 0;
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;
  double preprocessTime_kpts = 0, predictTime_kpts = 0,
         postprocessTime_kpts = 0;

  // Read pixels from FBO texture to CV image
  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // Feed the image, run inference and parse the results
  if (idx % 1 == 0 or results.empty()) {
    idx = 0;
    results.clear();
    detector_->Predict(rgbaImage, &results, &preprocessTime, &predictTime,
                       &postprocessTime);
  }
  idx++;

  // add keypoint pipeline
  std::vector<RESULT_KEYPOINT> results_kpts;
  detector_keypoint_->Predict(rgbaImage, &results, &results_kpts,
                              &preprocessTime_kpts, &predictTime_kpts,
                              &postprocessTime_kpts, true);

  // Visualize the objects to the origin image
//  VisualizeResults(results, &rgbaImage);
  VisualizeKptsResults(results, results_kpts, &rgbaImage, false);

  // Visualize the status(performance data) to the origin image
//  VisualizeStatus(readGLFBOTime, writeGLTextureTime, preprocessTime+preprocessTime_kpts,
//                  predictTime+predictTime_kpts, postprocessTime+postprocessTime_kpts, &rgbaImage, results_kpts, results);
  // Action_Process(&rgbaImage, results_kpts, results, actionid, single, textureWidth);

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

// void Pipeline::ClearCount() {
//   clear_action_count();
// }

// std::vector<int> Pipeline::GetCount() {
//   return std::vector<int> {get_action_count(0), get_action_count(1)};
// }
