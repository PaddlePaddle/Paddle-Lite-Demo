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

Pipeline::Pipeline() {
  // TODO(User) create and initialize an predictor
}

bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  static double readGLFBOTime = 0, writeGLTextureTime = 0;

  // Read pixels from FBO texture to CV image
  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // TODO(User) prepare the input tensors, run the predictor and fetch the
  // output tensors

  // Update the status on the camera preview image
  char text[255];
  cv::Scalar font_color = cv::Scalar(255, 255, 255);
  int font_face = cv::FONT_HERSHEY_PLAIN;
  double font_scale = 1.f;
  float font_thickness = 1;
  sprintf(text, "Read GLFBO time: %.1f ms", readGLFBOTime);
  cv::Size text_size =
      cv::getTextSize(text, font_face, font_scale, font_thickness, nullptr);
  text_size.height *= 1.25f;
  cv::Point2d offset(10, text_size.height + 15);
  cv::putText(rgbaImage, text, offset, font_face, font_scale, font_color,
              font_thickness);
  sprintf(text, "Write GLTexture time: %.1f ms", writeGLTextureTime);
  offset.y += text_size.height;
  cv::putText(rgbaImage, text, offset, font_face, font_scale, font_color,
              font_thickness);

  // Dump the modified camera preview image to file if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  // Write back to texture2D
  WriteRGBAImageBackToGLTexture(rgbaImage, outTextureId, &writeGLTextureTime);
  return true;
}
