// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "cls_process.h"
#include "rec_process.h"
#include "det_process.h"
#include "paddle_api.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
using namespace paddle::lite_api; // NOLINT

class Pipeline {
public:
  Pipeline(const std::string &detModelDir, const std::string &clsModelDir,
           const std::string &recModelDir, const std::string &cPUPowerMode,
           const int cPUThreadNum,
           const std::string &config_path, const std::string &dict_path);

  // bool Process(int inTextureId, int outTextureId, int textureWidth,
              //  int textureHeight, std::string savedImagePath);
  bool Process(std::string img_path, std::string output_img_path);

private:
  std::map<std::string, double> Config_;
  std::vector<std::string> charactor_dict_;
  std::shared_ptr<ClsPredictor> clsPredictor_;
  std::shared_ptr<DetPredictor> detPredictor_;
  std::shared_ptr<RecPredictor> recPredictor_;
};
