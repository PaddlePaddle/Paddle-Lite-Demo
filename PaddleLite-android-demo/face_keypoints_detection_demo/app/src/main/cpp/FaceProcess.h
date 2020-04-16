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

#pragma once

#include "Utils.h"
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

// face keypoints process
cv::Mat thin_face(cv::Mat image, std::vector<cv::Point2d> points);
cv::Mat enlarge_eyes(cv::Mat image, std::vector<cv::Point2d> points,
                     int radius = 15, int strength = 10);
cv::Mat rouge(cv::Mat image, std::vector<cv::Point2d> points, bool ruby = true);
cv::Mat whitening(cv::Mat image);
