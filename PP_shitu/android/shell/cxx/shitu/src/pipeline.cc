//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "pipeline.h" // NOLINT
#include <algorithm>  // NOLINT
#include <fstream>    // NOLINT
#include <iostream>   // NOLINT
#include <math.h>     // NOLINT

void PrintResult(const std::string image_path,
                 std::vector<PPShiTu::ObjectResult> det_result) {
  printf("%s:\n", image_path.c_str());
  for (int i = 0; i < det_result.size(); ++i) {
    printf("\tresult%d: bbox[%d, %d, %d, %d], score: %f, label: %s\n", i,
           det_result[i].rect[0], det_result[i].rect[1], det_result[i].rect[2],
           det_result[i].rect[3], det_result[i].rec_result[0].score,
           det_result[i].rec_result[0].class_name.c_str());
  }
}

void VisualResult(cv::Mat img, std::vector<PPShiTu::ObjectResult> results,
                  std::string out_path) {
  for (int i = 0; i < results.size(); i++) {
    int w = results[i].rect[2] - results[i].rect[0];
    int h = results[i].rect[3] - results[i].rect[1];
    cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
    cv::rectangle(img, roi, cv::Scalar(255, i * 50, i * 20), 2);
    std::ostringstream oss;
    oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    oss << "class_id: " << results[i].rec_result[0].class_id << " ";
    oss << results[i].rec_result[0].score;
    std::string text = oss.str();
    cv::putText(img, text,
                cv::Point(results[i].rect[0] + w / 3, results[i].rect[1] + 15),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                cv::Scalar(255, i * 50, i * 20), 1);
  }
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::imwrite(out_path, img);
}

int main(int argc, char **argv) {
  std::cout << "Usage: " << argv[0]
            << " [img_dir](just dir which contains image)\n"
            << "[det_model_path] [rec_model_path] [label_path]\n";
  if (argc < 5) {
    std::cout << "absent input params, please check your commandline"
              << std::endl;
    return -1;
  }
  std::string img_dir = argv[1];
  std::string det_model_path = argv[2];
  std::string rec_model_path = argv[3];
  std::string label_path = argv[4];
  int cpu_num_threads = 4;
  int warm_up = 2;
  int repeats = 5;
  int batch_size = 1;
  std::vector<int> det_input_shape{batch_size, 3, 640, 640};
  std::vector<int> rec_input_shape{batch_size, 3, 224, 224};

  // Do inference on input image
  std::vector<PPShiTu::ObjectResult> det_result;
  std::vector<cv::Mat> batch_imgs;
  std::vector<std::string> all_img_paths;
  std::vector<cv::String> cv_all_img_paths;
  cv::glob(img_dir, cv_all_img_paths);
  for (const auto &img_path : cv_all_img_paths) {
    all_img_paths.push_back(img_path);
  }

  PipeLine *pp_shitu =
      new PipeLine(det_model_path, rec_model_path, label_path, det_input_shape,
                   rec_input_shape, cpu_num_threads, warm_up, repeats);

  for (int i = 0; i < all_img_paths.size(); i++) {
    std::string img_path = all_img_paths[i];
    cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: " << img_path
                << "\n";
      exit(-1);
    }
    cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);
    batch_imgs.push_back(srcimg);

    pp_shitu->run(batch_imgs, det_result, batch_size);

    PrintResult(img_path, det_result);
    auto end = img_path.rfind(".");
    auto start = img_path.rfind("/");
    std::string path = img_path.substr(start + 1, end - start - 1);
    std::string suffix = "_result.jpg";
    std::string save_path = "./result/";
    VisualResult(srcimg, det_result, save_path + path + suffix);
    batch_imgs.clear();
    det_result.clear();

    pp_shitu->print_time();
  }
  delete pp_shitu;
  return 0;
}
