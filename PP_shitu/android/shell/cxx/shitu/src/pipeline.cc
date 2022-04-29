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

PipeLine::PipeLine(std::string det_model_path, std::string rec_model_path,
                   std::string label_path, std::vector<int> det_input_shape,
                   std::vector<int> rec_input_shape, int cpu_num_threads,
                   int warm_up, int repeats) {
  det_model_path_ = det_model_path;
  rec_model_path_ = rec_model_path;
  label_path_ = label_path;
  det_input_shape_ = det_input_shape;
  rec_input_shape_ = rec_input_shape;
  cpu_num_threads_ = cpu_num_threads;
  warm_up_ = warm_up;
  repeats_ = repeats;
  // create object detector
  det_ = std::make_shared<PPShiTu::ObjectDetector>(
      det_model_path_, det_input_shape_, cpu_num_threads_, warm_up_, repeats_);
  // create rec model
  rec_ = std::make_shared<PPShiTu::Recognition>(
      rec_model_path_, label_path_, rec_input_shape_, cpu_num_threads_,
      warm_up_, repeats_);
}

void PipeLine::run(std::vector<cv::Mat> batch_imgs,
                   std::vector<PPShiTu::ObjectResult> &det_result, // NOLINT
                   int batch_size) {
  std::fill(times_.begin(), times_.end(), 0);
  DetPredictImage(batch_imgs, &det_result, batch_size, det_, max_det_num_);
  // add the whole image for recognition to improve recall
  for (int i = 0; i < batch_imgs.size(); i++) {
    PPShiTu::ObjectResult result_whole_img = {
        {0, 0, batch_imgs[i].cols, batch_imgs[i].rows}, 0, 1.0};
    det_result.push_back(result_whole_img);
  }

  // get rec result
  for (int j = 0; j < det_result.size(); ++j) {
    std::vector<double> rec_time{};
    int w = det_result[j].rect[2] - det_result[j].rect[0];
    int h = det_result[j].rect[3] - det_result[j].rect[1];
    cv::Rect rect(det_result[j].rect[0], det_result[j].rect[1], w, h);
    cv::Mat crop_img = batch_imgs[0](rect);
    std::vector<PPShiTu::RectResult> result =
        rec_->RunRecModel(crop_img, rec_time);
    det_result[j].rec_result.assign(result.begin(), result.end());
    times_[3] += rec_time[0];
    times_[4] += rec_time[1];
    times_[5] += rec_time[2];
  }

  // rec nms
  auto nms_cost0 = PPShiTu::GetCurrentUS();
  PPShiTu::nms(&det_result, rec_nms_thresold_, true);
  auto nms_cost1 = PPShiTu::GetCurrentUS();
  times_[6] += (nms_cost1 - nms_cost0) / 1000.f;
}

void PipeLine::print_time() {
  std::cout << "\n===================benchmark summary=================\n";
  std::cout << "ObjectDetect Preprocess:  " << times_[0] << "ms\n";
  std::cout << "ObjectDetect inference :  " << times_[1] << "ms\n";
  std::cout << "ObjectDetect Postprocess: " << times_[2] << "ms\n";
  std::cout << "Recongnise   Preprocess:  " << times_[3] << "ms\n";
  std::cout << "Recongnise   inference:   " << times_[4] << "ms\n";
  std::cout << "Recongnise   Postprocess: " << times_[5] << "ms\n";
  std::cout << "nms                     : " << times_[6] << "ms\n";
  std::cout << "=====================================================\n\n";
}

void PipeLine::DetPredictImage(const std::vector<cv::Mat> batch_imgs,
                               std::vector<PPShiTu::ObjectResult> *im_result,
                               const int batch_size_det,
                               std::shared_ptr<PPShiTu::ObjectDetector> det,
                               const int max_det_num) {
  int steps = ceil(batch_imgs.size() * 1.f / batch_size_det);
  for (int idx = 0; idx < steps; idx++) {
    int left_image_cnt = batch_imgs.size() - idx * batch_size_det;
    if (left_image_cnt > batch_size_det) {
      left_image_cnt = batch_size_det;
    }
    // Store all detected result
    std::vector<PPShiTu::ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;

    bool is_rbox = false;
    det->Predict(batch_imgs, &result, &bbox_num, &det_times);
    int item_start_idx = 0;
    for (int i = 0; i < left_image_cnt; i++) {
      cv::Mat im = batch_imgs[i];
      int detect_num = 0;
      for (int j = 0; j < min(bbox_num[i], max_det_num); j++) {
        PPShiTu::ObjectResult item = result[item_start_idx + j];
        if (item.class_id == -1) {
          continue;
        }
        detect_num += 1;
        im_result->push_back(item);
      }
      item_start_idx = item_start_idx + bbox_num[i];
    }

    times_[0] += det_times[0];
    times_[1] += det_times[1];
    times_[2] += det_times[2];
  }
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
  if (argc > 7) {
    cpu_num_threads = atoi(argv[5]);
    warm_up = atoi(argv[6]);
    repeats = atoi(argv[7]);
  } else if (argc > 6) {
    cpu_num_threads = atoi(argv[5]);
    warm_up = atoi(argv[6]);
  } else if (argc > 5) {
    cpu_num_threads = atoi(argv[5]);
  }
  std::cout << "cpu num threads is  " << cpu_num_threads << std::endl;
  std::cout << "warm_up is  " << warm_up << std::endl;
  std::cout << "repeats is  " << repeats << std::endl;

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
