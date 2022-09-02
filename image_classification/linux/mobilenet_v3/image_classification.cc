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

#include "paddle_api.h"
#include "yaml-cpp/yaml.h"
#include <arm_neon.h>
#include <fstream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

int WARMUP_COUNT = 0;
int REPEAT_COUNT = 1;
const int CPU_THREAD_NUM = 2;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 224, 224};
std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};
float INPUT_SCALE = 1 / 255.f;

struct RESULT {
  std::string class_name;
  float score;
  float left;
  float top;
  float right;
  float bottom;
};

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

bool read_file(const std::string &filename, std::vector<char> *contents,
               bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp)
    return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

std::vector<std::string> load_labels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

bool load_yaml_config(std::string yaml_path) {
  YAML::Node cfg;
  try {
    std::cout << "before loadFile" << std::endl;
    cfg = YAML::LoadFile(yaml_path);
  } catch (YAML::BadFile &e) {
    std::cout << "Failed to load yaml file " << yaml_path
              << ", maybe you should check this file." << std::endl;
    return false;
  }
  auto preprocess_cfg = cfg["TestReader"]["sample_transforms"];
  for (const auto &op : preprocess_cfg) {
    if (!op.IsMap()) {
      std::cout << "Require the transform information in yaml be Map type."
                << std::endl;
      std::abort();
    }
    auto op_name = op.begin()->first.as<std::string>();
    if (op_name == "NormalizeImage") {
      INPUT_MEAN = op.begin()->second["mean"].as<std::vector<float>>();
      INPUT_STD = op.begin()->second["std"].as<std::vector<float>>();
      INPUT_SCALE = op.begin()->second["scale"].as<float>();
    }
  }
  return true;
}

void preprocess(cv::Mat &input_image, std::vector<float> &input_mean,
                std::vector<float> &input_std, float input_scale,
                int input_width, int input_height, float *input_data) {
  cv::Mat resize_image;
  cv::resize(input_image, resize_image, cv::Size(input_width, input_height), 0,
             0);
  if (resize_image.channels() == 4) {
    cv::cvtColor(resize_image, resize_image, cv::COLOR_BGRA2RGB);
  }
  cv::Mat norm_image;
  resize_image.convertTo(norm_image, CV_32FC3, input_scale);
  // NHWC->NCHW
  int image_size = input_height * input_width;
  const float *image_data = reinterpret_cast<const float *>(norm_image.data);
  float32x4_t vmean0 = vdupq_n_f32(input_mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(input_mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(input_mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.0f / input_std[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.0f / input_std[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.0f / input_std[2]);
  float *input_data_c0 = input_data;
  float *input_data_c1 = input_data + image_size;
  float *input_data_c2 = input_data + image_size * 2;
  int i = 0;
  for (; i < image_size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(image_data);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(input_data_c0, vs0);
    vst1q_f32(input_data_c1, vs1);
    vst1q_f32(input_data_c2, vs2);
    image_data += 12;
    input_data_c0 += 4;
    input_data_c1 += 4;
    input_data_c2 += 4;
  }
  for (; i < image_size; i++) {
    *(input_data_c0++) = (*(image_data++) - input_mean[0]) / input_std[0];
    *(input_data_c1++) = (*(image_data++) - input_mean[1]) / input_std[1];
    *(input_data_c2++) = (*(image_data++) - input_mean[2]) / input_std[2];
  }
}

std::vector<RESULT>
postprocess(std::shared_ptr<PaddlePredictor> predictor,
            const std::vector<std::string> &labels, // NOLINT
            cv::Mat &output_image) {                // NOLINT
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto *scores = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t size = 1;
  for (auto &i : shape_out) {
    size *= i;
  }
  std::vector<std::pair<float, int>> vec;
  vec.resize(size);
  for (int i = 0; i < size; i++) {
    vec[i] = std::make_pair(scores[i], i);
  }

  std::partial_sort(vec.begin(), vec.begin() + 3, vec.end(),
                    std::greater<std::pair<float, int>>());

  std::vector<RESULT> results(3);
  for (int i = 0; i < 3; i++) {
    float score = vec[i].first;
    int index = vec[i].second;
    results[i].class_name = "Unknown";
    if (index >= 0 && index < labels.size()) {
      results[i].class_name = labels[index];
    }
    results[i].score = score;
    cv::putText(output_image,
                "Top" + std::to_string(i + 1) + "." + results[i].class_name +
                    ":" + std::to_string(results[i].score),
                cv::Point2d(5, i * 18 + 20), cv::FONT_HERSHEY_PLAIN, 1,
                cv::Scalar(51, 255, 255));
  }
  return results;
}

cv::Mat process(cv::Mat &input_image, std::vector<std::string> &word_labels,
                std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // Preprocess image and fill the data of input tensor
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(
      std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  int input_width = INPUT_SHAPE[3];
  int input_height = INPUT_SHAPE[2];
  auto *input_data = input_tensor->mutable_data<float>();

  double preprocess_start_time = get_current_us();
  preprocess(input_image, INPUT_MEAN, INPUT_STD, INPUT_SCALE, input_width,
             input_height, input_data);
  double preprocess_end_time = get_current_us();
  double preprocess_time =
      (preprocess_end_time - preprocess_start_time) / 1000.0f;

  double prediction_time;
  // Run predictor
  // warm up to skip the first inference and get more stable time, remove it in
  // actual products
  for (int i = 0; i < WARMUP_COUNT; i++) {
    predictor->Run();
  }
  // repeat to obtain the average time, set REPEAT_COUNT=1 in actual products
  double max_time_cost = 0.0f;
  double min_time_cost = std::numeric_limits<float>::max();
  double total_time_cost = 0.0f;
  for (int i = 0; i < REPEAT_COUNT; i++) {
    auto start = get_current_us();
    predictor->Run();
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.0f;
    if (cur_time_cost > max_time_cost) {
      max_time_cost = cur_time_cost;
    }
    if (cur_time_cost < min_time_cost) {
      min_time_cost = cur_time_cost;
    }
    total_time_cost += cur_time_cost;
    prediction_time = total_time_cost / REPEAT_COUNT;
    printf("iter %d cost: %f ms\n", i, cur_time_cost);
  }
  printf("warmup: %d repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
         WARMUP_COUNT, REPEAT_COUNT, prediction_time, max_time_cost,
         min_time_cost);

  cv::Mat output_image = input_image.clone();
  double postprocess_start_time = get_current_us();
  std::vector<RESULT> results =
      postprocess(predictor, word_labels, output_image);
  printf("results: %d\n", results.size());
  for (int i = 0; i < results.size(); i++) {
    printf("Top%d %s - %f\n", i, results[i].class_name.c_str(),
           results[i].score);
  }
  double postprocess_end_time = get_current_us();
  double postprocess_time =
      (postprocess_end_time - postprocess_start_time) / 1000.0f;

  printf("Preprocess time: %f ms\n", preprocess_time);
  printf("Prediction time: %f ms\n", prediction_time);
  printf("Postprocess time: %f ms\n\n", postprocess_time);
  return output_image;
}

int main(int argc, char **argv) {
  if (argc < 5 || argc == 6) {
    printf("Usage: \n"
           "./object_detection_demo model_dir label_path [input_image_path] "
           "[output_image_path]"
           "use images from camera if input_image_path and input_image_path "
           "isn't provided.");
    return -1;
  }

  std::string model_path = argv[1];
  std::string label_path = argv[2];
  std::vector<std::string> word_labels = load_labels(label_path);
  std::string nnadapter_subgraph_partition_config_path = argv[3];

  std::string yaml_path = argv[4];
  if (yaml_path != "null") {
    load_yaml_config(yaml_path);
  }

  // Run inference by using full api with CxxConfig
  paddle::lite_api::CxxConfig cxx_config;
  if (1) { // combined model
    cxx_config.set_model_file(model_path + "/model");
    cxx_config.set_param_file(model_path + "/params");
  } else {
    cxx_config.set_model_dir(model_path);
  }
  cxx_config.set_threads(CPU_THREAD_NUM);
  cxx_config.set_power_mode(CPU_POWER_MODE);

  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
  std::vector<paddle::lite_api::Place> valid_places;
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
  cxx_config.set_valid_places(valid_places);
  std::string device = "verisilicon_timvx";
  cxx_config.set_nnadapter_device_names({device});
  // cxx_config.set_nnadapter_context_properties(nnadapter_context_properties);

  // cxx_config.set_nnadapter_model_cache_dir(nnadapter_model_cache_dir);
  // Set the subgraph custom partition configuration file

  if (!nnadapter_subgraph_partition_config_path.empty()) {
    std::vector<char> nnadapter_subgraph_partition_config_buffer;
    if (read_file(nnadapter_subgraph_partition_config_path,
                  &nnadapter_subgraph_partition_config_buffer, false)) {
      if (!nnadapter_subgraph_partition_config_buffer.empty()) {
        std::string nnadapter_subgraph_partition_config_string(
            nnadapter_subgraph_partition_config_buffer.data(),
            nnadapter_subgraph_partition_config_buffer.size());
        cxx_config.set_nnadapter_subgraph_partition_config_buffer(
            nnadapter_subgraph_partition_config_string);
      }
    } else {
      printf("Failed to load the subgraph custom partition configuration file "
             "%s\n",
             nnadapter_subgraph_partition_config_path.c_str());
    }
  }

  try {
    predictor = paddle::lite_api::CreatePaddlePredictor(cxx_config);
    predictor->SaveOptimizedModel(
        model_path, paddle::lite_api::LiteModelType::kNaiveBuffer);
  } catch (std::exception e) {
    printf("An internal error occurred in PaddleLite(cxx config).\n");
  }

  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(model_path + ".nb");
  config.set_threads(CPU_THREAD_NUM);
  config.set_power_mode(CPU_POWER_MODE);
  config.set_nnadapter_device_names({device});
  predictor =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  if (argc > 5) {
    WARMUP_COUNT = 1;
    REPEAT_COUNT = 5;
    std::string input_image_path = argv[5];
    std::string output_image_path = argv[6];
    cv::Mat input_image = cv::imread(input_image_path);
    cv::Mat output_image = process(input_image, word_labels, predictor);
    cv::imwrite(output_image_path, output_image);
    cv::imshow("Image Classification Demo", output_image);
    cv::waitKey(0);
  } else {
    cv::VideoCapture cap(1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
      return -1;
    }
    while (1) {
      cv::Mat input_image;
      cap >> input_image;
      cv::Mat output_image = process(input_image, word_labels, predictor);
      cv::imshow("Image Classification Demo", output_image);
      if (cv::waitKey(1) == char('q')) {
        break;
      }
    }
    cap.release();
    cv::destroyAllWindows();
  }
  return 0;
}
