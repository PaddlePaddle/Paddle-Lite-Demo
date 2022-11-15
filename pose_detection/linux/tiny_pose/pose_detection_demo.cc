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
#include <arm_neon.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <fstream>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "./paddle_api.h"
#include "yaml-cpp/yaml.h"

/*
*/
const int WARMUP_COUNT = 1;
const int REPEAT_COUNT = 2;
const int CPU_THREAD_NUM = 2;
int SEARCH_SCOPE = 4;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 128, 96};
std::vector<float> INPUT_MEAN = {0.485f, 0.456f, 0.406f};
std::vector<float> INPUT_STD = {0.229f, 0.224f, 0.225f};
float INPUT_SCALE = 1 / 255.f;
const float SCORE_THRESHOLD = 0.1f;

struct RESULT {
  std::vector<std::array<float, 2>> keypoints;
  std::vector<float> scores;
  int num_joints = -1;
  int kpt_count = 0;
};

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

RESULT get_keypoints(const float *score_map, int num_joints) {
  struct RESULT result;

  // rough center
  int x = 0, y = 0;
  float max_score = -1.f;
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 24; j++) {
      if (score_map[i * 24 + j] > max_score) {
        x = i, y = j;
        max_score = score_map[i * 24 + j];
      }
    }
  }

  // no target
  if (max_score < SCORE_THRESHOLD)
    return result;

  // get potential points
  SEARCH_SCOPE = std::min({SEARCH_SCOPE, x, y, 31 - x, 23 - y});
  for (int i = x - SEARCH_SCOPE; i <= x + SEARCH_SCOPE; i++) {
    for (int j = y - SEARCH_SCOPE; j <= y + SEARCH_SCOPE; j++) {
      result.keypoints.push_back(
          {static_cast<float>(i), static_cast<float>(j)});
      result.scores.push_back(score_map[i * 24 + j]);
      result.kpt_count++;
    }
  }
  result.num_joints = num_joints;
  return result;
}

std::array<float, 2> find_center(RESULT result, float scale_x, float scale_y) {
  float sum_scores = 0.f;
  float pow_scores = 0.f;
  float x = 0.f, y = 0.f;
  for (int i = 0; i < result.kpt_count; i++) {
    pow_scores = pow(result.scores[i], 3);
    x += pow_scores * result.keypoints[i][0];
    y += pow_scores * result.keypoints[i][1];
    sum_scores += pow_scores;
  }
  return {(x / sum_scores + 0.5f) * scale_x - 0.5f,
          (y / sum_scores + 0.5f) * scale_y - 0.5f};
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

void preprocess(const cv::Mat &input_image,
                const std::vector<float> &input_mean,
                const std::vector<float> &input_std, float input_scale,
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

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size,
                                const float score_threshold,
                                cv::Mat *output_image, double time) {
  bool target_detected = true;
  std::vector<RESULT> results;
  float scale_x = output_image.rows / 32.f;
  float scale_y = output_image.cols / 24.f;
  std::vector<cv::Point> kpts;
  std::vector<std::array<int, 2>> link_kpt = {
      {0, 1},   {1, 3},   {3, 5},   {5, 7},   {7, 9},  {5, 11},
      {11, 13}, {13, 15}, {0, 2},   {2, 4},   {4, 6},  {6, 8},
      {8, 10},  {6, 12},  {12, 14}, {14, 16}, {11, 12}};

  // get posted result
  for (int64_t i = 0; i < output_size; i += 768) {
    results.push_back(get_keypoints(output_data + i, i / 768));
    if (results[i / 768].num_joints < 0) {
      target_detected = false;
      break;
    }
  }

  // shell&vision show
  if (target_detected) {
    // shell show
    printf("results: %ld\n", results.size());
    for (int i = 0; i < results.size(); i++) {
      std::array<float, 2> center = find_center(results[i], scale_x, scale_y);
      kpts.push_back(cv::Point(center[1], center[0]));
      printf("[%d] - %f, %f\n", results[i].num_joints, center[0], center[1]);
    }
    // vision show
    for (auto p : kpts) {
      cv::circle(output_image, p, 2, cv::Scalar(0, 0, 255), -1);
    }
    for (auto idx : link_kpt) {
      cv::line(output_image, kpts[idx[0]], kpts[idx[1]], cv::Scalar(255, 0, 0),
               1);
    }
  }
  return results;
}

cv::Mat
process(const cv::Mat &input_image,
        const std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // get model input tensor
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(
      std::move(predictor->GetInput(0)));
  input_tensor->Resize(INPUT_SHAPE);
  int input_width = INPUT_SHAPE[3];
  int input_height = INPUT_SHAPE[2];
  auto *input_data = input_tensor->mutable_data<float>();

// get scale factor
#if 0
  auto scale_factor_tensor = predictor->GetInput(1);
  scale_factor_tensor->Resize({1, 2});
  auto scale_factor_data = scale_factor_tensor->mutable_data<float>();
  scale_factor_data[0] = 1.f;
  scale_factor_data[1] = 1.f;
#endif

  // Run preprocess
  double preprocess_start_time = get_current_us();
  preprocess(input_image, INPUT_MEAN, INPUT_STD, INPUT_SCALE, input_width,
             input_height, input_data);
  double preprocess_end_time = get_current_us();
  double preprocess_time =
      (preprocess_end_time - preprocess_start_time) / 1000.f;
  double prediction_time;

  // Run predictor
  // warm up to skip the first inference for stable time,
  // remove it in actual products
  for (int i = 0; i < WARMUP_COUNT; i++) {
    predictor->Run();
  }
  double max_time_cost = 0.f;
  double min_time_cost = std::numeric_limits<float>::max();
  double total_time_cost = 0.f;
  for (int i = 0; i < REPEAT_COUNT; i++) {
    auto start = get_current_us();
    predictor->Run();
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.f;
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

  // Get output
  std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->mutable_data<float>();
  int64_t output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }
  cv::Mat output_image = input_image.clone();
  double postprocess_start_time = get_current_us();

  // Run postprocess
  std::vector<RESULT> results = postprocess(
      output_data, output_size, SCORE_THRESHOLD, output_image, prediction_time);
  double postprocess_end_time = get_current_us();
  double postprocess_time =
      (postprocess_end_time - postprocess_start_time) / 1000.f;

  // runtime summary
  printf("Preprocess time: %f ms\n", preprocess_time);
  printf("Prediction time: %f ms\n", prediction_time);
  printf("Postprocess time: %f ms\n\n", postprocess_time);
  return output_image;
}

int main(int argc, char **argv) {
  // illegal input
  if (argc < 4 || argc == 5) {
    printf("Usage: \n"
           "./pose_detection_demo model_dir label_path [input_image_path] "
           "[output_image_path]"
           "use images from camera if input_image_path and input_image_path "
           "isn't provided.");
    return -1;
  }

  // save argvs: argv[1]=model argv[2]=subgragh argv[3]=yaml
  std::string model_path = argv[1];
  std::string nnadapter_subgraph_partition_config_path = argv[2];
  std::string yaml_path = argv[3];

  // read yaml
  if (yaml_path != "null") {
    load_yaml_config(yaml_path);
  }

  // Run inference by using full api with CxxConfig
  paddle::lite_api::CxxConfig cxx_config;

  // build model
  if (1) {
    cxx_config.set_model_file(model_path + "/model.pdmodel");
    cxx_config.set_param_file(model_path + "/model.pdiparams");
  } else {
    cxx_config.set_model_dir(model_path);
  }

  // CPU config
  cxx_config.set_threads(CPU_THREAD_NUM);
  cxx_config.set_power_mode(CPU_POWER_MODE);

  // create paddle predictor
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;

  // CPU mode
  std::vector<paddle::lite_api::Place> valid_places;
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});

// NPU mode
#if 1
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
  valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
  cxx_config.set_valid_places(valid_places);
#endif
  std::string device = "verisilicon_timvx";
  cxx_config.set_nnadapter_device_names({device});

// Set the subgraph custom partition configuration file
// cxx_config.set_nnadapter_context_properties(nnadapter_context_properties);
// cxx_config.set_nnadapter_model_cache_dir(nnadapter_model_cache_dir);

// subgragh config
#if 1
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
#endif

  // save opmodel
  try {
    predictor = paddle::lite_api::CreatePaddlePredictor(cxx_config);
    predictor->SaveOptimizedModel(
        model_path, paddle::lite_api::LiteModelType::kNaiveBuffer);
  } catch (std::exception e) {
    printf("An internal error occurred in PaddleLite(cxx config).\n");
  }

  // use ‘MobileConfig’ for .nb
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(model_path + ".nb");
  config.set_threads(CPU_THREAD_NUM);
  config.set_power_mode(CPU_POWER_MODE);
  config.set_nnadapter_device_names({device});
  predictor =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  int coco = 0;
  // image mode: argv[4]=input_image_path  argv[5]=output_image_path
  if (argc > 4) {
    std::string input_image_path = argv[4];
    std::string output_image_path = argv[5];
    cv::Mat input_image = cv::imread(input_image_path);
    cv::Mat output_image = process(input_image, predictor);
    cv::imwrite(output_image_path, output_image);
    cv::imshow("Pose Detection Demo", output_image);
    cv::waitKey(0);
  } else {
    // real-time mode: cap(camera port)
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
      return -1;
    }
    while (1) {
      cv::Mat input_image;
      cap >> input_image;
      cv::Mat output_image = process(input_image, predictor);
      cv::imwrite("../../assets/images/posedet_demo_output" +
                      std::to_string(coco) + ".jpg",
                  output_image);
      coco++;
      cv::imshow("Object Detection Demo", output_image);
      if (cv::waitKey(1) == static_cast<char>('q')) {
        break;
      }
    }
    cap.release();
    cv::destroyAllWindows();
  }

  return 0;
}
