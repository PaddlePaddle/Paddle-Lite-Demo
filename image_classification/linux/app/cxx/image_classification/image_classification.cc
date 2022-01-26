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

#include "paddle_api.h"       // NOLINT
#include <arm_neon.h>         // NOLINT
#include <fstream>            // NOLINT
#include <limits>             // NOLINT
#include <opencv2/opencv.hpp> // NOLINT
#include <stdio.h>            // NOLINT
#include <sys/time.h>         // NOLINT
#include <unistd.h>           // NOLINT
#include <vector>             // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library: libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

int WARMUP_COUNT = 0;
int REPEAT_COUNT = 1;
const int CPU_THREAD_NUM = 2;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 224, 224};

struct RESULT {
  std::string class_name;
  int class_id;
  float score;
};
using namespace paddle::lite_api; // NOLINT

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

std::vector<std::string> load_labels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos);
    }
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float *din, float *dout, int size, float *mean,
                     float *scale) {
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) / scale[0];
    *(dout_c0++) = (*(din++) - mean[1]) / scale[1];
    *(dout_c0++) = (*(din++) - mean[2]) / scale[2];
  }
}

void pre_process(std::shared_ptr<PaddlePredictor> predictor, cv::Mat img,
                 int width, int height) {
  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, height, width});
  // read img and pre-process
  float means[3] = {0.485f, 0.456f, 0.406f};
  float scales[3] = {0.229f, 0.224f, 0.225f};
  cv::Mat resize_image;
  cv::resize(img, resize_image, cv::Size(height, width), 0, 0);
  if (resize_image.channels() == 4) {
    cv::cvtColor(resize_image, resize_image, CV_BGRA2RGB);
  }

  cv::Mat norm_image;
  resize_image.convertTo(norm_image, CV_32FC3, 1 / 255.f);
  const float *dimg = reinterpret_cast<const float *>(norm_image.data);
  auto *data = input_tensor->mutable_data<float>();
  neon_mean_scale(dimg, data, width * height, means, scales);
}

std::vector<RESULT>
post_process(std::shared_ptr<PaddlePredictor> predictor, const int topk,
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

  std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                    std::greater<std::pair<float, int>>());

  std::vector<RESULT> results(topk);
  for (int i = 0; i < vec.size(); i++) {
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

cv::Mat process(const std::string model_file, cv::Mat &input_image, // NOLINT
                const std::vector<std::string> &word_labels,        // NOLINT
                const int topk,
                std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor) {
  // Preprocess image and fill the data of input tensor
  double preprocess_start_time = get_current_us();
  pre_process(predictor, input_image, INPUT_SHAPE[3], INPUT_SHAPE[2]);
  double preprocess_end_time = get_current_us();
  double preprocess_time =
      (preprocess_end_time - preprocess_start_time) / 1000.0f;

  // Run predictor
  // warm up to skip the first inference and get more stable time, remove it in
  // actual products
  for (int i = 0; i < WARMUP_COUNT; i++) {
    predictor->Run();
  }
  // repeat to obtain the average time, set REPEAT_COUNT=1 in actual products
  double sum_duration = 0.0;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (int i = 0; i < REPEAT_COUNT; i++) {
    auto start = get_current_us();
    predictor->Run();
    auto end = get_current_us();
    double cur_time_cost = (end - start) / 1000.0f;
    if (cur_time_cost > max_duration) {
      max_duration = cur_time_cost;
    }
    if (cur_time_cost < min_duration) {
      min_duration = cur_time_cost;
    }
    sum_duration += cur_time_cost;
    printf("iter %d cost: %f ms\n", i, cur_time_cost);
  }
  avg_duration = sum_duration / static_cast<float>(REPEAT_COUNT);
  printf("warmup: %d repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
         WARMUP_COUNT, REPEAT_COUNT, avg_duration, max_duration, min_duration);

  // 5. Get output and postprocess to output detected objects
  std::cout << "\n====== output summary ====== " << std::endl;
  cv::Mat output_image = input_image.clone();
  double postprocess_start_time = get_current_us();
  std::vector<RESULT> results =
      post_process(predictor, topk, word_labels, output_image);
  double postprocess_end_time = get_current_us();
  double postprocess_time =
      (postprocess_end_time - postprocess_start_time) / 1000.0f;

  printf("results: %d\n", results.size());
  for (int i = 0; i < results.size(); i++) {
    printf("Top%d %s - %f\n", i, results[i].class_name.c_str(),
           results[i].score);
  }
  printf("Preprocess time: %f ms\n", preprocess_time);
  printf("Prediction time: %f ms\n", avg_duration);
  printf("Postprocess time: %f ms\n\n", postprocess_time);
  return output_image;
}

void run_model(const std::string model_file,
               const std::vector<std::string> &word_labels, const int topk,
               bool use_cap, std::string input_image_path,
               std::string output_image_path) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_power_mode(CPU_POWER_MODE);
  config.set_threads(CPU_THREAD_NUM);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data from image
  if (use_cap) {
    cv::VideoCapture cap(-1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
      return;
    }
    while (1) {
      cv::Mat input_image;
      cap >> input_image;
      cv::Mat output_image =
          process(model_file, input_image, word_labels, topk, predictor);
      cv::imshow("image classification", output_image);
      if (cv::waitKey(1) == char('q')) { // NOLINT
        break;
      }
    }
    cap.release();
    cv::destroyAllWindows();
  } else {
    cv::Mat input_image = cv::imread(input_image_path, 1);
    cv::Mat output_image =
        process(model_file, input_image, word_labels, topk, predictor);
    cv::imwrite(output_image_path, output_image);
    cv::imshow("image classification", output_image);
    cv::waitKey(0);
  }
}
int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " ./image_classification_demo model_dir label_path [top_k] "
              << "[input_image_path] [output_image_path] \n"
              << "use images from camera if input_image_path isn't provided \n";
    exit(1);
  }

  std::string model_path = argv[1];
  std::string label_path = argv[2];
  int topk = 1;
  if (argc > 3) {
    topk = atoi(argv[3]);
  }
  // Load Labels
  std::vector<std::string> word_labels = load_labels(label_path);
  std::string input_image_path = "";
  std::string output_image_path = "";
  bool use_cap = true;
  if (argc > 4) {
    input_image_path = argv[4];
    WARMUP_COUNT = 1;
    REPEAT_COUNT = 5;
    use_cap = false;
  }
  if (argc > 4) {
    output_image_path = argv[5];
  }

  run_model(model_path, word_labels, topk, use_cap, input_image_path,
            output_image_path);

  return 0;
}
