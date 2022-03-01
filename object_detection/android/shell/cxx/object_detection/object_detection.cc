// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h" // NOLINT
#include <fstream>      // NOLINT
#include <iostream>     // NOLINT
#include <sys/time.h>   // NOLINT
#include <time.h>       // NOLINT
#include <vector>       // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library: libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api; // NOLINT

struct Object {
  std::string class_name;
  cv::Rect rec;
  int class_id;
  float prob;
};

void load_labels(const std::string &path, std::vector<std::string> *labels) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    std::cerr << "Load input label file error." << std::endl;
    exit(1);
  }
  std::string line;
  while (getline(ifs, line)) {
    labels->push_back(line);
  }
  ifs.close();
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
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

void pre_process(std::shared_ptr<PaddlePredictor> predictor, const cv::Mat img,
                 int width, int height) {
  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, height, width});
  float means[3] = {0.5f, 0.5f, 0.5f};
  float scales[3] = {0.5f, 0.5f, 0.5f};
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  const float *dimg = reinterpret_cast<const float *>(imgf.data);
  auto *data = input_tensor->mutable_data<float>();
  neon_mean_scale(dimg, data, width * height, means, scales);
}

void post_process(std::shared_ptr<PaddlePredictor> predictor, float thresh,
                  std::vector<std::string> class_names,
                  cv::Mat &image) { // NOLINT
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto *data = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t size = 1;
  for (auto &i : shape_out) {
    size *= i;
  }
  for (int iw = 0; iw < size; iw += 6) {
    int oriw = image.cols;
    int orih = image.rows;
    // Class id
    auto class_id = static_cast<int>(round(data[0]));
    // Confidence score
    auto score = data[1];
    if (score > thresh) {
      Object obj;
      int x = static_cast<int>(data[2] * oriw);
      int y = static_cast<int>(data[3] * orih);
      int w = static_cast<int>(data[4] * oriw) - x;
      int h = static_cast<int>(data[5] * orih) - y;
      cv::Rect rec_clip =
          cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
      obj.class_id = class_id;
      obj.class_name = class_id >= 0 && class_id < class_names.size()
                           ? class_names[class_id]
                           : "Unknow";
      obj.prob = score;
      obj.rec = rec_clip;
      if (w > 0 && h > 0 && obj.prob <= 1) {
        cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        std::string str_prob = std::to_string(obj.prob);
        std::string text =
            obj.class_name + ": " + str_prob.substr(0, str_prob.find(".") + 4);
        int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double font_scale = 1.f;
        int thickness = 2;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        float new_font_scale = w * 0.35 * font_scale / text_size.width;
        text_size = cv::getTextSize(text, font_face, new_font_scale, thickness,
                                    nullptr);
        cv::Point origin;
        origin.x = x + 10;
        origin.y = y + text_size.height + 10;
        cv::putText(image, text, origin, font_face, new_font_scale,
                    cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);

        std::cout << "detection, image size: " << image.cols << ", "
                  << image.rows << ", detect object: " << obj.class_name
                  << ", score: " << obj.prob << ", location: x=" << x
                  << ", y=" << y << ", width=" << w << ", height=" << h
                  << std::endl;
      }
    }
    data += 6;
  }
}

void run_model(std::string model_file, std::string img_path,
               const std::vector<std::string> &labels, const float thresh,
               int width, int height, int power_mode, int thread_num,
               int repeats, int warmup) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  config.set_threads(thread_num);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data from image
  // read img and pre-process
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  pre_process(predictor, img, width, height);

  // 4. Run predictor
  double first_duration{-1};
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      auto start = GetCurrentUS();
      predictor->Run();
      first_duration = (GetCurrentUS() - start) / 1000.0;
    } else {
      predictor->Run();
    }
  }

  double sum_duration = 0.0;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    auto start = GetCurrentUS();

    predictor->Run();

    auto duration = (GetCurrentUS() - start) / 1000.0;
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    if (first_duration < 0) {
      first_duration = duration;
    }
  }

  avg_duration = sum_duration / static_cast<float>(repeats);
  std::cout << "\n======= benchmark summary =======\n"
            << "input_shape(s) (NCHW): {1, 3, " << height << ", " << width
            << "}\n"
            << "model_dir:" << model_file << "\n"
            << "warmup:" << warmup << "\n"
            << "repeats:" << repeats << "\n"
            << "power_mode:" << power_mode << "\n"
            << "thread_num:" << thread_num << "\n"
            << "*** time info(ms) ***\n"
            << "1st_duration:" << first_duration << "\n"
            << "max_duration:" << max_duration << "\n"
            << "min_duration:" << min_duration << "\n"
            << "avg_duration:" << avg_duration << "\n";

  // 5. Get output and post process
  std::cout << "\n====== output summary ====== " << std::endl;

  post_process(predictor, thresh, labels, img);
  int start = img_path.find_last_of("/");
  int end = img_path.find_last_of(".");
  std::string img_name = img_path.substr(start + 1, end - start - 1);
  std::string result_name = img_name + "_object_detection_result.jpg";
  cv::imwrite(result_name, img);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("error\n");
    std::cerr << "[ERROR] usage: " << argv[0]
              << " model_file image_path label_file\n";
    exit(1);
  }
  std::cout << "This parameters are optional: \n"
            << " <thresh>, eg: 0.5 \n"
            << " <input_width>, eg: 300 \n"
            << " <input_height>, eg: 300 \n"
            << "  <power_mode>, 0: big cluster, high performance\n"
               "                1: little cluster\n"
               "                2: all cores\n"
               "                3: no bind\n"
            << "  <thread_num>, eg: 1 for single thread \n"
            << "  <repeats>, eg: 100\n"
            << "  <warmup>, eg: 10\n"
            << "  <use_gpu>, eg: 0\n"
            << std::endl;
  std::string model_file = argv[1];
  std::string img_path = argv[2];
  std::string label_file = argv[3];
  std::vector<std::string> labels;
  load_labels(label_file, &labels);
  float thresh = 0.5f;
  int height = 300;
  int width = 300;
  if (argc > 4) {
    thresh = atof(argv[4]);
  }
  int warmup = 0;
  int repeats = 1;
  int power_mode = 0;
  int thread_num = 1;
  int use_gpu = 0;
  if (argc > 6) {
    width = atoi(argv[5]);
    height = atoi(argv[6]);
  }
  if (argc > 7) {
    thread_num = atoi(argv[7]);
  }
  if (argc > 8) {
    power_mode = atoi(argv[8]);
  }
  if (argc > 9) {
    repeats = atoi(argv[9]);
  }
  if (argc > 10) {
    warmup = atoi(argv[10]);
  }
  if (argc > 11) {
    use_gpu = atoi(argv[11]);
  }
  std::cout << "model_file: " << model_file << "\n";
  if (use_gpu) {
    // check model file name
    int start = model_file.find_first_of("/");
    int end = model_file.find_last_of("/");
    std::string model_name = model_file.substr(start + 1, end - start - 1);
    std::cout << "model_name: " << model_name << "\n";
    if (model_name.find("gpu") == model_name.npos) {
      std::cerr << "[ERROR] predicted-model should use gpu model when use_gpu "
                   "is true \n";
      exit(1);
    }
  }

  run_model(model_file, img_path, labels, thresh, width, height, power_mode,
            thread_num, repeats, warmup);
  return 0;
}
