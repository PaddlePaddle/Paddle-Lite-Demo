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

struct RESULT {
  std::string class_name;
  cv::Scalar fill_color;
  float score;
  float x;
  float y;
  float w;
  float h;
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

std::vector<std::string> LoadLabelList(const std::string &labelPath) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(labelPath);
  while (file) {
    std::string line;
    std::getline(file, line);
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

std::vector<cv::Scalar> GenerateColorMap(int numOfClasses) {
  std::vector<cv::Scalar> colorMap = std::vector<cv::Scalar>(numOfClasses);
  for (int i = 0; i < numOfClasses; i++) {
    int j = 0;
    int label = i;
    int R = 0, G = 0, B = 0;
    while (label) {
      R |= (((label >> 0) & 1) << (7 - j));
      G |= (((label >> 1) & 1) << (7 - j));
      B |= (((label >> 2) & 1) << (7 - j));
      j++;
      label >>= 3;
    }
    colorMap[i] = cv::Scalar(R, G, B);
  }
  return colorMap;
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

void NHWC3ToNC3HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean0 = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vmean1 = vdupq_n_f32(mean ? mean[1] : 0.0f);
  float32x4_t vmean2 = vdupq_n_f32(mean ? mean[2] : 0.0f);
  float scale0 = std ? (1.0f / std[0]) : 1.0f;
  float scale1 = std ? (1.0f / std[1]) : 1.0f;
  float scale2 = std ? (1.0f / std[2]) : 1.0f;
  float32x4_t vscale0 = vdupq_n_f32(scale0);
  float32x4_t vscale1 = vdupq_n_f32(scale1);
  float32x4_t vscale2 = vdupq_n_f32(scale2);
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(src);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dst_c0, vs0);
    vst1q_f32(dst_c1, vs1);
    vst1q_f32(dst_c2, vs2);
    src += 12;
    dst_c0 += 4;
    dst_c1 += 4;
    dst_c2 += 4;
  }
  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) * scale0;
    *(dst_c1++) = (*(src++) - mean[1]) * scale1;
    *(dst_c2++) = (*(src++) - mean[2]) * scale2;
  }
}

void pre_process(std::shared_ptr<PaddlePredictor> predictor,
                 const cv::Mat rgbaImage, int width, int height) {
  // Set the data of input image
  auto inputTensor = predictor->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, height, width};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage,
             cv::Size(inputShape[3], inputShape[2]));
  cv::Mat resizedRGBImage;
  cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);
  resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0 / 255.0f);

  std::vector<float> inputMean = {0.485, 0.456, 0.406};
  std::vector<float> inputStd = {0.229, 0.224, 0.225};

  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data), inputData,
               inputMean.data(), inputStd.data(), inputShape[3], inputShape[2]);
  // Set the size of input image
  auto sizeTensor = predictor->GetInput(1);
  sizeTensor->Resize({1, 2});
  auto sizeData = sizeTensor->mutable_data<int32_t>();
  sizeData[0] = inputShape[3];
  sizeData[1] = inputShape[2];
}

int64_t ShapeProduction(const std::vector<int64_t> &shape) {
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
}

void post_process(std::shared_ptr<PaddlePredictor> predictor,
                  std::vector<RESULT> *results, float scoreThreshold, int width,
                  int height, std::vector<std::string> labelList,
                  std::vector<cv::Scalar> colorMap) { // NOLINT
  auto outputTensor = predictor->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  for (int i = 0; i < outputSize; i += 6) {
    // Class id
    auto class_id = static_cast<int>(round(outputData[i]));
    // Confidence score
    auto score = outputData[i + 1];
    if (score < scoreThreshold)
      continue;
    RESULT object;
    object.class_name = class_id >= 0 && class_id < labelList.size()
                            ? labelList[class_id]
                            : "Unknow";
    object.fill_color = class_id >= 0 && class_id < colorMap.size()
                            ? colorMap[class_id]
                            : cv::Scalar(0, 0, 0);
    object.score = score;
    object.x = outputData[i + 2] / width;
    object.y = outputData[i + 3] / height;
    object.w = (outputData[i + 4] - outputData[i + 2] + 1) / width;
    object.h = (outputData[i + 5] - outputData[i + 3] + 1) / height;
    results->push_back(object);
  }
}

void VisualizeResults(const std::vector<RESULT> &results, cv::Mat *rgbaImage) {
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
    std::cout << "detection, image size: " << w << ", " << h
              << ", detect object: " << object.class_name
              << ", score: " << object.score
              << ", location: x=" << boundingBox.x << ", y=" << boundingBox.y
              << ", width=" << boundingBox.width
              << ", height=" << boundingBox.height << std::endl;
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
  std::vector<RESULT> result;
  std::vector<cv::Scalar> colorMap;
  colorMap = GenerateColorMap(labels.size());
  post_process(predictor, &result, thresh, width, height, labels, colorMap);
  VisualizeResults(result, &img);

  int start = img_path.find_last_of("/");
  int end = img_path.find_last_of(".");
  std::string img_name = img_path.substr(start + 1, end - start - 1);
  std::string result_name =
      img_name + "_yolo_v3_mobilenetv3_detection_result.jpg";
  cv::imwrite(result_name, img);
  std::cout << "result has been saved to picture: " << result_name << " "
            << std::endl;
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
  int height = 320;
  int width = 320;
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
