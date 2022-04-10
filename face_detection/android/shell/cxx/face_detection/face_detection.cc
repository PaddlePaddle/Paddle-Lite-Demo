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
#include <math.h>
#include <sys/time.h> // NOLINT
#include <time.h>     // NOLINT
#include <vector>     // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library: libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api; // NOLINT

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
    *(dout_c1++) = (*(din++) - mean[1]) / scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) / scale[2];
  }
}

int partitionScore(float (*anchors)[4], float *scores, int left, int right) {
  float pivot = scores[left];
  float *pivotA = anchors[left];
  while (left < right) {
    while (left < right && scores[right] <= pivot)
      right--;
    if (left < right) {
      for (int i = 0; i < 4; i++) {
        anchors[left][i] = anchors[right][i];
      }
      scores[left++] = scores[right];
    }
    while (left < right && scores[left] >= pivot)
      left++;
    if (left < right) {
      for (int i = 0; i < 4; i++) {
        anchors[right][i] = anchors[left][i];
      }
      scores[right--] = scores[left];
    }
  }
  scores[left] = pivot;
  for (int i = 0; i < 4; i++) {
    anchors[left][i] = *(pivotA + i);
  }
  return left;
}

void quickSortScore(float (*anchors)[4], float *scores, int left, int right) {
  int dp;
  if (left < right) {
    dp = partitionScore(anchors, scores, left, right);
    quickSortScore(anchors, scores, left, dp - 1);
    quickSortScore(anchors, scores, dp + 1, right);
  }
}

void sortScores(float (*anchors)[4], float *scores, int left, int64_t right) {
  quickSortScore(anchors, scores, left, right);
}

std::vector<int> nmsScoreFilter(float (*anchors)[4], float *score,
                                const int topN, const float thresh,
                                const int length) {
  auto computeOverlapAreaRate = [](float *anchor1, float *anchor2) -> float {
    float xx1 = anchor1[0] > anchor2[0] ? anchor1[0] : anchor2[0];
    float yy1 = anchor1[1] > anchor2[1] ? anchor1[1] : anchor2[1];
    float xx2 = anchor1[2] < anchor2[2] ? anchor1[2] : anchor2[2];
    float yy2 = anchor1[3] < anchor2[3] ? anchor1[3] : anchor2[3];
    float w = xx2 - xx1 + 1;
    float h = yy2 - yy1 + 1;
    if (w < 0 || h < 0) {
      return 0;
    }
    float inter = w * h;
    float anchor1_area1 =
        (anchor1[2] - anchor1[0] + 1) * (anchor1[3] - anchor1[1] + 1);
    float anchor2_area1 =
        (anchor2[2] - anchor2[0] + 1) * (anchor2[3] - anchor2[1] + 1);
    return inter / (anchor1_area1 + anchor2_area1 - inter);
  };
  int count = 0;
  float INVALID_ANCHOR = -10000.0f;
  for (int i = 0; i < length; i++) {
    if (fabs(score[i] - INVALID_ANCHOR) < 1e-5) {
      continue;
    }
    if (++count >= topN) {
      break;
    }
    for (int j = i + 1; j < length; j++) {
      if (fabs(score[j] - INVALID_ANCHOR) > 1e-5) {
        if (computeOverlapAreaRate(anchors[i], anchors[j]) > thresh) {
          score[j] = INVALID_ANCHOR;
        }
      }
    }
  }
  std::vector<int> outputIndex;
  for (int i = 0; i < length && count > 0; i++) {
    if (fabs(score[i] - INVALID_ANCHOR) > 1e-5) {
      outputIndex.push_back(i);
      count--;
    }
  }
  return outputIndex;
}

void draw(std::vector<std::vector<float>> boxAndScores, cv::Mat &outputImage) {
  int i = 0;
  for (auto boxAndScore : boxAndScores) {
    cv::rectangle(outputImage, cv::Point(boxAndScore[0], boxAndScore[1]),
                  cv::Point(boxAndScore[2], boxAndScore[3]),
                  cv::Scalar(0, 0, 255), 2, 8);
  }
  cv::imwrite("face_detection.jpg", outputImage);
}

cv::Mat pre_process(std::shared_ptr<PaddlePredictor> predictor,
                    const std::string img_path, int width, int height) {
  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, height, width});
  // read img and pre-process
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  //   pre_process(img, width, height, data);
  float means[3] = {127 / 255.f, 127 / 255.f, 127 / 255.f};
  float scales[3] = {128 / 255.f, 128 / 255.f, 128 / 255.f};
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  const float *dimg = reinterpret_cast<const float *>(imgf.data);
  auto *data = input_tensor->mutable_data<float>();
  neon_mean_scale(dimg, data, width * height, means, scales);
  return img;
}

void post_process(cv::Mat &outputImage,
                  std::shared_ptr<PaddlePredictor> predictor, const int topk,
                  const float nmsParamNmsThreshold,
                  const float confidenceThreshold) {
  float output_boxs[8840][4];
  float output_scores[8840];

  std::unique_ptr<const Tensor> scoresTensor(
      std::move(predictor->GetOutput(0)));
  std::unique_ptr<const Tensor> boxsTensor(std::move(predictor->GetOutput(1)));
  auto scores_Shape = scoresTensor->shape();
  int64_t scoresSize = 1;
  for (auto s : scores_Shape) {
    scoresSize *= s;
  }

  int imgWidth = outputImage.cols;
  int imgHeight = outputImage.rows;
  int number_boxs = 0;
  float box[4];

  auto *scores = scoresTensor->data<float>();
  auto *boxs = boxsTensor->data<float>();

  for (int i = 0, j = 0; i < scoresSize; i += 2, j += 4) {
    float rawLeft = boxs[j];
    float rawTop = boxs[j + 1];
    float rawRight = boxs[j + 2];
    float rawBottom = boxs[j + 3];
    float clampedLeft = fmax(fmin(rawLeft, 1.f), 0.f);
    float clampedTop = fmax(fmin(rawTop, 1.f), 0.f);
    float clampedRight = fmax(fmin(rawRight, 1.f), 0.f);
    float clampedBottom = fmax(fmin(rawBottom, 1.f), 0.f);
    output_boxs[number_boxs][0] = clampedLeft * imgWidth;
    output_boxs[number_boxs][1] = clampedTop * imgHeight;
    output_boxs[number_boxs][2] = clampedRight * imgWidth;
    output_boxs[number_boxs][3] = clampedBottom * imgHeight;
    output_scores[number_boxs] = scores[i + 1];
    number_boxs = number_boxs + 1;
  }

  sortScores(output_boxs, output_scores, 0, scoresSize - 1);
  auto outputIndex = nmsScoreFilter(output_boxs, output_scores, topk,
                                    nmsParamNmsThreshold, scoresSize - 1);

  std::vector<std::vector<float>> boxAndScores;
  if (outputIndex.size() > 0) {
    for (auto id : outputIndex) {
      if (output_scores[id] < confidenceThreshold)
        continue;
      if (isnan(output_scores[id])) { // skip the NaN score, maybe not correct
        continue;
      }
      std::vector<float> boxScore;
      for (int k = 0; k < 4; k++)
        boxScore.push_back(output_boxs[id][k]); // x1,y1,x2,y2
      boxScore.push_back(output_scores[id]);    // possibility
      boxAndScores.push_back(boxScore);
    }
  }

  draw(boxAndScores, outputImage);
}

void run_model(std::string model_file, std::string img_path, const int topk,
               const float nmsParamNmsThreshold,
               const float confidenceThreshold, int width, int height,
               int power_mode, int thread_num, int repeats, int warmup) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  config.set_threads(thread_num);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data from image
  auto input_image = pre_process(predictor, img_path, width, height);

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

  post_process(input_image, predictor, topk, nmsParamNmsThreshold,
               confidenceThreshold);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0] << " model_file image_path \n";
    exit(1);
  }
  std::cout << "This parameters are optional: \n"
            << " <topk>, eg: 100 \n"
            << " <input_width>, eg: 320 \n"
            << " <input_height>, eg: 240 \n"
            << " <nmsThreshold>, eg: 0.5 \n"
            << " <confidenceThreshold>, eg: 0.5 \n"
            << "  <power_mode>, 0: big cluster, high performance\n"
               "                1: little cluster\n"
               "                2: all cores\n"
               "                3: no bind\n"
            << "  <thread_num>, eg: 1 for single thread \n"
            << "  <repeats>, eg: 100\n"
            << "  <warmup>, eg: 10\n"
            << "  <nmsThreshold>, eg: 0.5\n"
            << "  <confidenceThreshold>, eg: 0.5\n"
            << std::endl;
  std::string model_file = argv[1];
  std::string img_path = argv[2];
  int topk = 100;
  int height = 240;
  int width = 320;
  float nmsParamNmsThreshold = 0.5;
  float confidenceThreshold = 0.5;
  if (argc > 3) {
    topk = atoi(argv[3]);
  }
  if (argc > 4) {
    nmsParamNmsThreshold = atof(argv[4]);
  }
  if (argc > 5) {
    confidenceThreshold = atof(argv[5]);
  }
  int warmup = 0;
  int repeats = 1;
  int power_mode = 0;
  int thread_num = 1;
  int use_gpu = 0;
  if (argc > 7) {
    width = atoi(argv[6]);
    height = atoi(argv[7]);
  }
  if (argc > 8) {
    thread_num = atoi(argv[8]);
  }
  if (argc > 9) {
    power_mode = atoi(argv[9]);
  }
  if (argc > 10) {
    repeats = atoi(argv[10]);
  }
  if (argc > 11) {
    warmup = atoi(argv[11]);
  }

  run_model(model_file, img_path, topk, nmsParamNmsThreshold,
            confidenceThreshold, width, height, power_mode, thread_num, repeats,
            warmup);
  return 0;
}
