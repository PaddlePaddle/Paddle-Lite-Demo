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

struct Face {
  // Detection result: face rectangle
  cv::Rect roi;
  // Classification result: confidence
  float confidence;
  // Classification result : class id
  int classid;
};

int64_t ShapeProduction(const std::vector<int64_t> &shape) {
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
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

void Detector_Preprocess(std::shared_ptr<PaddlePredictor> predictor,
                         const std::string img_path, int width, int height,
                         cv::Mat &rgb_img, cv::Mat &bgr_img) {
  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));

  // read img and pre-process
  bgr_img = imread(img_path, cv::IMREAD_COLOR);
  float means[3] = {0.407843, 0.694118, 0.482353};
  float scales[3] = {0.5f, 0.5f, 0.5f};
  cv::Mat resizedBGRImage;
  cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(bgr_img, resizedBGRImage, cv::Size(width, height));
  resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1 / 255.f);
  input_tensor->Resize({1, 3, width, height});
  const float *dimg = reinterpret_cast<const float *>(resizedBGRImage.data);
  auto *data = input_tensor->mutable_data<float>();
  neon_mean_scale(dimg, data, resizedBGRImage.cols * resizedBGRImage.rows,
                  means, scales);
}

void Detector_Postprocess(const cv::Mat &rgbImage, std::vector<Face> *faces,
                          std::shared_ptr<PaddlePredictor> &predictor,
                          float scoreThreshold) {
  int imageWidth = rgbImage.cols;
  int imageHeight = rgbImage.rows;
  // Get output tensor
  auto outputTensor = predictor->GetOutput(2);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  faces->clear();
  for (int i = 0; i < outputSize; i += 6) {
    // Class id
    float class_id = outputData[i];
    // Confidence score
    float score = outputData[i + 1];
    int left = outputData[i + 2] * imageWidth;
    int top = outputData[i + 3] * imageHeight;
    int right = outputData[i + 4] * imageWidth;
    int bottom = outputData[i + 5] * imageHeight;
    int width = right - left;
    int height = bottom - top;
    if (score > scoreThreshold) {
      Face face;
      face.roi = cv::Rect(left, top, width, height) &
                 cv::Rect(0, 0, imageWidth - 1, imageHeight - 1);
      faces->push_back(face);
    }
  }
}

void MaskClassifier_Preprocess(const cv::Mat &rgbImage,
                               const std::vector<Face> &faces,
                               std::shared_ptr<PaddlePredictor> &predictor) {
  // Prepare input tensor
  auto inputTensor = predictor->GetInput(0);
  int batchSize = faces.size();
  std::vector<int64_t> inputShape = {batchSize, 3, 128, 128};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  for (int i = 0; i < batchSize; i++) {
    // Adjust the face region to improve the accuracy according to the aspect
    // ratio of input image of the target model
    int cx = faces[i].roi.x + faces[i].roi.width / 2.0f;
    int cy = faces[i].roi.y + faces[i].roi.height / 2.0f;
    int w = faces[i].roi.width;
    int h = faces[i].roi.height;
    float roiAspectRatio =
        static_cast<float>(faces[i].roi.width) / faces[i].roi.height;
    float inputAspectRatio = static_cast<float>(inputShape[3]) / inputShape[2];
    if (fabs(roiAspectRatio - inputAspectRatio) > 1e-5) {
      float widthRatio = static_cast<float>(faces[i].roi.width) / inputShape[3];
      float heightRatio =
          static_cast<float>(faces[i].roi.height) / inputShape[2];
      if (widthRatio > heightRatio) {
        h = w / inputAspectRatio;
      } else {
        w = h * inputAspectRatio;
      }
    }
    cv::Mat resizedRGBImage(
        rgbImage, cv::Rect(cx - w / 2, cy - h / 2, w, h) &
                      cv::Rect(0, 0, rgbImage.cols - 1, rgbImage.rows - 1));
    cv::resize(resizedRGBImage, resizedRGBImage,
               cv::Size(inputShape[3], inputShape[2]));
    cv::Mat resizedBGRImage;
    cv::cvtColor(resizedRGBImage, resizedBGRImage, cv::COLOR_RGB2BGR);
    resizedBGRImage.convertTo(resizedBGRImage, CV_32FC3, 1.0 / 255.0f);
    float means[3] = {0.5f, 0.5f, 0.5f};
    float scales[3] = {1.f, 1.f, 1.f};
    neon_mean_scale(reinterpret_cast<const float *>(resizedBGRImage.data),
                    inputData, inputShape[2] * inputShape[3], means, scales);
    inputData += inputShape[1] * inputShape[2] * inputShape[3];
  }
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

void VisualizeResults(const std::vector<Face> &faces, cv::Mat *gbrImage) {
  for (int i = 0; i < faces.size(); i++) {
    auto roi = faces[i].roi;
    // Configure color and text size
    cv::Scalar color;
    std::string text;
    if (faces[i].classid == 1) {
      text = "MASK: ";
      color = cv::Scalar(0, 255, 0);
    } else {
      text = "NO MASK: ";
      color = cv::Scalar(255, 0, 0);
    }
    text += std::to_string(static_cast<int>(faces[i].confidence * 100)) + "%";
    int font_face = cv::FONT_HERSHEY_PLAIN;
    double font_scale = 1.f;
    float thickness = 1;
    cv::Size text_size =
        cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
    font_scale = faces[i].roi.width * font_scale / text_size.width;
    text_size =
        cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
    // Draw roi object, text and background
    cv::rectangle(*gbrImage, faces[i].roi, color, 2);
    cv::rectangle(
        *gbrImage,
        cv::Point2d(faces[i].roi.x,
                    faces[i].roi.y - round(text_size.height * 1.25f)),
        cv::Point2d(faces[i].roi.x + faces[i].roi.width, faces[i].roi.y), color,
        -1);
    cv::putText(*gbrImage, text, cv::Point2d(faces[i].roi.x, faces[i].roi.y),
                font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
  }
}

void MaskClassifier_Postprocess(std::vector<Face> *faces,
                                std::shared_ptr<PaddlePredictor> &predictor) {
  auto outputTensor = predictor->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  int batchSize = faces->size();
  int classNum = outputSize / batchSize;
  for (int i = 0; i < batchSize; i++) {
    (*faces)[i].classid = 0;
    (*faces)[i].confidence = *(outputData++);
    for (int j = 1; j < classNum; j++) {
      auto confidence = *(outputData++);
      if (confidence > (*faces)[i].confidence) {
        (*faces)[i].classid = j;
        (*faces)[i].confidence = confidence;
      }
    }
  }
}

void run_model(std::string detection_model_file,
               std::string classify_model_file, std::string img_path,
               std::string result_img_path, int height, int width,
               int power_mode, int thread_num, int repeats, int warmup) {

  // 1. Set MobileConfig
  MobileConfig detection_config;
  detection_config.set_model_from_file(detection_model_file);
  detection_config.set_power_mode(
      static_cast<paddle::lite_api::PowerMode>(power_mode));
  detection_config.set_threads(thread_num);
  MobileConfig classify_config;
  classify_config.set_model_from_file(classify_model_file);
  classify_config.set_power_mode(
      static_cast<paddle::lite_api::PowerMode>(power_mode));
  classify_config.set_threads(thread_num);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> detection_predictor =
      CreatePaddlePredictor<MobileConfig>(detection_config);
  std::shared_ptr<PaddlePredictor> classify_predictor =
      CreatePaddlePredictor<MobileConfig>(classify_config);

  // 3. Prepare input data from image
  cv::Mat rgbImage;
  cv::Mat gbrImage;
  float scoreThreshold = 0.7;
  std::vector<Face> face;

  // 4. Run predictor
  double first_duration{-1};
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      auto start = GetCurrentUS();

      Detector_Preprocess(detection_predictor, img_path, width, height,
                          rgbImage, gbrImage);
      detection_predictor->Run();
      Detector_Postprocess(rgbImage, &face, detection_predictor,
                           scoreThreshold);
      MaskClassifier_Preprocess(rgbImage, face, classify_predictor);
      classify_predictor->Run();
      MaskClassifier_Postprocess(&face, classify_predictor);

      first_duration = (GetCurrentUS() - start) / 1000.0;
    } else {
      Detector_Preprocess(detection_predictor, img_path, width, height,
                          rgbImage, gbrImage);
      detection_predictor->Run();
      Detector_Postprocess(rgbImage, &face, detection_predictor,
                           scoreThreshold);
      MaskClassifier_Preprocess(rgbImage, face, classify_predictor);
      classify_predictor->Run();
      MaskClassifier_Postprocess(&face, classify_predictor);
    }
  }

  double sum_duration = 0.0;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    auto start = GetCurrentUS();

    Detector_Preprocess(detection_predictor, img_path, width, height, rgbImage,
                        gbrImage);
    detection_predictor->Run();
    Detector_Postprocess(rgbImage, &face, detection_predictor, scoreThreshold);
    MaskClassifier_Preprocess(rgbImage, face, classify_predictor);
    classify_predictor->Run();
    MaskClassifier_Postprocess(&face, classify_predictor);

    auto duration = (GetCurrentUS() - start) / 1000.0;
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    if (first_duration < 0) {
      first_duration = duration;
    }
  }

  VisualizeResults(face, &gbrImage);
  cv::imwrite(result_img_path, gbrImage);

  auto outputTensor = classify_predictor->GetOutput(0);
  auto outputData = outputTensor->data<float>();

  avg_duration = sum_duration / static_cast<float>(repeats);
  std::cout << "\n======= benchmark summary =======\n"
            << "input_shape(s) (NCHW): {1, 3, " << height << ", " << width
            << "}\n"
            << "model_dir:" << detection_model_file << "\n"
            << "model_dir:" << classify_model_file << "\n"
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
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << "detection_model_file, classify_model_file, image_path, "
                 "result_img_path\n";
    exit(1);
  }

  std::cout << "This parameters are optional: \n"
            << " <input_height>, eg: 224 \n"
            << " <input_width>, eg: 224 \n"
            << "  <thread_num>, eg: 1 for single thread \n"
            << "  <repeats>, eg: 100\n"
            << "  <warmup>, eg: 10\n"
            << "  <power_mode>, 0: big cluster, high performance\n"
               "                1: little cluster\n"
               "                2: all cores\n"
               "                3: no bind\n"
            << " <use_gpu>, eg: 0\n"
            << std::endl;
  std::string detection_model_file = argv[1];
  std::string classify_model_file = argv[2];

  std::string img_path = argv[3];
  std::string result_img_path = argv[4];
  int height = 224;
  int width = 224;
  int thread_num = 1;
  int repeats = 1;
  int warmup = 0;
  int power_mode = 0;
  int use_gpu = 0;
  if (argc > 5) {
    height = atoi(argv[5]);
  }
  if (argc > 6) {
    width = atoi(argv[6]);
  }
  if (argc > 7) {
    thread_num = atoi(argv[7]);
  }
  if (argc > 8) {
    repeats = atoi(argv[8]);
  }
  if (argc > 9) {
    warmup = atoi(argv[9]);
  }
  if (argc > 10) {
    power_mode = atoi(argv[10]);
  }
  if (argc > 11) {
    use_gpu = atoi(argv[11]);
  }
  if (use_gpu) {
    // check model file name
    int start = detection_model_file.find_first_of("/");
    int end = detection_model_file.find_last_of("/");
    std::string model_name =
        detection_model_file.substr(start + 1, end - start - 1);
    if (model_name.find("gpu") == model_name.npos) {
      std::cerr << "[ERROR] detection-model should use gpu model when use_gpu "
                   "is true \n";
      exit(1);
    }
    start = classify_model_file.find_first_of("/");
    end = classify_model_file.find_last_of("/");
    model_name = classify_model_file.substr(start + 1, end - start - 1);
    if (model_name.find("gpu") == model_name.npos) {
      std::cerr << "[ERROR] classify-model should use gpu model when use_gpu "
                   "is true \n";
      exit(1);
    }
  }

  run_model(detection_model_file, classify_model_file, img_path,
            result_img_path, height, width, power_mode, thread_num, repeats,
            warmup);
  return 0;
}
