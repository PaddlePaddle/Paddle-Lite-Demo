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
  // Face detection result: face rectangle
  cv::Rect roi;
  // Face keypoints detection result: keypoint coordiate
  std::vector<cv::Point2d> keypoints;
  // Score
  float score;
};

int64_t ShapeProduction(const std::vector<int64_t> &shape) {
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
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

void NHWC1ToNC1HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vscale = vdupq_n_f32(std ? (1.0f / std[0]) : 1.0f);
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4_t vin = vld1q_f32(src);
    float32x4_t vsub = vsubq_f32(vin, vmean);
    float32x4_t vs = vmulq_f32(vsub, vscale);
    vst1q_f32(dst, vs);
    src += 4;
    dst += 4;
  }
  for (; i < size; i++) {
    *(dst++) = (*(src++) - mean[0]) / std[0];
  }
}

void HardNms(std::vector<Face> *input, std::vector<Face> *output,
             float iou_threshold) {
  std::sort(input->begin(), input->end(),
            [](const Face &a, const Face &b) { return a.score > b.score; });
  int box_num = input->size();
  std::vector<int> merged(box_num, 0);
  for (int i = 0; i < box_num; i++) {
    if (merged[i])
      continue;
    std::vector<Face> buf;
    buf.push_back(input->at(i));
    merged[i] = 1;

    float h0 = input->at(i).roi.height;
    float w0 = input->at(i).roi.width;

    float area0 = h0 * w0;

    for (int j = i + 1; j < box_num; j++) {
      if (merged[j])
        continue;

      float inner_x0 = input->at(i).roi.x > input->at(j).roi.x
                           ? input->at(i).roi.x
                           : input->at(j).roi.x;
      float inner_y0 = input->at(i).roi.y > input->at(j).roi.y
                           ? input->at(i).roi.y
                           : input->at(j).roi.y;

      float inputi_x1 = input->at(i).roi.x + input->at(i).roi.width;
      float inputi_y1 = input->at(i).roi.y + input->at(i).roi.height;
      float inputj_x1 = input->at(j).roi.x + input->at(j).roi.width;
      float inputj_y1 = input->at(j).roi.y + input->at(j).roi.height;
      float inner_x1 = inputi_x1 < inputj_x1 ? inputi_x1 : inputj_x1;
      float inner_y1 = inputi_y1 < inputj_y1 ? inputi_y1 : inputj_y1;

      float inner_h = inner_y1 - inner_y0 + 1;
      float inner_w = inner_x1 - inner_x0 + 1;
      if (inner_h <= 0 || inner_w <= 0)
        continue;
      float inner_area = inner_h * inner_w;

      float h1 = input->at(j).roi.height;
      float w1 = input->at(j).roi.width;
      float area1 = h1 * w1;

      float score;
      score = inner_area / (area0 + area1 - inner_area);
      if (score > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input->at(j));
      }
    }
    output->push_back(buf[0]);
  }
}

void FaceDetector_Preprocess(std::shared_ptr<PaddlePredictor> predictor,
                             const std::string img_path, int width, int height,
                             cv::Mat &img) { //????
  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, height, width});
  // read img and pre-process
  img = imread(img_path, cv::IMREAD_COLOR);
  //   pre_process(img, width, height, data);
  float means[3] = {0.407843f, 0.694118f, 0.482353f};
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

void FaceDetector_Postprocess(const cv::Mat &rgbImage, std::vector<Face> *faces,
                              std::shared_ptr<PaddlePredictor> &predictor,
                              float scoreThreshold_) {
  int imageWidth = rgbImage.cols;
  int imageHeight = rgbImage.rows;
  // Get output tensor
  auto outputTensor = predictor->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);

  auto outputTensor1 = predictor->GetOutput(1);
  auto outputData1 = outputTensor1->data<float>();

  faces->clear();
  std::vector<Face> faces_tmp;
  for (int i = 0; i < outputSize; i += 2) {
    // Class id
    float class_id = outputData[i];
    // Confidence score
    float score = outputData[i + 1];
    int left = outputData1[2 * i] * imageWidth;
    int top = outputData1[2 * i + 1] * imageHeight;
    int right = outputData1[2 * i + 2] * imageWidth;
    int bottom = outputData1[2 * i + 3] * imageHeight;
    int width = right - left;
    int height = bottom - top;
    if (score > scoreThreshold_ && score < 1) {
      Face face;
      face.roi = cv::Rect(left, top, width, height) &
                 cv::Rect(0, 0, imageWidth - 1, imageHeight - 1);
      face.score = score;
      faces_tmp.push_back(face);
    }
  }
  HardNms(&faces_tmp, faces, 0.5);
}

void FaceKeypointsDetector_Preprocess(
    std::shared_ptr<PaddlePredictor> &predictor, const cv::Mat &rgbImage,
    const std::vector<Face> &faces, std::vector<cv::Rect> *adjustedFaceROIs,
    int height, int width) {
  // Prepare input tensor
  auto inputTensor = predictor->GetInput(0);
  int batchSize = faces.size();
  std::vector<int64_t> inputShape = {batchSize, 1, width, height};
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
    // Update the face region with adjusted roi
    (*adjustedFaceROIs)[i] =
        cv::Rect(cx - w / 2, cy - h / 2, w, h) &
        cv::Rect(0, 0, rgbImage.cols - 1, rgbImage.rows - 1);
    // Crop and obtain the face image
    cv::Mat resizedRGBImage(rgbImage, (*adjustedFaceROIs)[i]);
    cv::resize(resizedRGBImage, resizedRGBImage,
               cv::Size(inputShape[3], inputShape[2]));
    cv::Mat resizedGRAYImage;
    cv::cvtColor(resizedRGBImage, resizedGRAYImage, cv::COLOR_RGB2GRAY);
    resizedGRAYImage.convertTo(resizedGRAYImage, CV_32FC1);
    cv::Mat mean, std;
    cv::meanStdDev(resizedGRAYImage, mean, std);
    float inputMean = static_cast<float>(mean.at<double>(0, 0));
    float inputStd = static_cast<float>(std.at<double>(0, 0)) + 0.000001f;
    NHWC1ToNC1HW(reinterpret_cast<const float *>(resizedGRAYImage.data),
                 inputData, &inputMean, &inputStd, inputShape[3],
                 inputShape[2]);
    inputData += inputShape[1] * inputShape[2] * inputShape[3];
  }
}

void draw(const std::vector<Face> &faces, cv::Mat *rgbImage) {
  for (int i = 0; i < faces.size(); i++) {
    auto roi = faces[i].roi;
    // Configure color
    for (int j = 0; j < faces[i].keypoints.size(); j++) {
      cv::circle(*rgbImage, faces[i].keypoints[j], 1, cv::Scalar(0, 255, 0),
                 2); //在图像中画出特征点，1是圆的半径
    }
  }
  cv::imwrite("face_keypoints.jpg", *rgbImage);
}

void FaceKeypointsDetector_Postprocess(
    std::shared_ptr<PaddlePredictor> &predictor,
    const std::vector<cv::Rect> &adjustedFaceROIs, std::vector<Face> *faces,
    cv::Mat &img) {
  auto outputTensor = predictor->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  int batchSize = faces->size();
  int keypointsNum = outputSize / batchSize;
  assert(batchSize == adjustedFaceROIs.size());
  assert(keypointsNum == 136); // 68 x 2
  for (int i = 0; i < batchSize; i++) {
    // Face keypoints with coordinates (x, y)
    for (int j = 0; j < keypointsNum; j += 2) {
      (*faces)[i].keypoints.push_back(cv::Point2d(
          adjustedFaceROIs[i].x + outputData[j] * adjustedFaceROIs[i].width,
          adjustedFaceROIs[i].y +
              outputData[j + 1] * adjustedFaceROIs[i].height));
    }
    outputData += keypointsNum;
  }
  draw(*faces, &img);
}

void run_model(std::string facedetection_model_file,
               std::string facekeypoints_model_file, std::string img_path,
               int height0, int width0, int height1, int width1, int power_mode,
               int thread_num, int repeats, int warmup) {

  // 1. Set MobileConfig
  MobileConfig facedetection_config;
  facedetection_config.set_model_from_file(facedetection_model_file);
  facedetection_config.set_power_mode(
      static_cast<paddle::lite_api::PowerMode>(power_mode));
  facedetection_config.set_threads(thread_num);
  MobileConfig facekeypoints_config;
  facekeypoints_config.set_model_from_file(facekeypoints_model_file);
  facekeypoints_config.set_power_mode(
      static_cast<paddle::lite_api::PowerMode>(power_mode));
  facekeypoints_config.set_threads(thread_num);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> facedetection_predictor =
      CreatePaddlePredictor<MobileConfig>(facedetection_config);
  std::shared_ptr<PaddlePredictor> facekeypoints_predictor =
      CreatePaddlePredictor<MobileConfig>(facekeypoints_config);

  cv::Mat img;
  std::vector<Face> faces;
  std::vector<cv::Rect> adjustedFaceROIs;
  float scoreThreshold = 0.7;

  // 3. Prepare FaceDetector input data from image
  FaceDetector_Preprocess(facedetection_predictor, img_path, width0, height0,
                          img);

  // 4. Run predictor
  double facedetection_first_duration{-1};
  double facekeypoints_first_duration{-1};
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      auto start = GetCurrentUS();
      facedetection_predictor->Run();
      facedetection_first_duration = (GetCurrentUS() - start) / 1000.0;
      FaceDetector_Postprocess(img, &faces, facedetection_predictor,
                               scoreThreshold);
      if (faces.size() > 0) {
        adjustedFaceROIs.resize(faces.size());
        FaceKeypointsDetector_Preprocess(facekeypoints_predictor, img, faces,
                                         &adjustedFaceROIs, height1, width1);
        start = GetCurrentUS();
        facekeypoints_predictor->Run();
        facekeypoints_first_duration = (GetCurrentUS() - start) / 1000.0;
      } else {
        facekeypoints_first_duration = 0.f;
      }
    } else {
      facedetection_predictor->Run();
      FaceDetector_Postprocess(img, &faces, facedetection_predictor,
                               scoreThreshold);
      if (faces.size() > 0) {
        adjustedFaceROIs.resize(faces.size());
        FaceKeypointsDetector_Preprocess(facekeypoints_predictor, img, faces,
                                         &adjustedFaceROIs, height1, width1);
        facekeypoints_predictor->Run();
      }
    }
  }

  double facedetection_sum_duration = 0.0;
  double facedetection_max_duration = 1e-5;
  double facedetection_min_duration = 1e5;
  double facedetection_avg_duration = -1;
  double facekeypoints_sum_duration = 0.0;
  double facekeypoints_max_duration = 1e-5;
  double facekeypoints_min_duration = 1e5;
  double facekeypoints_avg_duration = -1;
  bool first = true;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    auto start = GetCurrentUS();
    facedetection_predictor->Run();
    auto facedetection_duration = (GetCurrentUS() - start) / 1000.0;
    facedetection_sum_duration += facedetection_duration;
    facedetection_max_duration =
        facedetection_duration > facedetection_max_duration
            ? facedetection_duration
            : facedetection_max_duration;
    facedetection_min_duration =
        facedetection_duration < facedetection_min_duration
            ? facedetection_duration
            : facedetection_min_duration;
    if (facedetection_first_duration < 0 && first) {
      facedetection_first_duration = facedetection_duration;
    }
    FaceDetector_Postprocess(img, &faces, facedetection_predictor,
                             scoreThreshold);
    if (faces.size() > 0) {
      adjustedFaceROIs.resize(faces.size());
      FaceKeypointsDetector_Preprocess(facekeypoints_predictor, img, faces,
                                       &adjustedFaceROIs, height1, width1);
      start = GetCurrentUS();
      facekeypoints_predictor->Run();
      auto facekeypoints_duration = (GetCurrentUS() - start) / 1000.0;
      facekeypoints_sum_duration += facekeypoints_duration;
      facekeypoints_max_duration =
          facekeypoints_duration > facekeypoints_max_duration
              ? facekeypoints_duration
              : facekeypoints_max_duration;
      facekeypoints_min_duration =
          facekeypoints_duration < facekeypoints_min_duration
              ? facekeypoints_duration
              : facekeypoints_min_duration;
      if (facekeypoints_first_duration < 0 && first) {
        facekeypoints_first_duration = facekeypoints_duration;
      }
    } else {
      facekeypoints_sum_duration = 0.f;
      facekeypoints_max_duration = 0.f;
      facekeypoints_min_duration = 0.f;
    }
    first = false;
  }

  FaceKeypointsDetector_Postprocess(facekeypoints_predictor, adjustedFaceROIs,
                                    &faces, img);

  facedetection_avg_duration =
      facedetection_sum_duration / static_cast<float>(repeats);
  facekeypoints_avg_duration =
      facekeypoints_sum_duration / static_cast<float>(repeats);
  std::cout
      << "\n======= benchmark summary =======\n"
      << "input_shape(s) (NCHW): {1, 3, " << height0 << ", " << width0 << "}\n"
      << "model_dir:" << facedetection_model_file << "\n"
      << "model_dir:" << facekeypoints_model_file << "\n"
      << "warmup:" << warmup << "\n"
      << "repeats:" << repeats << "\n"
      << "power_mode:" << power_mode << "\n"
      << "thread_num:" << thread_num << "\n"
      << "*** time info(ms) ***\n"
      << "facedetection_1st_duration:" << facedetection_first_duration << "\n"
      << "facedetection_max_duration:" << facedetection_max_duration << "\n"
      << "facedetection_min_duration:" << facedetection_min_duration << "\n"
      << "facedetection_avg_duration:" << facedetection_avg_duration << "\n"
      << "facekeypoints_1st_duration:" << facekeypoints_first_duration << "\n"
      << "facekeypoints_max_duration:" << facekeypoints_max_duration << "\n"
      << "facekeypoints_min_duration:" << facekeypoints_min_duration << "\n"
      << "facekeypoints_avg_duration:" << facekeypoints_avg_duration << "\n";

  // 5. Get output and post process
  std::cout << "\n====== output summary ====== " << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr
        << "[ERROR] usage: " << argv[0]
        << " facedetection_model_file  facekeypoints_model_file image_path\n";
    exit(1);
  }
  std::cout << "This parameters are optional: \n"
            << " <facedetection input_height>, eg: 480 \n"
            << " <facedetection input_width>, eg: 640 \n"
            << " <face_keypoints_detection input_height>, eg: 60 \n"
            << " <face_keypoints_detection input_width>, eg: 60 \n"
            << "  <thread_num>, eg: 1 for single thread \n"
            << "  <repeats>, eg: 100\n"
            << "  <warmup>, eg: 10\n"
            << "  <power_mode>, 0: big cluster, high performance\n"
               "                1: little cluster\n"
               "                2: all cores\n"
               "                3: no bind\n"
            << std::endl;

  std::string facedetection_model_file = argv[1];
  std::string facekeypoints_model_file = argv[2];
  std::string img_path = argv[3];
  int height0 = 480;
  int width0 = 640;
  int height1 = 60;
  int width1 = 60;
  int thread_num = 1;
  int repeats = 1;
  int warmup = 0;
  int power_mode = 0;
  if (argc > 4) {
    height0 = atoi(argv[4]);
  }
  if (argc > 5) {
    width0 = atoi(argv[5]);
  }
  if (argc > 6) {
    height1 = atoi(argv[6]);
  }
  if (argc > 7) {
    width1 = atoi(argv[7]);
  }
  if (argc > 8) {
    thread_num = atoi(argv[8]);
  }
  if (argc > 9) {
    repeats = atoi(argv[9]);
  }
  if (argc > 10) {
    warmup = atoi(argv[10]);
  }
  if (argc > 11) {
    power_mode = atoi(argv[11]);
  }

  run_model(facedetection_model_file, facekeypoints_model_file, img_path,
            height0, width0, height1, width1, power_mode, thread_num, repeats,
            warmup);
  return 0;
}
