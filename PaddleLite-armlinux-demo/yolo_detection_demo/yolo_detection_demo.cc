// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <vector>

int WARMUP_COUNT = 0;
int REPEAT_COUNT = 1;
const int CPU_THREAD_NUM = 2;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;
const std::vector<int64_t> INPUT0_SHAPE = {1, 3, 608, 608};
const std::vector<int64_t> INPUT1_SHAPE = {1, 2};
const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
const std::vector<float> scale = {0.229f, 0.224f, 0.225f};
const float SCORE_THRESHOLD = 0.5f;
const char* class_names[] = {"person",        "bicycle",      "car",
                             "motorcycle",    "airplane",     "bus",
                             "train",         "truck",        "boat",
                             "traffic light", "fire hydrant", "stop sign",
                             "parking meter", "bench",        "bird",
                             "cat",           "dog",          "horse",
                             "sheep",         "cow",          "elephant",
                             "bear",          "zebra",        "giraffe",
                             "backpack",      "umbrella",     "handbag",
                             "tie",           "suitcase",     "frisbee",
                             "skis",          "snowboard",    "sports ball",
                             "kite",          "baseball bat", "baseball glove",
                             "skateboard",    "surfboard",    "tennis racket",
                             "bottle",        "wine glass",   "cup",
                             "fork",          "knife",        "spoon",
                             "bowl",          "banana",       "apple",
                             "sandwich",      "orange",       "broccoli",
                             "carrot",        "hot dog",      "pizza",
                             "donut",         "cake",         "chair",
                             "couch",         "potted plant", "bed",
                             "dining table",  "toilet",       "tv",
                             "laptop",        "mouse",        "remote",
                             "keyboard",      "cell phone",   "microwave",
                             "oven",          "toaster",      "sink",
                             "refrigerator",  "book",         "clock",
                             "vase",          "scissors",     "teddy bear",
                             "hair drier",    "toothbrush"};

struct RESULT {
  cv::Rect rec;
  int class_id;
  float prob;
};

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
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

void preprocess(cv::Mat &img, int width,
                int height, float *input_data) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f, cv::INTER_CUBIC);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale = {0.229f, 0.224f, 0.225f};
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  int image_size = height * width;
  // NHWC->NCHW
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.0f / scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.0f / scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.0f / scale[2]);
  float *input_data_c0 = input_data;
  float *input_data_c1 = input_data + image_size;
  float *input_data_c2 = input_data + image_size * 2;
  int i = 0;
  for (; i < image_size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(dimg);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(input_data_c0, vs0);
    vst1q_f32(input_data_c1, vs1);
    vst1q_f32(input_data_c2, vs2);
    dimg += 12;
    input_data_c0 += 4;
    input_data_c1 += 4;
    input_data_c2 += 4;
  }
  for (; i < image_size; i++) {
    *(input_data_c0++) = (*(dimg++) - mean[0]) / scale[0];
    *(input_data_c1++) = (*(dimg++) - mean[1]) / scale[1];
    *(input_data_c2++) = (*(dimg++) - mean[2]) / scale[2];
  }
}

std::vector<RESULT> postprocess(const float *output_data, int64_t output_size,
                                const float score_threshold,
                                cv::Mat &image) {
  if (output_data == nullptr) {
    std::cerr << "[ERROR] data can not be nullptr\n";
    exit(1);
  }
  std::vector<RESULT> rect_out;
  for (int iw = 0; iw < output_size; iw++) {
    int oriw = image.cols;
    int orih = image.rows;
    if (output_data[1] > score_threshold) {
      RESULT obj;
      int x = static_cast<int>(output_data[2]);
      int y = static_cast<int>(output_data[3]);
      int w = static_cast<int>(output_data[4] - output_data[2] + 1);
      int h = static_cast<int>(output_data[5] - output_data[3] + 1);
      cv::Rect rec_clip =
          cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
      obj.class_id = static_cast<int>(output_data[0]);
      obj.prob = output_data[1];
      obj.rec = rec_clip;
      if (w > 0 && h > 0 && obj.prob <= 1) {
        rect_out.push_back(obj);
        cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        std::string str_prob = std::to_string(obj.prob);
        std::string text = std::string(class_names[obj.class_id]) + ": " +
                           str_prob.substr(0, str_prob.find(".") + 4);
        int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double font_scale = 1.f;
        int thickness = 1;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        float new_font_scale = w * 0.5 * font_scale / text_size.width;
        text_size = cv::getTextSize(
            text, font_face, new_font_scale, thickness, nullptr);
        cv::Point origin;
        origin.x = x + 3;
        origin.y = y + text_size.height + 3;
        cv::putText(image,
                    text,
                    origin,
                    font_face,
                    new_font_scale,
                    cv::Scalar(0, 255, 255),
                    thickness,
                    cv::LINE_AA);

        std::cout << "detection, image size: " << image.cols << ", "
                  << image.rows
                  << ", detect object: " << class_names[obj.class_id]
                  << ", score: " << obj.prob << ", location: x=" << x
                  << ", y=" << y << ", width=" << w << ", height=" << h
                  << std::endl;
      }
    }
    output_data += 6;
  }
  return rect_out;
}

cv::Mat process(cv::Mat &input_image, std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {
  // Preprocess image and fill the data of input tensor
  int input_width = INPUT0_SHAPE[2];
  int input_height = INPUT0_SHAPE[3];
  // Input 0
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize(INPUT0_SHAPE);
  auto *input_data0 = input_tensor0->mutable_data<float>();
  preprocess(input_image, input_width, input_height,
             input_data0);
  // Input 1
  std::unique_ptr<paddle::lite_api::Tensor> input_tensor1(std::move(predictor->GetInput(1)));
  input_tensor1->Resize(INPUT1_SHAPE);
  auto* input_data1 = input_tensor1->mutable_data<int>();
  input_data1[0] = input_image.rows;
  input_data1[1] = input_image.cols;
  // Run predictor
  // warm up to skip the first inference and get more stable time, remove it in
  // actual products
  for (int i = 0; i < WARMUP_COUNT; i++) {
    predictor->Run();
  }
  // repeat to obtain the average time, set REPEAT_COUNT=1 in actual products
  double prediction_time;
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
         WARMUP_COUNT, REPEAT_COUNT, prediction_time,
         max_time_cost, min_time_cost);

  // Get the data of output tensor and postprocess to output detected objects
  std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  const float *output_data = output_tensor->data<float>();
  int64_t output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }
  cv::Mat output_image = input_image.clone();
  auto rec_out = postprocess(output_data, static_cast<int>(output_size / 6), SCORE_THRESHOLD, output_image);
  return output_image;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf(
        "Usage: \n"
        "./yolov3_detection_demo model_dir label_path [input_image_path] [output_image_path]"
        "use images from camera if input_image_path isn't provided.");
    return -1;
  }

  std::string model_path = argv[1];

  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(model_path);
  config.set_threads(CPU_THREAD_NUM);
  config.set_power_mode(CPU_POWER_MODE);

  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);

  if (argc > 3) {
    WARMUP_COUNT = 1;
    REPEAT_COUNT = 5;
    std::string input_image_path = argv[2];
    std::string output_image_path = argv[3];
    cv::Mat input_image = cv::imread(input_image_path);
    cv::Mat output_image = process(input_image, predictor);
    cv::imwrite(output_image_path, output_image);
    cv::imshow("Object Detection Demo", output_image);
    cv::waitKey(0);
  } else {
    cv::VideoCapture cap(-1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    if (!cap.isOpened()) {
      return -1;
    }
    while (1) {
      cv::Mat input_image;
      cap >> input_image;
      cv::Mat output_image = process(input_image, predictor);
      cv::imshow("Object Detection Demo", output_image);
      if (cv::waitKey(1) == char('q')) {
        break;
      }
    }
    cap.release();
    cv::destroyAllWindows();
  }
  return 0;
}
