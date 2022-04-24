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

void pre_process(const cv::Mat& img_ori, int width, int height, float* data) {
    cv::Mat img = img_ori.clone();
    int w, h, x, y;
    int channelLength = width * height;

    float r_w = width / (img.cols * 1.0);
    float r_h = height / (img.rows * 1.0);
    if (r_h > r_w) {
        w = width;
        h = r_w * img.rows;
        x = 0;
        y = (height - h) / 2;
    } else {
        w = r_h * img.cols;
        h = height;
        x = (width - w) / 2;
        y = 0;
    }

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    //split channels
    out.convertTo(out, CV_32FC3, 1. / 255.);
    cv::Mat input_channels[3];
    cv::split(out, input_channels);
    for (int j = 0; j < 3; j++) {
        memcpy(data + width * height * j, input_channels[2-j].data, channelLength * sizeof(float));
    }
}

static float iou_calc(const cv::Rect& rec_a, const cv::Rect& rec_b) {
    cv::Rect u = rec_a | rec_b;
    cv::Rect s = rec_a & rec_b;
    float s_area = s.area();
    if (s_area < 20)
        return 0.f;
    return u.area() * 1.0 / s_area;
}

static bool cmp(const Object& a, const Object& b) {
    return a.prob > b.prob;
}

static void nms(std::map<int, std::vector<Object>>& src, std::vector<Object>& res, float nms_thresh = 0.45) {
    for (auto it = src.begin(); it != src.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou_calc(item.rec, dets[n].rec) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

void extract_boxes(const float* in, std::map<int, std::vector<Object>>& outs,
    const int& stride, const int* anchors, std::vector<long long>& shape,
    float ratio, float conf_thres, int offx, int offy, int xdim) {

    int cls_num = shape[3] - 5;
    for(int c=0; c < shape[1]; c++) {
        int step = c * shape[2] * shape[3];
        for (int r=0; r <  shape[2]; r++){
            int offset = step + r * shape[3];
            float score = in[offset + 4];

            if (score < conf_thres)
                continue;

	        int max_cls_id = 0;
		    float max_cls_val = 0;
            for (int i=0; i<cls_num; i++) {
                if (in[offset+5+i] > max_cls_val) {
                    max_cls_val = in[offset+5+i];
                    max_cls_id = i;
                }
            }

            score *= max_cls_val;
            if(score < conf_thres)
                continue;

            Object obj;
            int y = int(r / xdim);
            int x = int(r % xdim);
            int cx = static_cast<int>(((in[offset] * 2 - 0.5 + x) * stride - offx) / ratio);
            int cy = static_cast<int>(((in[offset+1] * 2 - 0.5 + y) * stride - offy) / ratio);
            int w = static_cast<int>(pow(in[offset+2] * 2, 2) * anchors[2 * c] / ratio);
            int h = static_cast<int>(pow(in[offset+3] * 2, 2) * anchors[2 * c + 1] /ratio);
            int left = cx - w / 2.0;
            int top = cy - h / 2.0;

            obj.rec = cv::Rect(left, top, w, h); //xywh
            obj.prob = score;
            obj.class_id = max_cls_id;

            if (outs.count(obj.class_id) == 0) outs.emplace(obj.class_id, std::vector<Object>());
            outs[obj.class_id].emplace_back(obj);
        }
    }
}

void post_process(std::shared_ptr<PaddlePredictor> predictor, float thresh,
                  std::vector<std::string> class_names, cv::Mat &image, int in_width, int in_height) { // NOLINT
    const int strides[3] = {8, 16, 32};
    const int anchors[3][6] = {{10, 13, 16, 30, 33, 23}, {30, 61, 62, 45, 59, 119}, {116, 90, 156, 198, 373, 326}};
    std::map<int, std::vector<Object>> raw_outputs;
    float r_w = in_width / float(image.cols);
    float r_h = in_height / float(image.rows);
    float r, off_x, off_y;
    if (r_h > r_w) {
        r = r_w;
        off_x = 0;
        off_y = static_cast<int>((in_height - r_w * image.rows) / 2);
    } else {
        r = r_h;
        off_y = 0;
        off_x = static_cast<int>((in_width - r_h * image.cols) / 2);
    }

    for (int k = 0; k < 3; k++) {
        std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(k)));
        auto* outptr = output_tensor->data<float>();
        auto shape_out = output_tensor->shape();
        int xdim = int(in_width / strides[k]);
        extract_boxes(outptr, raw_outputs, strides[k], anchors[k], shape_out, r, thresh, off_x, off_y, xdim);
    }

    std::vector<Object> outs;
    nms(raw_outputs, outs, 0.45);

    std::cout<<"cls name size: "<<class_names.size()<<std::endl;

    //visualize
    for(auto& obj : outs) {
        cv::rectangle(image, obj.rec, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        std::string str_prob = std::to_string(obj.prob);
        std::string class_name = obj.class_id >= 0 && obj.class_id < class_names.size()
                           ? class_names[obj.class_id]
                           : "Unknow";
        std::string text = class_name + ": " + str_prob.substr(0, str_prob.find(".") + 4);
        int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double font_scale = 1.f;
        int thickness = 2;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        float new_font_scale = obj.rec.width * 0.35 * font_scale / text_size.width;
        text_size = cv::getTextSize(text, font_face, new_font_scale, thickness,
                                    nullptr);
        cv::Point origin;
        origin.x = obj.rec.x + 10;
        origin.y = obj.rec.y + text_size.height + 10;
        cv::putText(image, text, origin, font_face, new_font_scale,
                    cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);

        std::cout << "detection, image size: " << image.cols << ", "
                  << image.rows << ", detect object: " << class_name << ", cls id: " << obj.class_id
                  << ", score: " << obj.prob << ", location: x=" << obj.rec.x
                  << ", y=" << obj.rec.y << ", width=" << obj.rec.width << ", height=" << obj.rec.height
                  << std::endl;
   }
}

void run_model(std::string model_file, std::string img_path,
               const std::vector<std::string> &labels, const float thresh,
               int width, int height, int power_mode, int thread_num,
               int repeats, int warmup) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);
  std::cout<<"model_file: "<<model_file<<std::endl;
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  config.set_threads(thread_num);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data from image
  // read img and pre-process
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, height, width});
  auto* data0 = input_tensor0->mutable_data<float>();
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  pre_process(img, width, height, data0);

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

  post_process(predictor, thresh, labels, img, width, height);
  int start = img_path.find_last_of("/");
  int end = img_path.find_last_of(".");
  std::string img_name = img_path.substr(start + 1, end - start - 1);
  std::string result_name = img_name + "_yolov5n_detection_result.jpg";
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
            << " <thresh>, eg: 0.25 \n"
            << " <input_width>, eg: 640 \n"
            << " <input_height>, eg: 640 \n"
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
  float thresh = 0.25f;
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
