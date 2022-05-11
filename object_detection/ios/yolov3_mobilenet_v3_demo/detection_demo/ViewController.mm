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

#import "ViewController.h"
#include "include/paddle_api.h"
#include "include/paddle_use_kernels.h"
#include "include/paddle_use_ops.h"
#include "timer.h"
#include <arm_neon.h>
#include <iostream>
#include <mutex>
#import <opencv2/highgui/cap_ios.h>
#import <opencv2/highgui/ios.h>
#import <opencv2/opencv.hpp>
#include <string>
#import <sys/timeb.h>
#include <vector>

using namespace paddle::lite_api;
using namespace cv;

struct Object {
  std::string class_name;
  cv::Scalar fill_color;
  cv::Rect rec;
  float prob;
};

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

std::mutex mtx;
std::shared_ptr<PaddlePredictor> predictor;
Timer tic;
long long count = 0;

double tensor_mean(const Tensor &tin) {
  auto shape = tin.shape();
  int64_t size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  double mean = 0.;
  auto ptr = tin.data<float>();
  for (int i = 0; i < size; i++) {
    mean += ptr[i];
  }
  return mean / size;
}
void NHWC3ToNC3HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean0 = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vmean1 = vdupq_n_f32(mean ? mean[1] : 0.0f);
  float32x4_t vmean2 = vdupq_n_f32(mean ? mean[2] : 0.0f);
  float32x4_t vscale0 = vdupq_n_f32(std ? (1.0f / std[0]) : 1.0f);
  float32x4_t vscale1 = vdupq_n_f32(std ? (1.0f / std[1]) : 1.0f);
  float32x4_t vscale2 = vdupq_n_f32(std ? (1.0f / std[2]) : 1.0f);
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
    *(dst_c0++) = (*(src++) - mean[0]) / std[0];
    *(dst_c1++) = (*(src++) - mean[1]) / std[1];
    *(dst_c2++) = (*(src++) - mean[2]) / std[2];
  }
}

void pre_process(const Mat &img_in, int width, int height, bool is_scale) {
  // Set the data of input image
  auto inputTensor = predictor->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, height, width};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(img_in, resizedRGBAImage, cv::Size(inputShape[3], inputShape[2]));
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

/*
// fill tensor with mean and scale, neon speed up
void pre_process(const Mat &img_in, int width, int height, bool is_scale) {
  if (img_in.channels() == 4) {
    cv::cvtColor(img_in, img_in, CV_RGBA2RGB);
  }
  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(predictor->GetInput(0));
  input_tensor->Resize({1, 3, height, width});
  float means[3] = {0.5f, 0.5f, 0.5f};
  float scales[3] = {0.5f, 0.5f, 0.5f};
  cv::Mat im;
  cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  float scale_factor = is_scale ? 1 / 255.f : 1.f;
  im.convertTo(imgf, CV_32FC3, scale_factor);
  const float *dimg = reinterpret_cast<const float *>(imgf.data);
  float *dout = input_tensor->mutable_data<float>();
  neon_mean_scale(dimg, dout, width * height, means, scales);
}

std::vector<Object> post_process(float thresh,
                                 std::vector<std::string> class_names,
                                 std::vector<cv::Scalar> color_map,
                                 cv::Mat &image) { // NOLINT
  std::unique_ptr<const Tensor> output_tensor(predictor->GetOutput(0));
  auto *data = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t size = 1;
  for (auto &i : shape_out) {
    size *= i;
  }
  std::vector<Object> result;
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
      obj.class_name = class_id >= 0 && class_id < class_names.size()
                           ? class_names[class_id]
                           : "Unknow";
      obj.fill_color = class_id >= 0 && class_id < color_map.size()
                           ? color_map[class_id]
                           : cv::Scalar(0, 0, 0);
      obj.prob = score;
      obj.rec = rec_clip;
      if (w > 0 && h > 0 && obj.prob <= 1) {
        cv::rectangle(image, rec_clip, obj.fill_color, 2);
        std::string str_prob = std::to_string(obj.prob);
        std::string text =
            obj.class_name + ": " + str_prob.substr(0, str_prob.find(".") + 4);
        int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double font_scale = 1.f;
        int thickness = 4;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        float new_font_scale = w * 0.35 * font_scale / text_size.width;
        text_size = cv::getTextSize(text, font_face, new_font_scale, thickness,
                                    nullptr);
        cv::Point origin;
        origin.x = x + 10;
        origin.y = y + text_size.height + 10;
        cv::putText(image, text, origin, font_face, new_font_scale,
                    cv::Scalar(0, 255, 255), thickness);
        result.push_back(obj);
      }
    }
    data += 6;
  }
  return result;
}
*/
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
  }
}

@interface ViewController () <CvVideoCameraDelegate>
@property(weak, nonatomic) IBOutlet UIImageView *imageView;
@property(weak, nonatomic) IBOutlet UISwitch *flag_process;
@property(weak, nonatomic) IBOutlet UISwitch *flag_video;
@property(weak, nonatomic) IBOutlet UIImageView *preView;
@property(weak, nonatomic) IBOutlet UISwitch *flag_back_cam;
@property(weak, nonatomic) IBOutlet UILabel *result;
@property(nonatomic, strong) CvVideoCamera *videoCamera;
@property(nonatomic, strong) UIImage *image;
@property(nonatomic) bool flag_init;
@property(nonatomic) bool flag_cap_photo;
@property(nonatomic) std::vector<float> scale;
@property(nonatomic) std::vector<float> mean;
@property(nonatomic) float thresh;
@property(nonatomic) long input_height;
@property(nonatomic) long input_width;
@property(nonatomic) std::vector<std::string> labels;
@property(nonatomic) std::vector<cv::Scalar> colorMap;
- (std::vector<std::string>)load_labels:(const std::string &)path;
@property(nonatomic) cv::Mat cvimg;
@end

@implementation ViewController
@synthesize imageView;
- (void)viewDidLoad {
  [super viewDidLoad];
  // Do any additional setup after loading the view, typically from a nib.
  _flag_process.on = NO;
  _flag_back_cam.on = NO;
  _flag_video.on = NO;
  _flag_cap_photo = false;
  _image = [UIImage imageNamed:@"third-party/assets/images/dog.jpg"];
  if (_image != nil) {
    printf("load image successed\n");
    imageView.image = _image;
  } else {
    printf("load image failed\n");
  }

  [_flag_process addTarget:self
                    action:@selector(PSwitchValueChanged:)
          forControlEvents:UIControlEventValueChanged];
  [_flag_back_cam addTarget:self
                     action:@selector(CSwitchValueChanged:)
           forControlEvents:UIControlEventValueChanged];

  self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.preView];
  self.videoCamera.delegate = self;
  self.videoCamera.defaultAVCaptureDevicePosition =
      AVCaptureDevicePositionFront;
  self.videoCamera.defaultAVCaptureSessionPreset =
      AVCaptureSessionPreset640x480;
  self.videoCamera.defaultAVCaptureVideoOrientation =
      AVCaptureVideoOrientationPortrait;
  self.videoCamera.rotateVideo = 90;
  self.videoCamera.defaultFPS = 30;
  [self.view insertSubview:self.imageView atIndex:0];
  self.cvimg.create(640, 480, CV_8UC3);
  NSString *path = [[NSBundle mainBundle] bundlePath];
  std::string app_dir = std::string([path UTF8String]) + "/third-party/assets";
  std::string label_file_str = app_dir + "/labels/coco_label_list.txt";
  self.labels = [self load_labels:label_file_str];
  self.colorMap = GenerateColorMap(self.labels.size());
  MobileConfig config;
  config.set_model_from_file(app_dir + "/models/"
                                       "yolov3_mobilenet_v3_prune86_FPGM_320_"
                                       "fp32_fluid_for_cpu_v2_10/model.nb");
  predictor = CreatePaddlePredictor<MobileConfig>(config);
  self.input_height = 320;
  self.input_width = 320;
  self.thresh = 0.5f;
  cv::Mat img_cat;
  UIImageToMat(self.image, img_cat);
  std::unique_ptr<Tensor> input_tensor(predictor->GetInput(0));
  input_tensor->Resize({1, 3, self.input_height, self.input_width});
  input_tensor->mutable_data<float>();
  cv::Mat img;
  if (img_cat.channels() == 4) {
    cv::cvtColor(img_cat, img, CV_RGBA2RGB);
  }
  Timer pre_tic;
  pre_tic.start();
  pre_process(img, self.input_height, self.input_width, true);
  pre_tic.end();
  // warmup
  predictor->Run();
  tic.start();
  predictor->Run();
  tic.end();
  Timer post_tic;
  post_tic.start();
  // auto rec_out = post_process(self.thresh, self.labels, self.colorMap, img);

  std::vector<RESULT> result;
  post_process(predictor, &result, thresh, self.input_width, self.input_height,
               self.labels, colorMap);
  VisualizeResults(result, &img);

  post_tic.end();
  std::ostringstream result;
  result << "preprocess time: " << pre_tic.get_average_ms() << " ms\n";
  result << "predict time: " << tic.get_average_ms() << " ms\n";
  result << "postprocess time: " << post_tic.get_average_ms() << " ms\n";
  self.result.numberOfLines = 0;
  self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
  self.flag_init = true;
  self.imageView.image = MatToUIImage(img);
}

- (std::vector<std::string>)load_labels:(const std::string &)path {
  std::vector<std::string> labels;
  FILE *fp = fopen(path.c_str(), "r");
  if (fp == nullptr) {
    return labels;
  }
  while (!feof(fp)) {
    char str[1024];
    fgets(str, 1024, fp);
    std::string str_s(str);
    labels.push_back(str);
  }
  fclose(fp);
  return labels;
}

- (IBAction)swith_video_photo:(UISwitch *)sender {
  NSLog(@"%@", sender.isOn ? @"video ON" : @"video OFF");
  if (sender.isOn) {
    self.flag_video.on = YES;
  } else {
    self.flag_video.on = NO;
  }
}

- (IBAction)cap_photo:(id)sender {
  if (!self.flag_process.isOn) {
    self.result.text = @"please turn on the camera firstly";
  } else {
    self.flag_cap_photo = true;
  }
}

- (void)PSwitchValueChanged:(UISwitch *)sender {
  NSLog(@"%@", sender.isOn ? @"process ON" : @"process OFF");
  if (sender.isOn) {
    [self.videoCamera start];
  } else {
    [self.videoCamera stop];
  }
}

- (void)CSwitchValueChanged:(UISwitch *)sender {
  NSLog(@"%@", sender.isOn ? @"back ON" : @"back OFF");
  if (sender.isOn) {
    if (self.flag_process.isOn) {
      [self.videoCamera stop];
    }
    self.videoCamera.defaultAVCaptureDevicePosition =
        AVCaptureDevicePositionBack;
    if (self.flag_process.isOn) {
      [self.videoCamera start];
    }
  } else {
    if (self.flag_process.isOn) {
      [self.videoCamera stop];
    }
    self.videoCamera.defaultAVCaptureDevicePosition =
        AVCaptureDevicePositionFront;
    if (self.flag_process.isOn) {
      [self.videoCamera start];
    }
  }
}

- (void)processImage:(cv::Mat &)image {

  dispatch_async(dispatch_get_main_queue(), ^{
    if (self.flag_process.isOn) {
      if (self.flag_init) {
        if (self.flag_video.isOn || self.flag_cap_photo) {
          self.flag_cap_photo = false;
          if (image.channels() == 4) {
            cvtColor(image, self->_cvimg, CV_RGBA2RGB);
          }
          Timer pre_tic;
          pre_tic.start();
          pre_process(self->_cvimg, self.input_height, self.input_width, true);
          pre_tic.end();
          tic.start();
          predictor->Run();
          tic.end();
          Timer post_tic;
          post_tic.start();
          auto rec_out = post_process(self.thresh, self.labels, self.colorMap,
                                      self->_cvimg);
          post_tic.end();
          std::ostringstream result;
          result << "preprocess time: " << pre_tic.get_average_ms() << " ms\n";
          result << "predict time: " << tic.get_average_ms() << " ms\n";
          result << "postprocess time: " << post_tic.get_average_ms()
                 << " ms\n";
          cvtColor(self->_cvimg, self->_cvimg, CV_RGB2BGR);
          self.result.numberOfLines = 0;
          self.result.text =
              [NSString stringWithUTF8String:result.str().c_str()];
          self.imageView.image = MatToUIImage(self->_cvimg);
        }
      }
    }
  });
}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
  // Dispose of any resources that can be recreated.
}

@end
