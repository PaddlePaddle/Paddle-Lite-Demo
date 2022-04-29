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

std::mutex mtx;
std::shared_ptr<PaddlePredictor> predictor;
int imgWidth;
int imgHeight;
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

int ShapeProduction(const std::vector<int64_t> &shape) {
  int res = 1;
  for (auto i : shape)
    res *= i;
  return res;
}

void neon_mean_scale(const float *din, float *dout, int size,
                     std::vector<float> mean, std::vector<float> scale) {
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

// fill tensor with mean and scale, neon speed up
void preprocess(const Mat &img, int height, int width, std::vector<float> mean,
                std::vector<float> scale, bool is_scale) {
  std::unique_ptr<Tensor> input_tensor(predictor->GetInput(0));
  input_tensor->Resize({1, 3, height, width});
  input_tensor->mutable_data<float>();
  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  float scale_factor = is_scale ? 1 / 255.f : 1.f;
  img_resize.convertTo(imgf, CV_32FC3, scale_factor);
  const float *dimg = reinterpret_cast<const float *>(imgf.data);
  float *dout = input_tensor->mutable_data<float>();
  neon_mean_scale(dimg, dout, width * height, mean, scale);
}

std::vector<std::vector<float>> postprocess() {
    int topk = 100;
    float nmsParamNmsThreshold = 0.5;
    float confidenceThreshold = 0.5;
    std::vector<std::pair<float, std::vector<float>>> vec;

    auto scoresTensor = predictor->GetOutput(0);
    auto boxsTensor = predictor->GetOutput(1);
    auto scores_Shape = scoresTensor->shape();
    int scoresSize = ShapeProduction(scores_Shape);

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
      std::vector<float> box;
      box.push_back(clampedLeft * imgWidth);
      box.push_back(clampedTop * imgHeight);
      box.push_back(clampedRight * imgWidth);
      box.push_back(clampedBottom * imgHeight);
      vec.push_back(std::make_pair(scores[i + 1], box));
    }

    std::sort(vec.begin(), vec.end(),
              std::greater<std::pair<float, std::vector<float>>>());

    std::vector<int> outputIndex;
    auto computeOverlapAreaRate = [](std::vector<float> anchor1,
                                     std::vector<float> anchor2) -> float {
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
      for (int i = 0; i < vec.size(); i++) {
        if (fabs(vec[i].first - INVALID_ANCHOR) < 1e-5) {
          continue;
        }
        if (++count >= topk) {
          break;
        }
        for (int j = i + 1; j < vec.size(); j++) {
          if (fabs(vec[j].first - INVALID_ANCHOR) > 1e-5) {
            if (computeOverlapAreaRate(vec[i].second, vec[j].second) >
                nmsParamNmsThreshold) {
              vec[j].first = INVALID_ANCHOR;
            }
          }
        }
      }
      for (int i = 0; i < vec.size() && count > 0; i++) {
        if (fabs(vec[i].first - INVALID_ANCHOR) > 1e-5) {
          outputIndex.push_back(i);
          count--;
        }
      }
      std::vector<std::vector<float>> boxAndScores;
      if (outputIndex.size() > 0) {
        for (auto id : outputIndex) {
          if (vec[id].first < confidenceThreshold)
            continue;
          if (isnan(vec[id].first)) { // skip the NaN score, maybe not correct
            continue;
          }
            std::vector<float> boxAndScore;
          for (int k = 0; k < 4; k++)
              boxAndScore.push_back((vec[id].second)[k]); // x1,y1,x2,y2
          boxAndScores.push_back((boxAndScore));       // possibility
        }
      }
      return boxAndScores;
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
@property(nonatomic) long input_height;
@property(nonatomic) long input_width;
@property(nonatomic) int topk;
@property(nonatomic) std::vector<std::string> labels;
@property(nonatomic) cv::Mat cvimg;
- (std::vector<std::string>)load_labels:(const std::string &)path;
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
  _image = [UIImage imageNamed:@"third-party/assets/images/face.jpg"];
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
  self.mean = {0.498,0.498,0.498};
  self.scale = {0.502,0.502,0.502};
  self.input_height = 240;
  self.input_width = 320;
  MobileConfig config;
  config.set_model_from_file(app_dir + "/models/facedetection_for_cpu/model.nb");
  predictor = CreatePaddlePredictor<MobileConfig>(config);
  cv::Mat img_face;
  UIImageToMat(self.image, img_face);
  cv::Mat img;
  if (img_face.channels() == 4) {
    cv::cvtColor(img_face, img, CV_RGBA2RGB);
  }
  imgWidth = img.cols;
  imgHeight = img.rows;
  preprocess(img, self.input_height, self.input_width, self.mean, self.scale,
             true);
  tic.start();
  predictor->Run();
  tic.end();
  auto boxs_scores = postprocess();
  std::ostringstream result;
  result << "\ntime: " << tic.get_average_ms() << " ms";
  self.result.numberOfLines = 0;
  self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
  self.flag_init = true;
    cv::Mat outputImage = img;
    for (auto boxAndScore : boxs_scores) {
      cv::rectangle(outputImage, cv::Point(boxAndScore[0], boxAndScore[1]),
                    cv::Point(boxAndScore[2], boxAndScore[3]),
                    cv::Scalar(0, 0, 255), 2, 8);
    }
  self.imageView.image = MatToUIImage(outputImage);
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
            
            imgWidth = image.cols;
            imgHeight = image.rows;
          preprocess(self->_cvimg, self.input_height, self.input_width,
                     self.mean, self.scale, true);
          tic.start();
          predictor->Run();
          tic.end();
            auto boxs_scores = postprocess();
            std::ostringstream result;
            result << "\ntime: " << tic.get_average_ms() << " ms";
            self.result.numberOfLines = 0;
            self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
            self.flag_init = true;
              cv::Mat outputImage = self->_cvimg;
            cvtColor(outputImage, outputImage, CV_RGB2BGR);
              for (auto boxAndScore : boxs_scores) {
                cv::rectangle(outputImage, cv::Point(boxAndScore[0], boxAndScore[1]),
                              cv::Point(boxAndScore[2], boxAndScore[3]),
                              cv::Scalar(0, 0, 255), 2, 8);
              }
            self.imageView.image = MatToUIImage(outputImage);
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
