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
Timer tic;
long long count = 0;

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

// fill tensor with mean and scale, neon speed up
void pre_process(std::shared_ptr<PaddlePredictor> predictor, cv::Mat img,
                 int width, int height) {
  // Prepare input data from image
  float scale[3] = {1.f, 1.f, 1.f};
  float mean[3] = {0.f, 0.f, 0.f};
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, height, width});
  // read img and pre-process
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(rgb_img, rgb_img, cv::Size(width, height), 0.f, 0.f);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1.f);
  const float *dimg = reinterpret_cast<const float *>(imgf.data);
  auto *data = input_tensor->mutable_data<float>();
  neon_mean_scale(dimg, data, width * height, mean, scale);
}

cv::Mat post_process(std::shared_ptr<PaddlePredictor> predictor, cv::Mat img,
                     const std::vector<std::string> &labels) {
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto output_data = output_tensor->data<int64_t>();
  auto output_shape = output_tensor->shape();
  cv::Mat rgb_img;
  cv::resize(img, rgb_img, cv::Size(output_shape[2], output_shape[1]), 0.f,
             0.f);
  if ("background" == labels[0]) {
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        int tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[2] +
                      output_data[i * output_shape[2] + j] * 150;
        rgb_img.at<cv::Vec3b>(i, j)[2] = tmp_pix > 255 ? 255 : tmp_pix;
        tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[1] +
                  output_data[i * output_shape[2] + j] * 150;
        rgb_img.at<cv::Vec3b>(i, j)[1] = tmp_pix > 255 ? 255 : tmp_pix;
      }
    }
  } else {
    for (int i = 0; i < output_shape[1]; i++) {
      for (int j = 0; j < output_shape[2]; j++) {
        int mask_pix = output_data[i * output_shape[2] + j] ^ 0;
        int tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[2] + mask_pix * 150;
        rgb_img.at<cv::Vec3b>(i, j)[2] = tmp_pix > 255 ? 255 : tmp_pix;
        tmp_pix = rgb_img.at<cv::Vec3b>(i, j)[1] + mask_pix * 150;
        rgb_img.at<cv::Vec3b>(i, j)[1] = tmp_pix > 255 ? 255 : tmp_pix;
      }
    }
  }
  cv::resize(rgb_img, rgb_img, cv::Size(img.cols, img.rows), 0.f, 0.f);
  cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
  return rgb_img;
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
  _image = [UIImage imageNamed:@"third-party/assets/images/human.jpg"];
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
  std::string app_dir = std::string([path UTF8String]) + "/third-party/";
  self.mean = {0, 0, 0};
  self.scale = {1, 1, 1};
  self.input_height = 513;
  self.input_width = 513;
  std::string label_file_str = app_dir + "assets/labels/label_list";
  self.labels = [self load_labels:label_file_str];
  MobileConfig config;
  config.set_model_from_file(app_dir + "assets/models/model.nb");
  predictor = CreatePaddlePredictor<MobileConfig>(config);
  cv::Mat img_cat;
  UIImageToMat(self.image, img_cat);
  cv::Mat img;
  if (img_cat.channels() == 4) {
    cv::cvtColor(img_cat, img, CV_RGBA2RGB);
  } else {
    img = img_cat;
  }
  std::cout << "pre0\n";
  pre_process(predictor, img, self.input_width, self.input_height);
  tic.start();
  predictor->Run();
  tic.end();
  cv::Mat res = post_process(predictor, img, self.labels);

  std::ostringstream result;
  result << "time: " << tic.get_average_ms() << " ms";
  self.result.numberOfLines = 0;
  self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
  self.flag_init = true;
  self.imageView.image = MatToUIImage(res);
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
    if (str_s.length() > 0) {
      labels.push_back(str_s);
    }
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
          pre_process(predictor, self->_cvimg, self.input_width,
                      self.input_height);
          tic.start();
          predictor->Run();
          tic.end();
          self->_cvimg = post_process(predictor, self->_cvimg, self.labels);
          std::ostringstream result;
          result << "time: " << tic.get_average_ms() << " ms";
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
