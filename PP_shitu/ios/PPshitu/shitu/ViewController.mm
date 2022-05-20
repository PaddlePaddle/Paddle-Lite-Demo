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
#include "pipeline.h"
#include "timer.h"
#include <arm_neon.h>
#include <iostream>
#include <mutex>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
#include <string>
#import <sys/timeb.h>
#include <vector>

using namespace paddle::lite_api;
using namespace cv;

std::mutex mtx;
Timer tic;
PipeLine *pp_shitu;
long long count = 0;

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
@property(nonatomic) int topk;
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
  _image = [UIImage imageNamed:@"third-party/assets/images/wu_ling.jpg"];
  if (_image != nil) {
    printf("test load image successed\n");
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
  self.topk = 1;
  std::string det_model_path =
      app_dir + "/models/mainbody_PPLCNet_x2_5_640_quant_v1.0_lite.nb";
  std::string rec_model_path =
      app_dir + "/models/general_PPLCNet_x2_5_quant_v1.0_lite.nb";
  std::string label_path = app_dir + "/labels/label.txt";
  std::string img_path = app_dir + "/images/wu_ling.jpg";
  std::vector<int> det_input_shape{1, 3, 640, 640};
  std::vector<int> rec_input_shape{1, 3, 224, 224};
  std::vector<cv::Mat> batch_imgs;
  std::vector<PPShiTu::ObjectResult> det_result;
  int cpu_num_threads = 1;
  int warm_up = 0;
  int repeats = 1;
  pp_shitu =
      new PipeLine(det_model_path, rec_model_path, label_path, det_input_shape,
                   rec_input_shape, cpu_num_threads, warm_up, repeats);

  cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
  if (!srcimg.data) {
    printf("%s\n", "image read failed!");
    exit(-1);
  }
  cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);
  batch_imgs.push_back(srcimg);
  cv::Mat out_img;
  auto res = pp_shitu->run(batch_imgs, det_result, &out_img, 1);
  std::ostringstream result;
  if (res.size() > 0)
    result << "inference time: " << res[0] << " ms\n\n";
  for (int i = 1; i < res.size(); i++)
    result << "classid " << i << ": " << res[i] << "\n";
  self.result.numberOfLines = 0;
  self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
  self.flag_init = true;
  self.imageView.image = MatToUIImage(out_img);
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
            cvtColor(image, self->_cvimg, cv::COLOR_RGBA2RGB);
          }
          std::vector<cv::Mat> batch_imgs;
          std::vector<PPShiTu::ObjectResult> det_result;
          batch_imgs.push_back(self->_cvimg);
          cv::Mat out_img;
          auto res = pp_shitu->run(batch_imgs, det_result, &out_img, 1);
          std::ostringstream result;
          if (res.size() > 0)
            result << "inference time: " << res[0] << " ms\n";
          for (int i = 1; i < res.size(); i++)
            result << "classid " << i << ": " << res[i] << "\n";
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
