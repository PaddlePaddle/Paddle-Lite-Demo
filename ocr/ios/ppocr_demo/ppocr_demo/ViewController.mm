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
// clang-format off
#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
// clang-format on
#import "ViewController.h"
#include "pipeline.h"
#include "timer.h"
#include <arm_neon.h>
#include <iostream>
#include <mutex>
#include <paddle_api.h>
#include <paddle_use_kernels.h>
#include <paddle_use_ops.h>
#include <string>
#import <sys/timeb.h>
#include <vector>

using namespace paddle::lite_api;
using namespace cv;

std::mutex mtx;
Pipeline *pipe_;
Timer tic;
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
@property(nonatomic) std::string dict_path;
@property(nonatomic) std::string config_path;
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
  _image = [UIImage imageNamed:@"test.jpg"];
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
  std::string paddle_dir = std::string([path UTF8String]);
  std::string det_model_file =
      paddle_dir + "/ch_ppocr_mobile_v2.0_det_slim_opt.nb";
  std::string rec_model_file =
      paddle_dir + "/ch_ppocr_mobile_v2.0_rec_slim_opt.nb";
  std::string cls_model_file =
      paddle_dir + "/ch_ppocr_mobile_v2.0_cls_slim_opt.nb";
  std::string img_path = paddle_dir + "/test.jpg";
  std::string output_img_path = paddle_dir + "/test_result.jpg";
  self.dict_path = paddle_dir + "/ppocr_keys_v1.txt";
  self.config_path = paddle_dir + "/config.txt";

  tic.start();
  cv::Mat srcimg = imread(img_path);
  pipe_ = new Pipeline(det_model_file, cls_model_file, rec_model_file,
                       "LITE_POWER_HIGH", 1, self.config_path, self.dict_path);
  std::ostringstream result;
  std::vector<std::string> res_txt;
  cv::Mat img_vis = pipe_->Process(srcimg, output_img_path, res_txt);

  tic.end();
  // print result
  //    for (int i = 0; i < res_txt.size() / 2; i++) {
  //        result << i << "\t" << res_txt[2*i] << "\t" << res_txt[2*i + 1] <<
  //        "\n";
  //    }

  result << "花费了" << tic.get_average_ms() << " ms\n";

  self.result.numberOfLines = 0;
  self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
  self.flag_init = true;
  self.imageView.image = MatToUIImage(img_vis);
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
            cvtColor(image, self->_cvimg, COLOR_RGBA2RGB);
          }

          tic.start();

          std::vector<std::string> res_txt;
          cv::Mat img_vis =
              pipe_->Process(self->_cvimg, "output_img_result.jpg", res_txt);

          tic.end();
          // print recognized text
          std::ostringstream result;
          // print result
          //    for (int i = 0; i < res_txt.size() / 2; i++) {
          //        result << i << "\t" << res_txt[2*i] << "\t" <<
          //        res_txt[2*i+1] << "\n";
          //    }

          result << "花费了" << tic.get_average_ms() << " ms\n";

          cvtColor(img_vis, self->_cvimg, COLOR_RGB2BGR);
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
