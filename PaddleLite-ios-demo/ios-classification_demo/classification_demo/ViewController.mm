//
//  ViewController.m
//  seg_demo
//
//  Created by Li,Xiaoyang(SYS) on 2018/11/13.
//  Copyright © 2018年 Li,Xiaoyang(SYS). All rights reserved.
//

#import "ViewController.h"
#import <sys/timeb.h>
#import <opencv2/opencv.hpp>
#import <opencv2/highgui/ios.h>
#import <opencv2/highgui/cap_ios.h>
#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include "include/paddle_api.h"
#include "include/paddle_use_ops.h"
#include "include/paddle_use_kernels.h"
#include "timer.h"

using namespace paddle::lite_api;
using namespace cv;

std::mutex mtx;
std::shared_ptr<PaddlePredictor> net_mbv1;
Timer tic;
long long count = 0;

double tensor_mean(const Tensor& tin) {
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

void neon_mean_scale(const float* din, float* dout, int size, std::vector<float> mean, std::vector<float> scale) {
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);
    
    float* dout_c0 = dout;
    float* dout_c1 = dout + size;
    float* dout_c2 = dout + size * 2;
    
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
void fill_tensor_with_cvmat(const Mat& img_in, Tensor& tout, int width, int height, std::vector<float> mean, std::vector<float> scale) {
    if (img_in.channels() == 4) {
        cv::cvtColor(img_in, img_in, CV_RGBA2RGB);
    }
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    cv::Mat imgf;
    im.convertTo(imgf, CV_32FC3, 1 / 255.f);
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    float* dout = tout.mutable_data<float>();
    neon_mean_scale(dimg, dout, width * height, mean, scale);
}

std::string print_topk(const float *scores, const int size, const int topk, \
                       const std::vector<std::string> &labels) {
    
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(scores[i], i);
    }
    
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());
    std::string rst;
    // print topk and score
    for (int i = 0; i < topk; i++) {
        float score = vec[i].first;
        int index = vec[i].second;
        std::ostringstream buff1;
        buff1 << "\nscore: " << score << "\n";
        std::string str = "class: " + labels[index] + buff1.str();
        rst = rst + str;
    }
    return rst;
}

@interface ViewController () <CvVideoCameraDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UISwitch *flag_process;
@property (weak, nonatomic) IBOutlet UISwitch *flag_back_cam;
@property (weak, nonatomic) IBOutlet UILabel *result;
@property (nonatomic,strong) CvVideoCamera *videoCamera;
@property (nonatomic,strong) UIImage* image;
@property (nonatomic) bool flag_init;
@property (nonatomic) std::vector<float> scale;
@property (nonatomic) std::vector<float> mean;
@property (nonatomic) std::vector<std::string> labels;
@property (nonatomic) cv::Mat cvimg;
@property (nonatomic,strong) UIImage* ui_img_test;
-(NSArray *)listFileAtPath:(NSString *)path;
-(std::vector<std::string>) load_labels:(const std::string&)path;
@end

@implementation ViewController
@synthesize imageView;
- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    _flag_process.on = NO;
    _flag_back_cam.on = NO;
    _image = [UIImage imageNamed:@"cat.jpg"];
    if (_image != nil) {
        printf("load image successed\n");
        imageView.image = _image;
    } else {
        printf("load image failed\n");
    }
    self.ui_img_test =[UIImage imageNamed:@"face.jpg"];
    if (self.ui_img_test != nil) {
        printf("load test image successed\n");
    } else {
        printf("load image failed\n");
    }
    
    [_flag_process addTarget:self action:@selector(PSwitchValueChanged:) forControlEvents:UIControlEventValueChanged];
    [_flag_back_cam addTarget:self action:@selector(CSwitchValueChanged:) forControlEvents:UIControlEventValueChanged];
    
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.imageView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.rotateVideo = 90;
    self.videoCamera.defaultFPS = 30;
    [self.view insertSubview:self.imageView atIndex:0];
    self.cvimg.create(640, 480, CV_8UC3);
    NSString *path = [[NSBundle mainBundle] bundlePath];
    std::string paddle_mobilenetv1_dir = std::string([path UTF8String]);
    MobileConfig config;
    config.set_model_from_file(paddle_mobilenetv1_dir + "/model.nb");
    net_mbv1 = CreatePaddlePredictor<MobileConfig>(config);
    self.mean = {0.485f, 0.456f, 0.406f};
    self.scale = {0.229f, 0.224f, 0.225f};
    NSString *label_file = [NSBundle.mainBundle pathForResource:@"labels" ofType:@"txt"];
    std::string label_file_str = std::string([label_file UTF8String]);
    self.labels = [self load_labels:label_file_str];
    cv::Mat img_cat;
    UIImageToMat(self.image, img_cat);
    std::unique_ptr<Tensor> input_tensor(net_mbv1->GetInput(0));
    input_tensor->Resize({1, 3, 224, 224});
    input_tensor->mutable_data<float>();
    cv::Mat img;
    if (img_cat.channels() == 4) {
        cv::cvtColor(img_cat, img, CV_RGBA2RGB);
    } else {
        img = img_cat;
    }
    for (int i = 0; i < 5; i++) {
        net_mbv1->Run();
    }
    fill_tensor_with_cvmat(img, *(input_tensor.get()), 224, 224, self.mean, self.scale);
    tic.start();
    net_mbv1->Run();
    tic.end();
    std::unique_ptr<const Tensor> output_tensor(net_mbv1->GetOutput(0));
    auto ptr = output_tensor->mutable_data<float>();
    std::string out_class = print_topk(ptr, 1000, 1, self.labels);
    std::ostringstream result;
    result << out_class << "\ntime: " << tic.get_average_ms() << " ms";
    self.result.numberOfLines = 0;
    self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
    self.flag_init = true;
}

-(std::vector<std::string>) load_labels:(const std::string&)path {
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
            for (int i = 0; i < str_s.length(); i++) {
                if (str_s[i] == ' ') {
                    std::string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    fclose(fp);
    return labels;
}

-(NSArray *)listFileAtPath:(NSString *)path {
    //-----> LIST ALL FILES <-----//
    NSLog(@"LISTING ALL FILES FOUND");
    
    int count;
    
    NSArray *directoryContent = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:path error:NULL];
    for (count = 0; count < (int)[directoryContent count]; count++) {
        NSLog(@"File %d: %@/%@", (count + 1), path, [directoryContent objectAtIndex:count]);
    }
    return directoryContent;
}

- (void)PSwitchValueChanged:(UISwitch *) sender {
    NSLog(@"%@", sender.isOn ? @"process ON" : @"process OFF");
    if (sender.isOn) {
        [self.videoCamera start];
    } else {
        [self.videoCamera stop];
    }
}

- (void)CSwitchValueChanged:(UISwitch *) sender {
    NSLog(@"%@", sender.isOn ? @"back ON" : @"back OFF");
    if (sender.isOn) {
        if (self.flag_process.isOn) {
            [self.videoCamera stop];
        }
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
        if (self.flag_process.isOn) {
            [self.videoCamera start];
        }
    } else {
        if (self.flag_process.isOn) {
            [self.videoCamera stop];
        }
        self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
        if (self.flag_process.isOn) {
            [self.videoCamera start];
        }
    }
}

- (void)processImage:(cv::Mat &)image {
    
    dispatch_async(dispatch_get_main_queue(), ^{
        if (self.flag_process.isOn) {
            if (self.flag_init) {
                count++;
                if (image.channels() == 4) {
                    cvtColor(image, self->_cvimg, CV_RGBA2RGB);
                }
                std::unique_ptr<Tensor> input_tensor(net_mbv1->GetInput(0));
                input_tensor->Resize({1, 3, 224, 224});
                fill_tensor_with_cvmat(self->_cvimg, *(input_tensor.get()), 224, 224, self.mean, self.scale);
                tic.start();
                net_mbv1->Run();
                tic.end();
                if (count > 10) {
                    count=0;
                    std::unique_ptr<const Tensor> output_tensor(net_mbv1->GetOutput(0));
                    auto ptr = output_tensor->mutable_data<float>();
                    std::string out_class = print_topk(ptr, 1000, 1, self.labels);
                    std::ostringstream result;
                    result << out_class << "\ntime: " << tic.get_average_ms() << " ms";
                    self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
                    tic.clear();
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
