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

struct Object{
    int batch_id;
    cv::Rect rec;
    int class_id;
    float prob;
};

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
void fill_tensor_with_cvmat(const Mat& img_in, Tensor& tout, int width, int height,
                            std::vector<float> mean, std::vector<float> scale, bool is_scale) {
    if (img_in.channels() == 4) {
        cv::cvtColor(img_in, img_in, CV_RGBA2RGB);
    }
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    cv::Mat imgf;
    float scale_factor = is_scale? 1 / 255.f : 1.f;
    im.convertTo(imgf, CV_32FC3, scale_factor);
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    float* dout = tout.mutable_data<float>();
    neon_mean_scale(dimg, dout, width * height, mean, scale);
}

const char* class_names[] = {
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
};

std::vector<Object> detect_object(const float* data,
                                    int count,
                                    const std::vector<std::vector<uint64_t>>& lod,
                                    const float thresh,
                                    Mat& image) {
    std::vector<Object> rect_out;
    const float* dout = data;
    for (int iw = 0; iw < count; iw++) {
        int oriw = image.cols;
        int orih = image.rows;
        if (dout[1] > thresh && static_cast<int>(dout[0]) > 0) {
            Object obj;
            int x = static_cast<int>(dout[2] * oriw);
            int y = static_cast<int>(dout[3] * orih);
            int w = static_cast<int>(dout[4] * oriw) - x;
            int h = static_cast<int>(dout[5] * orih) - y;
            cv::Rect rec_clip = cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
            obj.batch_id = 0;
            obj.class_id = static_cast<int>(dout[0]);
            obj.prob = dout[1];
            obj.rec = rec_clip;
            if (w > 0 && h > 0 && obj.prob <= 1) {
                rect_out.push_back(obj);
                cv::rectangle(image, rec_clip, cv::Scalar(255, 0, 0));
                std::cout << "detection, image size: " << image.cols
                << ", " << image.rows << ", detect object: "
                << class_names[obj.class_id] << ", score: "
                << obj.prob << ", location: x=" << x << ", y="
                << y << ", width=" << w << ", height=" << h << std::endl;
            }
        }
        dout += 6;
    }
    return rect_out;
}

@interface ViewController () <CvVideoCameraDelegate>
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UISwitch *flag_process;
@property (weak, nonatomic) IBOutlet UISwitch *flag_video;
@property (weak, nonatomic) IBOutlet UIImageView *preView;
@property (weak, nonatomic) IBOutlet UISwitch *flag_back_cam;
@property (weak, nonatomic) IBOutlet UILabel *result;
@property (nonatomic,strong) CvVideoCamera *videoCamera;
@property (nonatomic,strong) UIImage* image;
@property (nonatomic) bool flag_init;
@property (nonatomic) bool flag_cap_photo;
@property (nonatomic) std::vector<float> scale;
@property (nonatomic) std::vector<float> mean;
@property (nonatomic) std::vector<std::string> labels;
@property (nonatomic) cv::Mat cvimg;
@property (nonatomic,strong) UIImage* ui_img_test;
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
    
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.preView];
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
    self.mean = {0.5f, 0.5f, 0.5f};
    self.scale = {0.5f, 0.5f, 0.5f};
    cv::Mat img_cat;
    UIImageToMat(self.image, img_cat);
    std::unique_ptr<Tensor> input_tensor(net_mbv1->GetInput(0));
    input_tensor->Resize({1, 3, 300, 300});
    input_tensor->mutable_data<float>();
    cv::Mat img;
    if (img_cat.channels() == 4) {
        cv::cvtColor(img_cat, img, CV_RGBA2RGB);
    }
    fill_tensor_with_cvmat(img, *(input_tensor.get()), 300, 300, self.mean, self.scale, true);
    tic.start();
    net_mbv1->Run();
    tic.end();
    std::unique_ptr<const Tensor> output_tensor(net_mbv1->GetOutput(0));
    auto ptr = output_tensor->mutable_data<float>();
    auto shape_out = output_tensor->shape();
    int64_t cnt = 1;
    for (auto& i : shape_out) {
        cnt *= i;
    }
    auto rec_out = detect_object(ptr, static_cast<int>(cnt / 6), output_tensor->lod(), 0.6f, img);
    std::ostringstream result;
    for (auto& obj : rec_out) {
        result << "class: " << class_names[obj.class_id] << ", score: " << obj.prob << "\n";
    }
    self.result.numberOfLines = 0;
    self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
    self.flag_init = true;
    self.imageView.image = MatToUIImage(img);
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
                if (self.flag_video.isOn || self.flag_cap_photo) {
                    self.flag_cap_photo = false;
                    if (image.channels() == 4) {
                        cvtColor(image, self->_cvimg, CV_RGBA2RGB);
                    }
                    std::unique_ptr<Tensor> input_tensor(net_mbv1->GetInput(0));
                    input_tensor->Resize({1, 3, 300, 300});
                    fill_tensor_with_cvmat(self->_cvimg, *(input_tensor.get()), 300, 300, self.mean, self.scale, true);
                    tic.start();
                    net_mbv1->Run();
                    tic.end();
                    std::unique_ptr<const Tensor> output_tensor(net_mbv1->GetOutput(0));
                    auto ptr = output_tensor->mutable_data<float>();
                    auto shape_out = output_tensor->shape();
                    int64_t cnt = 1;
                    for (auto& i : shape_out) {
                        cnt *= i;
                    }
                    std::cout << "cnt: " << cnt << std::endl;
                    auto rec_out = detect_object(ptr, static_cast<int>(cnt / 6),
                                                 output_tensor->lod(), 0.6f, self->_cvimg);
                    std::ostringstream result;
                    for (auto& obj : rec_out) {
                        result << "class: " << class_names[obj.class_id] << ", score: " << obj.prob << "\n";
                    }
                    cvtColor(self->_cvimg, self->_cvimg, CV_RGB2BGR);
                    self.result.numberOfLines = 0;
                    self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
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
