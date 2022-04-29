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

struct Face {
  // Face detection result: face rectangle
  cv::Rect roi;
  // Face keypoints detection result: keypoint coordiate
  std::vector<cv::Point2d> keypoints;
  // Score
  float score;
};

std::mutex mtx;
std::shared_ptr<PaddlePredictor> face_detection_predictor;
std::shared_ptr<PaddlePredictor> face_keypoints_detection_predictor;
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
void FaceDetector_Preprocess(const Mat &img, int height, int width, std::vector<float> mean,
                std::vector<float> scale, bool is_scale) {
  std::unique_ptr<Tensor> input_tensor(face_detection_predictor->GetInput(0));
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

void FaceDetector_Postprocess(const cv::Mat &rgbImage, std::vector<Face> *faces,
                              float scoreThreshold_) {
  int imageWidth = rgbImage.cols;
  int imageHeight = rgbImage.rows;
  // Get output tensor
  auto outputTensor = face_detection_predictor->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);

  auto outputTensor1 = face_detection_predictor->GetOutput(1);
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

void FaceKeypointsDetector_Preprocess(const cv::Mat &rgbImage,
    const std::vector<Face> &faces, std::vector<cv::Rect> *adjustedFaceROIs,
    int height, int width) {
  // Prepare input tensor
  auto inputTensor = face_keypoints_detection_predictor->GetInput(0);
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

void FaceKeypointsDetector_Postprocess(const std::vector<cv::Rect> &adjustedFaceROIs, std::vector<Face> *faces,
    cv::Mat &img) {
  auto outputTensor = face_keypoints_detection_predictor->GetOutput(0);
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
  self.mean = {0.407843f, 0.694118f, 0.482353f};
  self.scale = {0.5f, 0.5f, 0.5f};
  self.input_height = 480;
  self.input_width = 640;
    
  MobileConfig face_detection_config;
  face_detection_config.set_model_from_file(app_dir + "/models/face_detector_for_cpu/model.nb");
  face_detection_predictor = CreatePaddlePredictor<MobileConfig>(face_detection_config);
    
  MobileConfig face_keypoints_detection_config;
  face_keypoints_detection_config.set_model_from_file(app_dir + "/models/facekeypoints_detector_for_cpu/model.nb");
  face_keypoints_detection_predictor = CreatePaddlePredictor<MobileConfig>(face_keypoints_detection_config);
    
  cv::Mat img_face;
  UIImageToMat(self.image, img_face);
  cv::Mat img;
  if (img_face.channels() == 4) {
    cv::cvtColor(img_face, img, CV_RGBA2RGB);
  }
    
  imgWidth = img.cols;
  imgHeight = img.rows;
  std::vector<Face> faces;
  std::vector<cv::Rect> adjustedFaceROIs;
  float scoreThreshold = 0.7;
  std::ostringstream result;
    
  FaceDetector_Preprocess(img, self.input_height, self.input_width, self.mean, self.scale,
             true);
  tic.start();
  face_detection_predictor->Run();
  tic.end();
  FaceDetector_Postprocess(img, &faces, scoreThreshold);
    
  float face_detection_time = tic.get_average_ms();
  float face_keypoints_detection_time = 0.f;

  if (faces.size() > 0) {
      adjustedFaceROIs.resize(faces.size());
      FaceKeypointsDetector_Preprocess(img, faces,
                                       &adjustedFaceROIs, 60, 60);
      tic.start();
      face_keypoints_detection_predictor->Run();
      tic.end();
      FaceKeypointsDetector_Postprocess(adjustedFaceROIs, &faces, img);
      
      face_keypoints_detection_time = tic.get_average_ms();
  }
 
  result << "face_detection_time: " << face_detection_time << " ms" << "\nface_keypoints_detection_time: " << face_keypoints_detection_time << " ms" << "\ntotal_time: " << face_detection_time + face_keypoints_detection_time << " ms";
  self.result.numberOfLines = 0;
  self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
  self.flag_init = true;
    cv::Mat outputImage = img;
    for (int i = 0; i < faces.size(); i++) {
      // Configure color
      for (int j = 0; j < faces[i].keypoints.size(); j++) {
        cv::circle(outputImage, faces[i].keypoints[j], 1, cv::Scalar(0, 255, 0),
                   2); //在图像中画出特征点，1是圆的半径
      }
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
          std::vector<Face> faces;
          std::vector<cv::Rect> adjustedFaceROIs;
          float scoreThreshold = 0.7;
            
          FaceDetector_Preprocess(self->_cvimg, self.input_height, self.input_width,
                     self.mean, self.scale, true);
          tic.start();
          face_detection_predictor->Run();
          tic.end();
          FaceDetector_Postprocess(self->_cvimg, &faces, scoreThreshold);
              
          float face_detection_time = tic.get_average_ms();
          float face_keypoints_detection_time = 0.f;

          if (faces.size() > 0) {
              adjustedFaceROIs.resize(faces.size());
              FaceKeypointsDetector_Preprocess(self->_cvimg, faces,
                                                 &adjustedFaceROIs, 60, 60);
              tic.start();
              face_keypoints_detection_predictor->Run();
              tic.end();
              FaceKeypointsDetector_Postprocess(adjustedFaceROIs, &faces, self->_cvimg);

              face_keypoints_detection_time = tic.get_average_ms();
          }
            
          
            std::ostringstream result;
            result << "face_detection_time: " << face_detection_time << " ms" << "\nface_keypoints_detection_time: " << face_keypoints_detection_time << " ms" << "\ntotal_time: " << face_detection_time + face_keypoints_detection_time << " ms";
            self.result.numberOfLines = 0;
            self.result.text = [NSString stringWithUTF8String:result.str().c_str()];
            self.flag_init = true;
              cv::Mat outputImage = self->_cvimg;
            cvtColor(outputImage, outputImage, CV_RGB2BGR);
            for (int i = 0; i < faces.size(); i++) {
              // Configure color
              for (int j = 0; j < faces[i].keypoints.size(); j++) {
                cv::circle(outputImage, faces[i].keypoints[j], 1, cv::Scalar(0, 255, 0),
                           2); //在图像中画出特征点，1是圆的半径
              }
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
