// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "Detector_Kpts.h"
#include "postprocess.h"
#include <set>

// KeyPoint Detector

Detector_KeyPoint::Detector_KeyPoint(
    const std::string &modelDir, const int accelerate_opencl,
    const int cpuThreadNum, const std::string &cpuPowerMode, int inputWidth,
    int inputHeight, const std::vector<float> &inputMean,
    const std::vector<float> &inputStd, float scoreThreshold)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
      inputStd_(inputStd), scoreThreshold_(scoreThreshold) {
  paddle::lite_api::MobileConfig config;
  std::string dir = modelDir + "models/tinypose/";
  // if (accelerate_opencl && paddle::lite_api::IsOpenCLBackendValid()) {
  if (0) { // TODO: only use cpu model, before debug of opencl inference
    config.set_model_from_file(dir + "tinypose_256x192_opencl.nb");
    const std::string bin_name = "pose_opencl_kernel.bin";
    config.set_opencl_binary_path_name(dir, bin_name);
    const std::string tuned_name = "pose_opencl_tuned.bin";
    config.set_opencl_tune(paddle::lite_api::CL_TUNE_NORMAL, dir, tuned_name);
    config.set_opencl_precision(paddle::lite_api::CL_PRECISION_FP16);
  } else {
    config.set_model_from_file(dir + "tinypose_256x192_arm.nb");
  }

  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_keypoint_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);

  colorMap_ = GenerateColorMap(20); // coco keypoint number is 17
}

std::vector<cv::Scalar> Detector_KeyPoint::GenerateColorMap(int numOfClasses) {
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

void Detector_KeyPoint::Preprocess(std::vector<cv::Mat> &bs_images) {
  // Set the data of input image
  int batchsize = bs_images.size();
  auto inputTensor = predictor_keypoint_->GetInput(0);
  std::vector<int64_t> inputShape = {batchsize, 3, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  for (int i = 0; i < batchsize; i++) {
    cv::Mat resizedRGBAImage;
    cv::resize(bs_images[i], resizedRGBAImage,
               cv::Size(inputShape[3], inputShape[2]));
    cv::Mat resizedRGBImage;
    cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);

    resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0f);
    auto dst_inptr = inputData + i * (3 * inputHeight_ * inputWidth_);
    Permute(&resizedRGBImage, dst_inptr);
//    NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data),
//                 dst_inptr, inputMean_.data(), inputStd_.data(), inputShape[3],
//                 inputShape[2]);
  }
}

void Detector_KeyPoint::Postprocess(std::vector<RESULT_KEYPOINT> *results,
                                    std::vector<std::vector<float>> &center_bs,
                                    std::vector<std::vector<float>> &scale_bs) {
  auto outputTensor = predictor_keypoint_->GetOutput(0);
  auto outputdata = outputTensor->data<float>();
  auto output_shape = outputTensor->shape();
  int outputSize = ShapeProduction(output_shape);
  auto outidx_Tensor = predictor_keypoint_->GetOutput(1);
  auto idxoutdata = outidx_Tensor->data<int64_t>();
  auto idx_shape = outidx_Tensor->shape();
  int idxoutSize = ShapeProduction(idx_shape);

  std::vector<float> output;
  std::vector<int64_t> idxout;
  output.resize(outputSize);
  std::copy_n(outputdata, outputSize, output.data());
  idxout.resize(idxoutSize);
  std::copy_n(idxoutdata, idxoutSize, idxout.data());

  std::vector<float> preds(output_shape[1] * 3, 0);

  for (int batchid = 0; batchid < output_shape[0]; batchid++) {
    get_final_preds(output, output_shape, idxout, idx_shape, center_bs[batchid],
                    scale_bs[batchid], preds, batchid, this->use_dark);
    RESULT_KEYPOINT result_item;
    result_item.num_joints = output_shape[1];
    result_item.keypoints.clear();
    for (int i = 0; i < output_shape[1]; i++) {
      result_item.keypoints.emplace_back(preds[i * 3]);
      result_item.keypoints.emplace_back(preds[i * 3 + 1]);
      result_item.keypoints.emplace_back(preds[i * 3 + 2]);
    }
    results->push_back(result_item);
  }
}

void Detector_KeyPoint::Predict(const cv::Mat &rgbaImage,
                                std::vector<RESULT> *results,
                                std::vector<RESULT_KEYPOINT> *results_kpts,
                                double *preprocessTime, double *predictTime,
                                double *postprocessTime, bool single) {
  if (results->empty())
    return;
  auto t = GetCurrentTime();
  std::vector<std::vector<float>> center_bs;
  std::vector<std::vector<float>> scale_bs;
  std::vector<cv::Mat> cropimgs;
  std::vector<RESULT> sresult;
  FindMaxRect(results, sresult, single);
  CropImg(rgbaImage, cropimgs, sresult, center_bs, scale_bs);
  t = GetCurrentTime();
  Preprocess(cropimgs);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Detector_KeyPoint preprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_keypoint_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Detector_KeyPoint predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(results_kpts, center_bs, scale_bs);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Detector_KeyPoint postprocess costs %f ms", *postprocessTime);
}

void Detector_KeyPoint::FindMaxRect(std::vector<RESULT> *results, std::vector<RESULT> &rect_buff, bool single) {
  int findnum = 2;
  if (single) {
    findnum = 1;
  }
  std::set<int> findset;
  for (int i=0; i<findnum; i++) {
    int maxid = 0;
    for (int i = 0; i < results->size(); i++) {
      if (findset.count(i) > 0) {
        if (i == 0) {
          maxid = 1;
        }
        continue;
      }
      if ((*results)[i].h + (*results)[i].w >
          (*results)[maxid].h + (*results)[maxid].w) {
        maxid = i;
      }
    }

    (*results)[maxid].fill_color = colorMap_[1];
    rect_buff.emplace_back((*results)[maxid]);
    findset.insert(maxid);
  }
}

void Detector_KeyPoint::CropImg(const cv::Mat &img,
                                std::vector<cv::Mat> &crop_img,
                                std::vector<RESULT> &results,
                                std::vector<std::vector<float>> &center_bs,
                                std::vector<std::vector<float>> &scale_bs,
                                float expandratio) {
  int w = img.cols;
  int h = img.rows;
  for (int i = 0; i < std::min(int(results.size()), 2); i++) {
    auto area = results[i];
    int crop_x1 = std::max(0, static_cast<int>(area.x * w));
    int crop_y1 = std::max(0, static_cast<int>(area.y * h));
    int crop_x2 =
        std::min(img.cols - 1, static_cast<int>((area.x + area.w) * w));
    int crop_y2 =
        std::min(img.rows - 1, static_cast<int>((area.y + area.h) * h));
    int center_x = (crop_x1 + crop_x2) / 2.;
    int center_y = (crop_y1 + crop_y2) / 2.;
    int half_h = (crop_y2 - crop_y1) / 2.;
    int half_w = (crop_x2 - crop_x1) / 2.;

    // adjust h or w to keep image ratio, expand the shorter edge
    if (half_h * 3 > half_w * 4) {
      half_w = static_cast<int>(half_h * 0.75);
    } else {
      half_h = static_cast<int>(half_w * 4 / 3);
    }

    int roi_x1 = center_x - static_cast<int>(half_w * (1 + expandratio));
    int roi_y1 = center_y - static_cast<int>(half_h * (1 + expandratio));
    int roi_x2 = center_x + static_cast<int>(half_w * (1 + expandratio));
    int roi_y2 = center_y + static_cast<int>(half_h * (1 + expandratio));
    crop_x1 = std::max(0, roi_x1);
    crop_y1 = std::max(0, roi_y1);
    crop_x2 = std::min(img.cols - 1, roi_x2);
    crop_y2 = std::min(img.rows - 1, roi_y2);
    cv::Mat src_img =
        img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));

    cv::Mat dst_img;
    cv::copyMakeBorder(src_img, dst_img, std::max(0, -roi_y1),
                       std::max(0, roi_y2 - img.rows + 1), std::max(0, -roi_x1),
                       std::max(0, roi_x2 - img.cols + 1),
                       cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    crop_img.emplace_back(dst_img);

    std::vector<float> center, scale;
    center.emplace_back(center_x);
    center.emplace_back(center_y);
    scale.emplace_back((half_w * 2) * (1 + expandratio));
    scale.emplace_back((half_h * 2) * (1 + expandratio));

    center_bs.emplace_back(center);
    scale_bs.emplace_back(scale);
  }
}
