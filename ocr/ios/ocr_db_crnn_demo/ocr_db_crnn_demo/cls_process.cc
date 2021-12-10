//
// Created by chenjiao04 on 2021/12/4.
//

#include "cls_process.h"
#include "timer.h"

const std::vector<int> cls_image_shape{3, 48, 192};
cv::Mat ClsResizeImg(cv::Mat img) {
  int imgC, imgH, imgW;
  imgC = cls_image_shape[0];
  imgH = cls_image_shape[1];
  imgW = cls_image_shape[2];

  float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);

  int resize_w, resize_h;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
             cv::INTER_LINEAR);
  if (resize_w < imgW) {
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, imgW - resize_w,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  }
  return resize_img;
}

ClsPredictor::ClsPredictor(const std::string &modelDir, const int cpuThreadNum,
                           const std::string &cpuPowerMode) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir);
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
}

void ClsPredictor::Preprocess(const cv::Mat &img) {
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    cv::Mat crop_img;
    img.copyTo(crop_img);
    cv::Mat resize_img;

    int index = 0;
    float wh_ratio =
        static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

    resize_img = ClsResizeImg(crop_img);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor0(std::move(predictor_->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();
    NHWC3ToNC3HW(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
}

cv::Mat ClsPredictor::Postprocess(const cv::Mat &srcimg, const float thresh) {
  // Get output and run postprocess
    std::unique_ptr<const Tensor> softmax_out(
        std::move(predictor_->GetOutput(0)));
    auto *softmax_scores = softmax_out->mutable_data<float>();
    auto softmax_out_shape = softmax_out->shape();
    float score = 0;
    int label = 0;
    for (int i = 0; i < softmax_out_shape[1]; i++) {
      if (softmax_scores[i] > score) {
        score = softmax_scores[i];
        label = i;
      }
    }
    if (label % 2 == 1 && score > thresh) {
      cv::rotate(srcimg, srcimg, 1);
    }
    return srcimg;
}

cv::Mat ClsPredictor::Predict(const cv::Mat &img, double *preprocessTime, double *predictTime, double *postprocessTime, const float thresh) {
  cv::Mat src_img;
  img.copyTo(src_img);
//  Timer tic;
//  tic.start();
  Preprocess(img);
// tic.end();
// *preprocessTime = tic.get_average_ms();
// std::cout << "cls predictor preprocess costs" <<  *preprocessTime;

// tic.start();
  predictor_->Run();
// tic.end();
// *predictTime = tic.get_average_ms();
// std::cout << "cls predictor predict costs" <<  *predictTime;

//  tic.start();
  cv::Mat srcimg = Postprocess(src_img, thresh);
// tic.end();
// *postprocessTime = tic.get_average_ms();
// std::cout << "cls predictor predict costs" <<  *postprocessTime;
  return srcimg;
}
