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

#include "FaceProcess.h"

std::vector<unsigned char>  bilinear_insert(cv::Mat image, double new_x, double new_y){
    int x1 = (int)new_x;
    int y1 = (int)new_y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    int channels = image.channels();
    std::vector<unsigned char> rst;
    auto data = image.data;
    int cols = image.cols;
    float a1 = float(x2) - new_x;
    float a2 = new_x - float(x1);
    float b1 = float(y2) - new_y;
    float b2 = new_y - float(y1);
    for (int i = 0; i < channels; i++){
        float aa1 = float(data[(x1 * cols+ y1) * channels + i]) * a1 * b1;
        float aa2 = float(data[(x2 * cols + y1) * channels + i]) * a2 * b1;
        float aa3 = float(data[(x1 * cols + y2) * channels + i]) * a1 * b2;
        float aa4 = float(data[(x2 * cols  + y2) * channels + i]) * a2 *b2;
        float res = aa1 + aa2 + aa3 + aa4;
        rst.push_back(res);
    }
    return rst;
}

cv::Mat local_traslation_warp(cv::Mat image, cv::Point2d start_point,
                              cv::Point2d end_point, double radius){
    // 局部平移算法
    double radius_square = pow(radius, 2);
    cv::Mat image_cp;
    image.copyTo(image_cp);
    double dist_se = (start_point.x - end_point.x) * (start_point.x - end_point.x) +
                     (start_point.y - end_point.y) * (start_point.y - end_point.y);
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            // 计算该点是否在形变圆的范围之内
            // 优化，第一步，直接判断是会在（start_point[0], start_point[1])的矩阵框中
            if (abs(i - start_point.x > radius) && abs(j - start_point.y > radius)){
                continue;
            }
            double distance = (i - start_point.x) * (i - start_point.x) +
                              (j - start_point.y) * (j - start_point.y);
            if (distance < radius_square) {
                // 计算出（i,j）坐标的原坐标
                // 计算公式中右边平方号里的部分
                double ratio = (radius_square - distance) / (radius_square - distance + dist_se);
                ratio *= ratio;

                // 映射原位置
                double new_x = i - ratio * (end_point.x - start_point.x);
                double new_y = j - ratio * (end_point.y - start_point.y);

                // 根据双线性插值法得到new_x, new_y的值
                std::vector<unsigned char> new_val = bilinear_insert(image, new_x, new_y);
//                image_cp.at(j, i) = new_val;
                for (int k = 0; k < new_val.size(); k++)
                    image_cp.data[(i * cols+ j) * channels + k] = new_val[k];
            }
        }
    }
    return image_cp;
}

cv::Mat thin_face(cv::Mat image, std::vector<cv::Point2d> points){
    cv::Point2d end_point = points[30];
    // 瘦左脸，3号点到5号点的距离作为瘦脸距离
    double dist_left = (points[3].x - points[5].x) * (points[3].x - points[5].x) +
                       (points[3].y - points[5].y) * (points[3].x - points[5].y);
    dist_left = sqrt(dist_left);
    cv::Mat image_left = local_traslation_warp(image, points[3], end_point, dist_left);

    // 瘦右脸，13号点到15号点的距离作为瘦脸距离
    double dist_right = (points[13].x - points[15].x) * (points[13].x - points[15].x) +
                        (points[1].y - points[15].y) * (points[13].x - points[15].y);
    dist_right = sqrt(dist_right);
    cv::Mat image_right = local_traslation_warp(image_left, points[13], end_point, dist_right);
    return image_right;
}

void local_zoom_warp(cv::Mat image, cv::Point2d point, int radius, int strength){
    // 图像局部缩放算法
    int height = image.rows;
    int width = image.cols;
    int left = point.x - radius;
    int top = point.y - radius;
    int right = point.x + radius;
    int bottom = point.y + radius;
    left = left > 0 ? left : 0;
    top = top > 0 ? top : 0;
    right = right < width ? right : width - 1;
    bottom = bottom < height ? bottom : height - 1;
    int channels = image.channels();
    int cols = image.cols;

    double radius_square = pow(radius, 2);
    for (int y = top; y <= bottom; y++){
        int offset_y = y - point.y;
        for (int x = left; x <= right; x++){
            int offset_x = x - point.x;
            double dist_xy = offset_x * offset_x + offset_y * offset_y;
            if (dist_xy < radius_square){
                double scale = 1 - dist_xy / radius_square;
                scale = 1 - strength / 100 * scale;
                double new_x = offset_x * scale + point.x;
                double new_y = offset_y * scale + point.y;
                new_x = new_x > 0 ? (new_x < height ? new_x : height - 1) : 0;
                new_y = new_y > 0 ? (new_y < width ? new_y : width - 1) : 0;
                std::vector<unsigned char> new_val = bilinear_insert(image, new_x, new_y);
                for (int k = 0; k < new_val.size(); k++)
                    image.data[(x * cols+ y) * channels + k] = new_val[k];
            }
        }
    }
}

cv::Mat enlarge_eyes(cv::Mat image, std::vector<cv::Point2d> points, int radius, int strength) {
    // 放大眼睛
    // image： 人像图片 face_landmark: 人脸关键点 radius: 眼睛放大范围半径 strength：眼睛放大程度
    cv::Mat image_cp;
    image.copyTo(image_cp);
    // 以左眼最低点和最高点之间的中点为圆心
    cv::Point2d left_eye_top = points[37];
    cv::Point2d left_eye_bottom = points[41];
    cv::Point2d left_eye_center = (left_eye_top + left_eye_bottom) / 2;
    //以右眼最低点和最高点之间的中点为圆心
    cv::Point2d right_eye_top = points[43];
    cv::Point2d right_eye_bottom = points[47];
    cv::Point2d right_eye_center = (right_eye_top + right_eye_bottom) / 2;

    // 放大双眼
    local_zoom_warp(image_cp, left_eye_center, radius, strength);
    local_zoom_warp(image_cp, right_eye_center, radius, strength);

    return image_cp;
}

cv::Mat whitening(cv::Mat image){
    // 美白
    cv::Mat dst;
    int v1 = 5; // 磨皮程度
    int v2 = 2; // 细节程度
    cv::Mat tmp;
    image.copyTo(tmp);

    cv::bilateralFilter(image, tmp, v1 * 5, v1 * 12.5, v1 * 12.5);
    cv::subtract(tmp, image, tmp);
    cv::GaussianBlur(tmp, tmp, cv::Size(2*v2-1, 2*v2-1), 0);
    cv::add(image, tmp, tmp);
    cv::addWeighted(image, 0.1, tmp, 0.9, 0.0, dst);
    return dst;
}