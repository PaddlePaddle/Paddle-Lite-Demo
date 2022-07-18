from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from numpy.lib.shape_base import column_stack

from paddlelite.lite import *
import numpy as np
import cv2
import time

# image_path是图片的路径，单张图片路径为xxx.png/jpg,多张图片路径为xxx.txt
model_dir  = "../../../assets/models/PPLCNet/model.nb"
image_path = '../../../assets/images/test_list.txt'
labels_path= '../../../assets/labels/label.txt'
input_shape= [1,3,224,224]
thread_num = 4
warmup_num = 10
repeat_num = 100
topk = 3

# 加载标签集labels
def load_labels():
    labels = [] 
    fp_r = open(labels_path, 'r')
    data = fp_r.readline()[:-1]
    while data:
        labels.append(data)
        data = fp_r.readline()[:-1]
    # print(labels)
    fp_r.close()
    return labels

# (1) 设置配置信息并创建预测器
config = MobileConfig()
config.set_model_from_file(model_dir)
predictor = create_paddle_predictor(config)
labels = load_labels()

# (2) 从图片读入数据 image_data
def RunModel():
    # 将任意分辨率的图像数据缩放为224*224
    resize_img = cv2.resize(img, [input_shape[2], input_shape[3]]
                            ,interpolation=cv2.INTER_AREA)
    # 将RGB图像数据转换为HSV
    arr = np.array(resize_img)
    rows = resize_img.shape[0]
    cols = resize_img.shape[1]
    means = [0.485, 0.456, 0.406]
    scales = [0.229, 0.224, 0.225]
    dst_data = []
    for i in range(rows):
        for j in range(cols):
            val = arr[i][j][0] / 255.0
            val = (val - means[0]) / scales[0]
            dst_data.append(val)
    for i in range(rows):
        for j in range(cols):
            val = arr[i][j][1] / 255.0
            val = (val - means[1]) / scales[1]
            dst_data.append(val)
    for i in range(rows):
        for j in range(cols):
            val = arr[i][j][2] / 255.0
            val = (val - means[2]) / scales[2]
            dst_data.append(val)
    # print(dst_data)
    image_data = np.array(dst_data).reshape(input_shape).astype(np.float32)

    # (3) 设置输入数据 input_tensor
    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(image_data)

    # (4) 执行预测
    predictor.run()

    # (5) 得到输出数据 output_data
    output_tensor = predictor.get_output(0)
    output_data = output_tensor.numpy()

    # (6) 将评分数组output_data[0]转换为排序后的评分字典res_val，并显示图片分类结果result和评分score
    # size为labels中的类别数
    size = len(output_data[0])
    scores_dict = {}
    for i in range(size):
        scores_dict[i] = float(output_data[0][i])
    res_val = sorted(
        scores_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # print(res_val)
    return res_val
       
# 单张图片分类
if image_path[-3:] != 'txt':
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    res_val = RunModel()

    # 模型预热，warmup_num = 10
    for i in range(0, warmup_num):
        predictor.run()

    # 输出前三名的预测结果，topk = 3
    print("================== 预测报告 ===================")
    for i in range(topk):
        result = labels[res_val[i][0]]
        score = res_val[i][1]
        print('类别：' + result + ", 评估分数: " + str(score))

    # 评估单张图片的实际预测速度,repeat_num = 100
    time_val = []
    sum = 0
    for i in range(0, repeat_num):
        start_time = time.time()
        predictor.run()
        end_time = time.time()

        elapse = (end_time - start_time) / 1000.0
        time_val.append(elapse)
        sum = sum + elapse
    print("================== 速度报告 ===================")
    print("模型: " + model_dir + ", 平均运行时长: ", sum / repeat_num,"ms, 最短时长: ", min(time_val), "ms")

# 多张图片分类，数组img_list用于存放批量图片路径
else:
    img_list = []

    f = open(image_path)
    img = f.readline()[:-3]
    while img:
        img_list.append(img)
        img = f.readline()[:-3]
    f.close()
    print("================== 预测报告 ===================")
    for img_path in img_list:
        img = cv2.imread('../../../assets/images/'+img_path, cv2.COLOR_BGR2RGB)
        res_val = RunModel()
        result = labels[res_val[0][0]]
        score = res_val[0][1]
        # 多张图片的分类结果
        print('类别：' + result + ", 评估分数: " + str(score))