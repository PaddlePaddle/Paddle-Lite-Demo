# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
image classfiy python api demo
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from numpy.lib.shape_base import column_stack

from paddlelite.lite import *
import numpy as np
import cv2
import time

# Command arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", default="", type=str, required=True, help="Opt Model dir path")
parser.add_argument(
    "--input_shape", default=[], type=str, required=True, help="Model input shape, eg: 1,3,224,224")
parser.add_argument(
    "--image_path", default="", type=str, help="Test image path")
parser.add_argument(
    "--label_path", default="", type=str, help="Label path")
parser.add_argument(
    "--topk", default=1, type=int, help="Topk")
parser.add_argument(
    "--threads", default=1, type=int, help="Test thread num")
parser.add_argument(
    "--warmup", default=0, type=int, help="Test warmup num")
parser.add_argument(
    "--repeats", default=1, type=int, help="Test repeats num")
def parse_shape(shape_val):
    str_val = shape_val.split(",")
    input_shape = []
    for i in range(0, len(str_val)):
        val = int(str_val[i])
        input_shape.append(val)
    return input_shape

def load_labels(label_path):
    labels = []
    fp_r = open(label_path, 'r')
    data = fp_r.readline()
    while data:
        labels.append(data)
        data = fp_r.readline()
    fp_r.close()
    return labels

def Init(model_dir, threads):
    # 1. Set config information
    config = MobileConfig()
    config.set_model_from_file(model_dir)
    # run arm 
    config.set_threads(threads)
    # 2. Create paddle predictor
    predictor = create_paddle_predictor(config)
    return predictor

def Preprocss(predictor, input_shape, has_img, image_path, means, scales):
    # 3. Set input data
    input_tensor = predictor.get_input(0)
    if has_img:
        img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        resize_img = cv2.resize(img, [input_shape[2], input_shape[3]], interpolation = cv2.INTER_AREA)
        arr = np.array(resize_img)
        rows = resize_img.shape[0]
        cols =resize_img.shape[1]
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
        input_tensor.from_numpy(np.array(dst_data).reshape(input_shape).astype(np.float32))
    else:
        input_tensor.from_numpy(np.ones(input_shape).astype("float32"))

def Postprocss(predictor, labels, topk):
    #5. Get output data
    output_tensor = predictor.get_output(0)
    output_data = output_tensor.numpy()
    # print("output_data: ", output_data)
    
    size = len(output_data[0])
    scores_dict = {}
    for i in range(size):
        scores_dict[i] = float(output_data[0][i])
    res_val = sorted(scores_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    # print("--out: ", res_val)

    # print topk
    print("================== Precision Report ===================")
    for i in range(topk):
        index = res_val[i][0]
        score = res_val[i][1]
        res_str = "i: " + str(i) + ", index: " + str(index)
        if len(labels):
            res_str = res_str + ", name: " + labels[index]
        res_str = res_str + ", score: " + str(score)
        print(res_str)
    print("================== Report End ===================")
 
def RunModel(args):
    thread_num = 1
    warmup_num = 0
    repeat_num = 1
    if args.threads:
        thread_num = args.threads
    if args.warmup:
        warmup_num = args.warmup
    if args.repeats:
        repeat_num = args.repeats
    model_dir = args.model_dir
    predictor = Init(model_dir, thread_num)
    has_img = False
    image_path = ""
    if args.image_path:
        image_path = args.image_path
        has_img = True
    label_path = ""
    labels = []
    if args.label_path:
        label_path = args.label_path
        labels = load_labels(label_path)
    topk = 1
    if args.topk:
        topk = args.topk
    input_shape = parse_shape(args.input_shape)
    means = [0.485, 0.456, 0.406]
    scales = [0.229, 0.224, 0.225]
    Preprocss(predictor, input_shape, has_img, image_path, means, scales)
    # warmup
    for i in range(0, warmup_num):
        predictor.run()
    # Run model
    time_val = []
    sum = 0
    for i in range(0, repeat_num):
        start_time = time.time()
        predictor.run()
        end_time = time.time()
        elapse = (end_time - start_time) / 1000.0
        # print("i: ", i, elapse, "ms")
        time_val.append(elapse)
        sum = sum + elapse
    print("================== Speed Report ===================")
    print("model: " + model_dir + ", run avg_time: ", sum /repeat_num , "ms, min_time: ", min(time_val), "ms")
    # 5. postprocess
    Postprocss(predictor, labels, topk)


if __name__ == '__main__':
    args = parser.parse_args()
    RunModel(args)
