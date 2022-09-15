#!/bin/bash
#run

TARGET_ABI=armv8 # for 64bit, such as Amlogic A311D
#TARGET_ABI=armv7hf # for 32bit, such as Rockchip 1109/1126
if [ -n "$1" ]; then
    TARGET_ABI=$1
fi
export LD_LIBRARY_PATH=../Paddle-Lite/libs/$TARGET_ABI/
export GLOG_v=0
export VSI_NN_LOG_LEVEL=0
export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1
export VIV_VX_SET_PER_CHANNEL_ENTROPY=100
build/image_classification models/mobilenet_v3_perchannel_int8 ../../assets/labels/coco_label_list.txt null null 
