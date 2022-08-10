#!/bin/bash
#run

export LD_LIBRARY_PATH=../Paddle-Lite/libs/armv8/
export GLOG_v=0
export VSI_NN_LOG_LEVEL=0
export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1
export VIV_VX_SET_PER_CHANNEL_ENTROPY=100
build/object_detection_demo models/picodetv2_relu6_coco_no_fuse ../../assets/labels/coco_label_list.txt models/picodetv2_relu6_coco_no_fuse/subgraph.txt models/picodetv2_relu6_coco_no_fuse/picodet.yml 
