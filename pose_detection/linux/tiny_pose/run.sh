#!/bin/bash
#run

TARGET_ABI=armv8 # for 64bit
#TARGET_ABI=armv7hf # for 32bit
if [ -n "$1" ]; then
    TARGET_ABI=$1
fi
export LD_LIBRARY_PATH=../Paddle-Lite/libs/$TARGET_ABI/
export GLOG_v=0
export VSI_NN_LOG_LEVEL=0
export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1
export VIV_VX_SET_PER_CHANNEL_ENTROPY=100
export TIMVX_BATCHNORM_FUSION_MAX_ALLOWED_QUANT_SCALE_DEVIATION=30000
build/pose_detection_demo ../../assets/models/PP_TinyPose_128x96_qat_dis_nopact ../../assets/models/PP_TinyPose_128x96_qat_dis_nopact/subgraph.txt ../../assets/models/PP_TinyPose_128x96_qat_dis_nopact/infer_cfg.yml ../../assets/images/posedet_demo.jpg ../../assets/images/posedet_demo_output.jpg
