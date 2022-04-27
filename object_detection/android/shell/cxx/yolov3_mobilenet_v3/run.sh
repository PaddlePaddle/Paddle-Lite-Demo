#!/bin/bash
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/yolo_v3_mobilenetv3_detection"
ARM_ABI=$1 # arm64-v8a or armeabi-v7a

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb -s 4c4f947c shell "cd /data/local/tmp/ && mkdir yolo_v3_mobilenetv3_detection"
# push
adb -s 4c4f947c push ./build/yolov3_mobilenet_v3 ${ADB_DIR}
adb -s 4c4f947c push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb -s 4c4f947c push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb -s 4c4f947c push ${ASSETS_DIR}/labels/ ${ADB_DIR}
adb -s 4c4f947c push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb -s 4c4f947c push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
adb -s 4c4f947c shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10//model.nb \
               ./images/dog.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 10 1 0 \
           "
adb -s 4c4f947c  pull ${ADB_DIR}/dog_yolo_v3_mobilenetv3_detection_result.jpg ./

# if run on gpu
#adb shell "cd ${ADB_DIR} \
#           && chmod +x ./yolov3_mobilenet_v3 \
#           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
#           &&  ./yolov3_mobilenet_v3 \
#               ./models/yolo_v3_mobilenetv3_detection_for_cpu/model.nb \
#               ./images/dog.jpg \
#               ./labels/coco_label_list.txt \
#               0.5 320 320 \
#                0 1 10 1 1 \
#           "
