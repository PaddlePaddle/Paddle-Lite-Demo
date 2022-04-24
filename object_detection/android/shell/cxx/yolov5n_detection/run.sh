#!/bin/bash
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/yolov5n_detection"
ARM_ABI=$1 # arm64-v8a or armeabi-v7a

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb shell "rm -rf /data/local/tmp/* && cd /data/local/tmp/ && mkdir yolov5n_detection"
# push
adb push ./build/yolov5n_detection ${ADB_DIR}
adb push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb push ${ASSETS_DIR}/labels/ ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
adb shell "cd ${ADB_DIR} \
            && chmod +x ./yolov5n_detection \
            && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
            &&  ./yolov5n_detection \
                ./models/yolov5n_coco_for_cpu/model.nb \
                ./images/dog.jpg \
                ./labels/coco_label_list.txt \
                0.25 320 320 \
                0 1 10 1 0 \
            "
adb pull ${ADB_DIR}/dog_yolov5n_detection_result.jpg ./

# if run on gpu
#adb shell "cd ${ADB_DIR} \
#            && chmod +x ./yolov5n_detection \
#            && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
#            &&  ./yolov5n_detection \
#                ./models/yolov5n_coco_for_gpu/model.nb \
#                ./images/dog.jpg \
#                ./labels/coco_label_list.txt \
#                0.25 320 320 \
#                0 1 10 1 1 \
#            "
#adb pull ${ADB_DIR}/dog_yolov5n_detection_result.jpg ./
