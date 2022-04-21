#!/bin/bash
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/face_detection"
ARM_ABI=$1 # arm64-v8a or armeabi-v7a

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb shell "cd /data/local/tmp/ && mkdir face_detection"
# push
adb push ./build/face_detection ${ADB_DIR}
adb push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
echo "--run model on cpu---"
adb -s 1ddcf602 shell "cd ${ADB_DIR} \
           && chmod +x ./face_detection \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./face_detection \
               ./models/facedetection_for_cpu/model.nb \
               ./images/face.jpg \
               100 0.5 0.5 320 240 \
               1 0 100 10 \
               "
