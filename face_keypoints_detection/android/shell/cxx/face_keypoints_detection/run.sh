#!/bin/bash
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/face_keypoints_detection"
ARM_ABI=$1 # arm64-v8a or armeabi-v7a

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb -s shell "cd /data/local/tmp/ && mkdir face_keypoints_detection"
# push
adb -s push ./build/face_keypoints_detection ${ADB_DIR}
adb -s push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb -s push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb -s push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb -s push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
echo "--run model on cpu---"
adb shell "cd ${ADB_DIR} \
           && chmod +x ./face_keypoints_detection \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./face_keypoints_detection \
               ./models/face_detection_for_cpu/model.nb \
               ./models/facekeypoints_detection_for_cpu/model.nb ./images/face.jpg \
               480 640 60 60\
               1 10 1 0 \
               "
