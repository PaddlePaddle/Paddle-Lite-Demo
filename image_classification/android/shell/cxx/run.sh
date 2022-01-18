#!/bin/bash
PADDLE_LITE_DIR="$(pwd)/../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../assets"
ADB_DIR =="/data/local/tmp/image_classify"

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb shell "cd /data/local/tmp/ && mkdir image_classify"
# push
adb push ./build/image_classification ${ADB_DIR}
adb push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb push ${ASSETS_DIR}/labels/ ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
                ./models/mobilenet_v1_for_cpu/model.nb \
                ./images/tabby_cat.jpg \
                ./labels/labels.txt \
                3 224 224 \
                0 1 100 10 \
                "

