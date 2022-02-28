#!/bin/bash
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/object_detection"
ARM_ABI=$1 # arm64-v8a or armeabi-v7a

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb shell "cd /data/local/tmp/ && mkdir object_detection"
# push
adb push ./build/object_detection ${ADB_DIR}
adb push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb push ${ASSETS_DIR}/labels/ ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
adb shell "cd ${ADB_DIR} \
           && chmod +x ./object_detection \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./object_detection \
               ./models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb \
               ./images/dog.jpg \
               ./labels/pascalvoc_label_list \
               0.5 300 300 \
               0 1 100 5 0 \
           "
adb pull ${ADB_DIR}/dog_object_detection_result.jpg ./

# if run on gpu
# adb shell "cd ${ADB_DIR} \
#            && chmod +x ./object_detection \
#            && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
#            &&  ./object_detection \
#                ./models/ssd_mobilenet_v1_pascalvoc_for_gpu/model.nb \
#                ./images/dog.jpg \
#                ./labels/pascalvoc_label_list \
#                0.5 300 300 \
#                0 1 100 5 1 \
#            "
# adb pull ${ADB_DIR}/dog_object_detection_result.jpg ./
