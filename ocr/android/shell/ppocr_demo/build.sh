#!/bin/bash
# setting NDK_ROOT root
export NDK_ROOT=/opt/android-ndk-r17c
# export NDK_ROOT=/disk/android-ndk-r23
echo "NDK_ROOT is ${NDK_ROOT}"
# build
cd $(pwd)/src
# configure
#ARM_ABI=arm64-v8a
ARM_ABI=armeabi-v7a
# ARM_TARGET_LANG=gcc
ARM_TARGET_LANG=clang
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"

if [ "x$1" != "x" ]; then
    ARM_ABI=$1
fi
export ARM_TARGET_LANG
export ARM_ABI
export PADDLE_LITE_DIR
export OPENCV_LITE_DIR

echo "ARM_TARGET_LANG is ${ARM_TARGET_LANG}"
echo "ARM_ABI is ${ARM_ABI}"
echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
rm -rf build
mkdir build
make clean
cd build
cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} -DARM_ABI=${ARM_ABI} -DARM_TARGET_LANG=${ARM_TARGET_LANG} -DOPENCV_LITE_DIR=${OPENCV_LITE_DIR} ..
make

echo "make successful!"

# mkdir
cd ..
if [ ! -d "./ppocr_demo" ]; then
mkdir ppocr_demo
fi

cp ./src/pipeline ./ppocr_demo
cp -r ../../../assets/config.txt ./ppocr_demo
cp -r ../../../assets/models ./ppocr_demo
cp -r ../../../assets/labels ./ppocr_demo
cp -r ../../../assets/images ./ppocr_demo
cp ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so ./ppocr_demo
cp ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so ./ppocr_demo

echo "copy successful!