#!/bin/bash
# setting NDK_ROOT root
export NDK_ROOT=/Users/zhoukangkang/Library/Android/sdk/ndk/21.1.6352462
echo "NDK_ROOT is ${NDK_ROOT}"

# configure
ARM_ABI=arm64-v8a
# ARM_ABI=armeabi-v7a
# ARM_TARGET_LANG=gcc
ARM_TARGET_LANG=clang
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"

if [ "x$1" != "x" ]; then
    ARM_ABI=$1
fi

echo "ARM_TARGET_LANG is ${ARM_TARGET_LANG}"
echo "ARM_ABI is ${ARM_ABI}"
echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
# build
if [ -d "$(pwd)/build" ]; then
  rm -rf build
fi
mkdir build
#make clean
cd build
cmake -DANDROID_PLATFORM=android-21 -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} -DARM_ABI=${ARM_ABI} -DARM_TARGET_LANG=${ARM_TARGET_LANG} -DOPENCV_LITE_DIR=${OPENCV_LITE_DIR} -DNDK_ROOT=${NDK_ROOT} ..
make -j10
cd ..

echo "make successful!"
sh run.sh ${ARM_ABI}
