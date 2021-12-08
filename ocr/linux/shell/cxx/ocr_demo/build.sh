#!/bin/bash
# setting NDK_ROOT root
export NDK_ROOT=/opt/android-ndk-r17c
# export NDK_ROOT=/disk/android-ndk-r23
echo "NDK_ROOT is ${NDK_ROOT}"
 [ ! -d "$(pwd)/models" ]; then

# build
cd $(pwd)/ocr_db_crnn_code
# configure
ARM_ABI=armv8
# ARM_ABI=armv7hf
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/linux/cxx"

if [ "x$1" != "x" ]; then
    ARM_ABI=$1
fi

echo "ARM_ABI is ${ARM_ABI}"
echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
rm -rf build
mkdir build
make clean
cd build
cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} -DARM_ABI=${ARM_ABI} ..
make

echo "make successful!"

# mkdir
cd ..
if [ ! -d "./ocr_demo_exec" ]; then
mkdir ocr_demo_exec
fi

cp ./ocr_db_crnn_code/pipeline ./ocr_demo_exec
cp -r ../../../assets/config.txt ./ocr_demo_exec
cp -r ../../../assets/models ./ocr_demo_exec
cp -r ../../../assets/labels ./ocr_demo_exec
cp -r ../../../assets/images ./ocr_demo_exec
cp ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so ./ocr_demo_exec
cp ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so ./ocr_demo_exec

echo "copy successful!"
