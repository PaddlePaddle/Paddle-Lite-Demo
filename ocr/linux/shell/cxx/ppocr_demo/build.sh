#!/bin/bash
# configure
ARM_ABI=armv8
# ARM_ABI=armv7hf
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/linux"
# build
cd $(pwd)/src

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
cd ../../
if [ ! -d "./ppocr_demo" ]; then
mkdir ppocr_demo
fils

cp ./src/ppocr_demo ./ppocr_demo
cp -r ../../../../assets/config.txt ./ppocr_demo
cp -r ../../../../assets/models ./ppocr_demo
cp -r ../../../../assets/labels ./ppocr_demo
cp -r ../../../../assets/images ./ppocr_demo
cp ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so ./ppocr_demo

echo "copy successful!"
