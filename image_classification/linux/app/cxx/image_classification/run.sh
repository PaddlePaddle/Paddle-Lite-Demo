#!/bin/bash

# configure
TARGET_ARCH_ABI=armv8 # for RK3399, set to default arch abi
#TARGET_ARCH_ABI=armv7hf # for Raspberry Pi 3B
PADDLE_LITE_DIR="$(pwd)/../../../../../libs/linux"
IMAGES_DIR="$(pwd)/../../../../assets/images"
LABELS_DIR="$(pwd)/../../../../assets/labels"
MODELS_DIR="$(pwd)/../../../../assets/models"
if [ "x$1" != "x" ]; then
    TARGET_ARCH_ABI=$1
fi

# build
rm -rf build
mkdir build
cd build
cmake -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR} -DTARGET_ARCH_ABI=${TARGET_ARCH_ABI} ..
make

#run
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} 
./image_classification ${MODELS_DIR}/mobilenet_v1_for_cpu/model.nb ${LABELS_DIR}/labels.txt 3 ${IMAGES_DIR}/tabby_cat.jpg ./result.jpg
