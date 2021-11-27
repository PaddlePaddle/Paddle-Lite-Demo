#!/bin/bash
# setting NDK_ROOT root
export NDK_ROOT=/opt/android-ndk-r17c
echo "NDK_ROOT is ${NDK_ROOT}"
# download model
set -e
if [ ! -d "$(pwd)/models" ]; then
mkdir $(pwd)/models
fi
if [ ! -d "$(pwd)/Paddle-Lite" ]; then
mkdir $(pwd)/Paddle-Lite
fi

MODEL_DIR="$(pwd)/models/"
LIBS_DIR="$(pwd)/Paddle-Lite"

CLS_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ch_ppocr_mobile_v2.0_cls_slim_opt_for_cpu_v2_10_rc.tar.gz"
DET_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ch_ppocr_mobile_v2.0_det_slim_opt_for_cpu_v2_10_rc.tar.gz"
REC_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ch_ppocr_mobile_v2.0_rec_slim_opt_for_cpu_v2_10_rc.tar.gz"
LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/android/paddle_lite_libs_v2_10_rc.tar.gz"

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -zxvf download.tar.gz
  rm -f download.tar.gz
  cd ..
}

download_and_uncompress "${CLS_MODEL_URL}" "${MODEL_DIR}"
download_and_uncompress "${DET_MODEL_URL}" "${MODEL_DIR}"
download_and_uncompress "${REC_MODEL_URL}" "${MODEL_DIR}"
download_and_uncompress "${LIBS_URL}" "${LIBS_DIR}"

echo "Download successful!"

# build
cd $(pwd)/ocr_db_crnn_code
# configure
ARM_ABI=arm64-v8a
# ARM_ABI=armveabi-v7a
export ARM_ABI
echo "ARM_ABI is ${ARM_ABI}"
make
echo "make successful!"

# mkdir
cd ..
if [ ! -d "./ocr_demo_exec" ]; then
mkdir ocr_demo_exec
fi

cp ./ocr_db_crnn_code/ocr_db_crnn ./ocr_demo_exec
cp ./config.txt ./ocr_demo_exec
cp -r ./models ./ocr_demo_exec
cp -r ./labels ./ocr_demo_exec
cp -r ./images ./ocr_demo_exec
cp ${LIBS_DIR}/cxx/libs/${ARM_ABI}/libc++_shared.so ./ocr_demo_exec
cp ${LIBS_DIR}/cxx/libs/${ARM_ABI}/libpaddle_light_api_shared.so ./ocr_demo_exec

echo "copy successful!"
