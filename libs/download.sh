#!/bin/bash
ANDROID_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/android/paddle_lite_libs_v2_10_rc.tar.gz"
IOS_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/paddle_lite_libs_v2_10_rc.tar.gz"
IOS_METAL_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/paddle_lite_libs_v2_10_rc_metal.tar.gz"
ARMLINUX_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/armlinux/paddle_lite_libs_v2_10_rc.tar.gz"
OPENCV_ANDROID_URL="https://paddle-inference-dist.bj.bcebos.com/opencv4.1.0.tar.gz"
OPENCV_IOS_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/opencv2.framework.tar.gz"

if [ ! -d "$(pwd)/android" ]; then
  mkdir $(pwd)/android
fi
if [ ! -d "$(pwd)/ios" ]; then
  mkdir $(pwd)/ios
fi

if [ ! -d "$(pwd)/linux" ]; then
  mkdir $(pwd)/linux
fi

ANDROID_DIR="$(pwd)/android"
IOS_DIR="$(pwd)/ios"
LINUX_DIR="$(pwd)/linux"


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

download_and_uncompress "${ANDROID_LIBS_URL}" "${ANDROID_DIR}"
download_and_uncompress "${OPENCV_ANDROID_URL}" "${ANDROID_DIR}"
download_and_uncompress "${IOS_LIBS_URL}" "${IOS_DIR}"
download_and_uncompress "${OPENCV_IOS_URL}" "${IOS_DIR}"
download_and_uncompress "${ARMLINUX_LIBS_URL}" "${LINUX_DIR}"
download_and_uncompress "${IOS_METAL_LIBS_URL}" "${IOS_DIR}"
echo "Download successful!"
