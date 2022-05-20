#!/bin/sh

#  prepare.sh
#  ppshitu

# copy asset
if [ ! -d "./shitu/third-party" ]; then
 mkdir ./shitu/third-party
fi
cp -r ../../assets ./shitu/third-party

# copy PaddleLite
if [ ! -d "./shitu/third-party/PaddleLite" ]; then
 mkdir ./shitu/third-party/PaddleLite
fi
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./shitu/third-party/PaddleLite

# download opencv
OPENCV_IOS_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/opencv-4.5.5-ios-framework.tar.gz"
IOS_DIR="$(pwd)/shitu/third-party"
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
download_and_uncompress "${OPENCV_IOS_URL}" "${IOS_DIR}"

echo "copy resource successed!"
