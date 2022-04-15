#!/bin/bash
# copy asset
cp -r ../../assets ./ppocr_demo
# mkdir PaddleLite
if [ ! -d "./ppocr_demo/third-party/PaddleLite" ]; then
 mkdir ./ppocr_demo/third-party/PaddleLite
fi
# copy paddle lite
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./ppocr_demo/third-party/PaddleLite
# download opencv
OPENCV_IOS_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/opencv-4.5.5-ios-framework.tar.gz"
IOS_DIR="$(pwd)/ppocr_demo/third-party"
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
