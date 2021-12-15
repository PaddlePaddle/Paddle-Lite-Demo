#!/bin/bash
# copy asset
cp -r ../../assets ./ppocr_demo
# mkdir PaddleLite
if [ ! -d "./ppocr_demo/third-party/PaddleLite" ]; then
 mkdir ./ppocr_demo/third-party/PaddleLite
fi
# copy paddle lite
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./ppocr_demo/third-party/PaddleLite
# copy opencv
cp -r ../../../libs/ios/opencv2.framework ./ppocr_demo/third-party

echo "copy resource successed!"