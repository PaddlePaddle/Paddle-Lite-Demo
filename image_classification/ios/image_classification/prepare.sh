#!/bin/sh

#  prepare.sh
#  image_classification
# copy asset
cp -r ../../assets ./image_classification/third-party
# mkdir PaddleLite
if [ ! -d "./ppocr_demo/third-party/PaddleLite" ]; then
 mkdir ./image_classification/third-party/PaddleLite
fi
# copy paddle lite
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./image_classification/third-party/PaddleLite
# copy opencv
cp -r ../../../libs/ios/opencv2.framework ./image_classification/third-party

echo "copy resource successed!"
