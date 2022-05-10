#!/bin/sh

#  prepare.sh
#  image_classification
# copy asset
cp -r ../../assets ./detection_demo/third-party/assets
# mkdir PaddleLite
if [ ! -d "./detection_demo/third-party/PaddleLite" ]; then
 mkdir ./detection_demo/third-party/PaddleLite
fi
# copy paddle lite
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./detection_demo/third-party/PaddleLite
# copy opencv
cp -r ../../../libs/ios/opencv2.framework ./detection_demo/third-party

echo "copy resource successed!"
