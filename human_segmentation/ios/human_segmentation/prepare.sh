#!/bin/sh
#  human_segmentation

# copy asset
if [ ! -d "./human_segmentation/third-party" ]; then
 mkdir ./human_segmentation/third-party
fi
cp -r ../../assets ./human_segmentation/third-party/

# mkdir PaddleLite
if [ ! -d "./human_segmentation/third-party/PaddleLite" ]; then
 mkdir ./human_segmentation/third-party/PaddleLite
fi

# copy paddle lite
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./human_segmentation/third-party/PaddleLite
# copy opencv
cp -r ../../../libs/ios/opencv2.framework ./human_segmentation/third-party

echo "copy resource successed!"
