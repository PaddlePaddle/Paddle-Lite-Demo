#!/bin/sh

#  prepare.sh
#  facedetection
# copy asset
cp -r ../../assets ./face_detection/third-party
# mkdir PaddleLite
if [ ! -d "./face_detection/third-party/PaddleLite" ]; then
 mkdir ./face_detection/third-party/PaddleLite
fi
# copy paddle lite
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./face_detection/third-party/PaddleLite
# copy opencv
cp -r ../../../libs/ios/opencv2.framework ./face_detection/third-party

echo "copy resource successed!"
