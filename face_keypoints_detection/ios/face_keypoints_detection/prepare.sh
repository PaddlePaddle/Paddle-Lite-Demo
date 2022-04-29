#!/bin/sh

#  prepare.sh
#  face_keypoints_detection
# copy asset
cp -r ../../assets ./face_keypoints_detection/third-party
# mkdir PaddleLite
if [ ! -d "./face_keypoints_detection/third-party/PaddleLite" ]; then
 mkdir ./face_keypoints_detection/third-party/PaddleLite
fi
# copy paddle lite
cp -r ../../../libs/ios/inference_lite_lib.ios64.armv8/ ./face_keypoints_detection/third-party/PaddleLite
# copy opencv
cp -r ../../../libs/ios/opencv2.framework ./face_keypoints_detection/third-party

echo "copy resource successed!"
