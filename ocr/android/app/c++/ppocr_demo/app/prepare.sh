#!/bin/bash
if [ ! -d "./PaddleLite" ]; then
mkdir PaddleLite
fi
if [ ! -d "./OpenCV" ]; then
mkdir OpenCV
fi

if [ ! -d "./src/main/assets" ]; then
mkdir ./src/main/assets
fi

# copy paddle lite cxx
cp -r ../../../../../../libs/android/cxx ./PaddleLite
echo "copy paddle-lite lib successed"
# copy opencv
cp -r ../../../../../../libs/android/sdk ./OpenCV
echo "copy opencv lib successed"
# copy model
cp -r ../../../../../assets/models/ src/main/assets/
echo "copy model successed"
# copy images
cp -r ../../../../../assets/images/   src/main/assets/
echo "copy images successed"
# copy labels
cp -r ../../../../../assets/labels/ src/main/assets/
echo "copy labels successed"
cp ../../../../../assets/config.txt src/main/assets/
echo "copy config successed"
