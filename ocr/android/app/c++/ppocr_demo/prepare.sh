#!/bin/bash
if [ ! -d "./app/PaddleLite" ]; then
mkdir PaddleLite
fi
if [ ! -d "./app/OpenCV" ]; then
mkdir ./app/OpenCV
fi

if [ ! -d "./app/src/main/assets" ]; then
mkdir ./app/src/main/assets
fi

# copy paddle lite cxx
#cp -r ../../../../../libs/android/cxx ./app/PaddleLite
echo "copy paddle-lite lib successed"
# copy opencv
#cp -r ../../../../../libs/android/sdk ./app/OpenCV
echo "copy opencv lib successed"
# copy model
cp -r ../../../../assets/models/ ./app/src/main/assets/
echo "copy model successed"
# copy images
cp -r ../../../../assets/images/   ./app/src/main/assets/
echo "copy images successed"
# copy labels
cp -r ../../../../assets/labels/ ./app/src/main/assets/
echo "copy labels successed"
cp ../../../../assets/config.txt ./app/src/main/assets/
echo "copy config successed"
