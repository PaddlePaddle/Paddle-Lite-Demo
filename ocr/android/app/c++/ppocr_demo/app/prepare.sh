#!/bin/bash
if [ ! -d "./PaddleLite" ]; then
mkdir PaddleLite
fi
if [ ! -d "./OpenCV" ]; then
mkdir OpenCV
fi
# copy paddle lite cxx
cp -r ../../../../../../libs/android/cxx ./PaddleLite
# copy opencv
cp -r ../../../../../../libs/android/sdk ./OpenCV

# copy model
cp -r ../../../../../assets/models/ src/main/assets/models/
# copy images
cp -r ../../../../../assets/images/   src/main/assets/images/
# copy labels
cp -r ../../../../../assets/labels/ src/main/assets/labels/
