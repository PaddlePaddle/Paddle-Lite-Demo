#!/bin/bash
if [ ! -d "./app/src/main/assets" ]; then
mkdir ./app/src/main/assets
fi


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
