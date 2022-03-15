#!/bin/bash
set -ex

# Set proxy if exists
if [ -n "$PROXY" ]; then
     export http_proxy=$PROXY
     export https_proxy=$PROXY
fi
if [ -n "$NO_PROXY" ]; then
     export no_proxy=$NO_PROXY
fi
####################################################################################################

function download_lib {
    chmod +x ./libs/download.sh
    cd ./libs
    bash ./download.sh
    cd ../
}

function compile_image_classification {
     cd ./image_classification/assets
     chmod +x ./download.sh
     bash ./download.sh
     echo "image_classification"
     cd ../android/shell/cxx/image_classification
     chmod +x ./build.sh
     sed -i '3s/disk/opt/g' ./build.sh
     echo "-- arm_v8 --"
     bash ./build.sh "arm64-v8a"
     echo "-- arm_v7 --"
     bash ./build.sh "armeabi-v7a"
     cd ../../../../../
}

function compile_object_detection {
     cd ./object_detection/assets
     chmod +x ./download.sh
     bash ./download.sh
     echo "ssd_mobilenetv1_detection"
     cd ../android/shell/cxx/ssd_mobilenetv1_detection
     chmod +x ./build.sh
     sed -i '3s/disk/opt/g' ./build.sh
     echo "-- arm_v8 --"
     bash ./build.sh "arm64-v8a"
     echo "-- arm_v7 --"
     bash ./build.sh "armeabi-v7a"
     echo "picodet_detection"
     cd ../picodet_detection
     chmod +x ./build.sh
     sed -i '3s/disk/opt/g' ./build.sh
     echo "-- arm_v8 --"
     bash ./build.sh "arm64-v8a"
     echo "-- arm_v7 --"
     bash ./build.sh "armeabi-v7a"
     cd ../../../../../
}

function compile_ocr {
     cd ./ocr/assets
     chmod +x ./download.sh
     bash ./download.sh
     cd ../android/shell/ppocr_demo
     chmod +x ./build.sh
     sed -i '3s/disk/opt/g' ./build.sh
     echo "-- arm_v8 --"
     bash ./build.sh "arm64-v8a"
     echo "-- arm_v7 --"
     # run error
     # bash ./build.sh "armeabi-v7a"
     cd ../../../../../ 
}

function main {
  # step1. download android lib
  echo "--download_lib--"
  download_lib
  # step2. build and run image_classification
  echo "--compile_image_classification--"
  compile_image_classification
  # step3. build and run ocr
  echo "--compile_ocr--"
  compile_ocr
  # step4. build and run object_detection
  echo "--compile_object_detection--"
  compile_object_detection
  echo "--end--"
}

main
