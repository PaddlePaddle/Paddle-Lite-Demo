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

function compile_human_segmentation {
     cd ./human_segmentation/assets
     chmod +x ./download.sh
     bash ./download.sh
     echo "human_segmentation"
     cd ../android/shell/cxx/human_segmentation
     chmod +x ./build.sh
     sed -i '3s/disk/opt/g' ./build.sh
     echo "-- arm_v8 --"
     bash ./build.sh "arm64-v8a"
     echo "-- arm_v7 --"
     bash ./build.sh "armeabi-v7a"
     cd ../../../../../
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
     cd ../../../../ 
}

function compile_face_detection {
     cd ./face_detection/assets
     chmod +x ./download.sh
     bash ./download.sh
     cd ../android/shell/cxx/face_detection
     chmod +x ./build.sh
     sed -i '3s/disk/opt/g' ./build.sh
     echo "-- arm_v8 --"
     bash ./build.sh "arm64-v8a"
     echo "-- arm_v7 --"
     bash ./build.sh "armeabi-v7a"
     cd ../../../../../
}

function compile_face_keypoints_detection {
     cd ./face_keypoints_detection/assets
     chmod +x ./download.sh
     bash ./download.sh
     cd ../android/shell/cxx/face_keypoints_detection
     chmod +x ./build.sh
     sed -i '3s/disk/opt/g' ./build.sh
     echo "-- arm_v8 --"
     bash ./build.sh "arm64-v8a"
     echo "-- arm_v7 --"
     bash ./build.sh "armeabi-v7a"
     cd ../../../../../ 
}

function main {
  # step1. download android lib
  echo "--download_lib--: $(pwd)"
  download_lib
  
  # step2. build and run human_segmentation
  echo "--compile_human_segmentation--: $(pwd)"
  compile_human_segmentation
  
  # step3. build and run image_classification
  echo "--compile_image_classification--: $(pwd)"
  compile_image_classification
  
  # step4. build and run ocr
  echo "--compile_ocr--: $(pwd)"
  compile_ocr
  
  # step5. build and run face_detection
  echo "--compile_face_detection--: $(pwd)"
  compile_face_detection
  echo "--end--: $(pwd)"    

  # step6. build and run face_keypoints_detection
  echo "--compile_face_keypoints_detection--: $(pwd)"
  compile_face_keypoints_detection
  echo "--end--: $(pwd)"     
  
  # step7. build and run object_detection
  echo "--compile_object_detection--: $(pwd)"
  compile_object_detection
  echo "--end--: $(pwd)"

}

main
