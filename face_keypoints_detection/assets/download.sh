#!/bin/bash
FACE_DETECTION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ultra_light_fast_generic_face_detector_1mb_640_for_cpu_v2_10_rc.tar.gz"
FACE_KEYPOINTS_DETECTION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/facekeypoints_detector_fp32_60_60_for_cpu_v2_10_rc.tar.gz"

FACE_DETECTION_MODEL_DIR="$(pwd)/models/face_detection_for_cpu"
FACE_KEYPOINTS_DETECTION_MODEL_DIR="$(pwd)/models/facekeypoints_detection_for_cpu"

if [ ! -d "$(pwd)/models" ]; then
  mkdir -p $(pwd)/models/face_detection_for_cpu
  mkdir -p $(pwd)/models/facekeypoints_detection_for_cpu  
fi

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -zxvf download.tar.gz
  rm -f download.tar.gz
  
  cd ../../
}

download_and_uncompress "${FACE_DETECTION_MODEL_URL}" "${FACE_DETECTION_MODEL_DIR}"
download_and_uncompress "${FACE_KEYPOINTS_DETECTION_MODEL_URL}" "${FACE_KEYPOINTS_DETECTION_MODEL_DIR}"

echo "Download successful!"
