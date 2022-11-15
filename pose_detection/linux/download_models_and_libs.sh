#!/bin/bash

set -e

DETECTION_MODEL_DIR="../assets/models"
LIBS_DIR="$(pwd)"

DETECTION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/models/PP_TinyPose_128x96_qat_dis_nopact.tgz"
LIBS_URL="https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/Paddle-Lite-libs.tar.gz"

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -zxvf download.tar.gz
  rm -f download.tar.gz
}

download_and_uncompress "${DETECTION_MODEL_URL}" "${DETECTION_MODEL_DIR}"
download_and_uncompress "${LIBS_URL}" "${LIBS_DIR}"

echo "Download successful!"
