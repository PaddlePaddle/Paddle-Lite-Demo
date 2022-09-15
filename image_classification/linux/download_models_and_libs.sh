#!/bin/bash

set -e

MODEL_DIR="$(pwd)/mobilenet_v3/models"
LIBS_DIR="$(pwd)"

MODEL_URL="https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/models/mobilenet_v3_perchannel_int8.tar.gz"
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

download_and_uncompress "${MODEL_URL}" "${MODEL_DIR}"
download_and_uncompress "${LIBS_URL}" "${LIBS_DIR}"

echo "Download successful!"
