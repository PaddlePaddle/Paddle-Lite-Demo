#!/bin/bash

set -e

DETECTION_MODEL_DIR="$(pwd)/object_detection_demo/models/ssd_mobilenet_v1_pascalvoc_for_cpu"
CLASSIFICATION_MODEL_DIR="$(pwd)/image_classification_demo/models/mobilenet_v1_for_cpu"
LIBS_DIR="$(pwd)/Paddle-Lite"

CLASSIFICATION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_for_cpu_v2_9_1.tar.gz"
DETECTION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_for_cpu_v2_9_1.tar.gz"
LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/armlinux/paddle_lite_libs_v2_9_1.tar.gz"

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  mkdir -p "${dir}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -zxvf download.tar.gz
  rm -f download.tar.gz
}

download_and_uncompress "${DETECTION_MODEL_URL}" "${DETECTION_MODEL_DIR}"
download_and_uncompress "${CLASSIFICATION_MODEL_URL}" "${CLASSIFICATION_MODEL_DIR}"
download_and_uncompress "${LIBS_URL}" "${LIBS_DIR}"

echo "Download successful!"
