#!/bin/bash
IMAGES_URL="https://paddlelite-demo.bj.bcebos.com/images/CLS/image.tar.gz"
LABELS_URL="https://paddlelite-demo.bj.bcebos.com/labels/CLS/labels.tar.gz"
MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/CLS/mobilenet_v1_for_cpu.tar.gz"

MODELS_DIR="$(pwd)/models/"
IMAGES_DIR="$(pwd)/images/"
LABELS_DIR="$(pwd)/labels/"

if [ ! -d "$(pwd)/models" ]; then
  mkdir $(pwd)/models
fi
if [ ! -d "$(pwd)/images" ]; then
  mkdir $(pwd)/images
fi

if [ ! -d "$(pwd)/labels" ]; then
  mkdir $(pwd)/labels
fi

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -zxvf download.tar.gz
  rm -f download.tar.gz
  
  cd ..
}

download_and_uncompress "${MODEL_URL}" "${MODELS_DIR}"
download_and_uncompress "${IMAGES_URL}" "${IMAGES_DIR}"
download_and_uncompress "${LABELS_URL}" "${LABELS_DIR}"

echo "Download successful!"
