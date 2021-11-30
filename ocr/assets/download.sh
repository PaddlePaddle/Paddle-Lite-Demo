#!/bin/bash
MODELS_URL="https://paddlelite-demo.bj.bcebos.com/models/ch_ppocr_mobile_v2.0_rec_slim_opt_for_cpu_v2_10_rc.tar.gz"
IMAGES_URL="https://paddlelite-demo.bj.bcebos.com/images/OCR/images.tar.gz"
LABELS_URL="https://paddlelite-demo.bj.bcebos.com/labels/OCR/labels.tar.gz"
tempdir=$(mktemp -d)

MODELS_DIR="$(pwd)/models/"
IMAGES_DIR="$(pwd)/images/"
LABELS_DIR="$(pwd)/labels/"

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

download_and_uncompress "${MODELS_URL}" "${MODELS_DIR}"
download_and_uncompress "${IMAGES_URL}" "${IMAGES_DIR}"
download_and_uncompress "${LABELS_URL}" "${LABELS_DIR}"

echo "Download successful!"