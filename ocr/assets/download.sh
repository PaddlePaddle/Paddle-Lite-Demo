#!/bin/bash
IMAGES_URL="https://paddlelite-demo.bj.bcebos.com/images/OCR/images.tar.gz"
LABELS_URL="https://paddlelite-demo.bj.bcebos.com/labels/OCR/labels.tar.gz"
CLS_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ch_ppocr_mobile_v2.0_cls_slim_opt_for_cpu_v2_10_rc.tar.gz"
DET_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ch_ppocr_mobile_v2.0_det_slim_opt_for_cpu_v2_10_rc.tar.gz"
REC_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ch_ppocr_mobile_v2.0_rec_slim_opt_for_cpu_v2_10_rc.tar.gz"
CONFIG_TXT_URL="https://paddlelite-demo.bj.bcebos.com/labels/OCR/config.tar.gz"

if [ ! -d "$(pwd)/models" ]; then
 mkdir $(pwd)/models
fi
if [ ! -d "$(pwd)/images" ]; then
 mkdir $(pwd)/images
fi

if [ ! -d "$(pwd)/labels" ]; then
 mkdir $(pwd)/labels
fi

MODELS_DIR="$(pwd)/models/"
IMAGES_DIR="$(pwd)/images/"
LABELS_DIR="$(pwd)/labels/"
CONFIG_DIR="$(pwd)"

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

download_and_uncompress "${CLS_MODEL_URL}" "${MODELS_DIR}"
download_and_uncompress "${DET_MODEL_URL}" "${MODELS_DIR}"
download_and_uncompress "${REC_MODEL_URL}" "${MODELS_DIR}"
download_and_uncompress "${IMAGES_URL}" "${IMAGES_DIR}"
download_and_uncompress "${LABELS_URL}" "${LABELS_DIR}"
download_and_uncompress "${CONFIG_TXT_URL}" "${CONFIG_DIR}"

echo "Download successful!"
