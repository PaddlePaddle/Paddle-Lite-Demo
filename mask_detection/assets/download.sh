#!/bin/bash
MODEL_URL_DETEC="https://paddlelite-demo.bj.bcebos.com/models/pyramidbox_lite_fp32_for_cpu_v2_10_rc.tar.gz"
MODEL_URL_CLASSIFY="https://paddlelite-demo.bj.bcebos.com/models/mask_detector_fp32_128_128_for_cpu_v2_10_rc.tar.gz"

MODELS_DIR_DETEC="$(pwd)/models/pyramidbox_lite_for_cpu"
MODELS_DIR_CLASSIFY="$(pwd)/models/mask_detector_for_cpu"

if [ ! -d "$(pwd)/models" ]; then
  mkdir -p "${MODELS_DIR_DETEC}"
  mkdir -p "${MODELS_DIR_CLASSIFY}"
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

download_and_uncompress "${MODEL_URL_DETEC}" "${MODELS_DIR_DETEC}"
download_and_uncompress "${MODEL_URL_CLASSIFY}" "${MODELS_DIR_CLASSIFY}"

echo "Download successful!"
