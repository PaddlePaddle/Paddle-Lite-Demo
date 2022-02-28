#!/bin/bash
# MODEL_URL="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/mobilenet_v1_for_cpu_v2_10.tar.gz"
MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_for_cpu_v2_10_rc.tar.gz"

MODELS_DIR="$(pwd)/models/"

if [ ! -d "$(pwd)/models" ]; then
  mkdir $(pwd)/models
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

echo "Download successful!"
