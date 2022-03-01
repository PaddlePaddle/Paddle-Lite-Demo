#!/bin/bash
MODEL_URL="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/ssd_mobilenet_v1_pascalvoc_for_cpu_v2_10.tar.gz"
GPU_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/ssd_mobilenet_v1_pascalvoc_for_gpu_v2_10.tar.gz"
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
download_and_uncompress "${GPU_MODEL_URL}" "${MODELS_DIR}"

echo "Download successful!"
