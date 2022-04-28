#!/bin/bash
MODEL_URL="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/ssd_mobilenet_v1_pascalvoc_for_cpu_v2_10.tar.gz"
GPU_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/ssd_mobilenet_v1_pascalvoc_for_gpu_v2_10.tar.gz"
PICODET_MODLE_URL="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/picodet_s_320_coco_for_cpu.tar.gz"
YOLO_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10.tar.gz"
MODELS_DIR="$(pwd)/models/"
# dev cpu + gpu lib(fix picode run error)
ANDROID_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/android/paddle_lite_libs_dev_gpu.tar.gz"
# dev cpu lib(fix picode run error)
IOS_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/paddle_lite_libs_dev.tar.gz"

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
download_and_uncompress "${PICODET_MODLE_URL}" "${MODELS_DIR}"
download_and_uncompress "${YOLO_MODEL_URL}" "${MODELS_DIR}"
ANDROID_DIR="$(pwd)/../../libs/android"
IOS_DIR="$(pwd)/../../libs/ios"
download_and_uncompress "${ANDROID_LIBS_URL}" "${ANDROID_DIR}"
download_and_uncompress "${IOS_LIBS_URL}" "${IOS_DIR}"

echo "Download successful!"
