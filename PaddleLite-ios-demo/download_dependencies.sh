#!/bin/bash
set -e
tempdir=$(mktemp -d)

CLASSIFICATION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/mobilenet_v1_fp32_224_for_cpu_v2_8_0.tar.gz"
DETECTION_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/models/ssd_mobilenet_v1_pascalvoc_fp32_300_for_cpu_v2_8_0.tar.gz"
PADDLE_LITE_LIB_URL="https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.8/inference_lite_lib.ios.armv8.with_extra.tiny_publish.tar.gz"
OPENCV_FRAMEWORK_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/opencv2.framework.tar.gz"


download_and_extract() {
    local url="$1"
    local dst_dir="$2"

    echo "Downloading ${url} ..."
    curl -L ${url} > ${tempdir}/temp.tar.gz
    echo "Download ${url} done "

    if [ ! -d ${dst_dir} ];then
        mkdir -p ${dst_dir}
    fi

    echo "Extracting ..."
    tar -zxvf ${tempdir}/temp.tar.gz -C ${dst_dir}
    echo "Extract done "

}
download_and_extract_once(){
    local url="$1"
    local dst_dir="$2"
    local file_name="$3"
    if [ ! -f  ${tempdir}/${file_name}.tar.gz ];then
        echo "Downloading ${url} ..."
        curl -L ${url} > ${tempdir}/${file_name}.tar.gz
        echo "Download ${url} done "
        echo "Extracting ..."
        tar -zxvf ${tempdir}/${file_name}.tar.gz -C ${tempdir}
        echo "Extract done "
    fi
    if [ ! -d ${dst_dir} ];then
        mkdir -p ${dst_dir}
    fi

    if [ "$file_name" == "lib" ];then
        echo "Copying lib "
        cp -rf ${tempdir}/inference_lite_lib.ios64.armv8/* ${dst_dir}
    fi
    if [ "$file_name" == "opencv" ];then
        echo "Copying opencv "
        cp -rf ${tempdir}/opencv2.framework ${dst_dir}
    fi
}

# for classification demo
echo -e "[Download ios classificiton demo denpendancy]\n"
download_and_extract "${CLASSIFICATION_MODEL_URL}" "./ios-classification_demo/classification_demo/models/mobilenetv1"
download_and_extract_once "${PADDLE_LITE_LIB_URL}" "./ios-classification_demo/classification_demo" "lib"
download_and_extract_once "${OPENCV_FRAMEWORK_URL}" "./ios-classification_demo/classification_demo" "opencv"
echo -e "[done]\n"

# for detection demo
echo -e "[Download ios detection demo denpendancy]\n"
download_and_extract "${DETECTION_MODEL_URL}" "./ios-detection_demo/detection_demo/models/mobilenetv1-ssd"
download_and_extract_once "${PADDLE_LITE_LIB_URL}" "./ios-detection_demo/detection_demo" "lib"
download_and_extract_once "${OPENCV_FRAMEWORK_URL}" "./ios-detection_demo/detection_demo" "opencv"

rm -rf ${tempdir}

echo -e "[done]"

