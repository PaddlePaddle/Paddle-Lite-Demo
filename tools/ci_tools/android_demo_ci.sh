#!/bin/bash
set -ex

# Set proxy if exists
if [ -n "$PROXY" ]; then
     export http_proxy=$PROXY
     export https_proxy=$PROXY∂
fi
if [ -n "$NO_PROXY" ]; then
     export no_proxy=$NO_PROXY∑
fi
cd .. && rm -rf Paddle-Lite-Demo

####################################################################################################
# 1. functions of prepare workspace before compiling
####################################################################################################
# 1.1 compile paddlelite android lib
function compile_lib {
    git clone https://github.com/PaddlePaddle/Paddle-Lite-Demo.git && cd Paddle-Lite-Demo
    git branch -a && git checkout develop
    chmod +x ./libs/download.sh
    bash ./libs/download.sh
    bash ./object_detection/assets/download.sh
    sed -i '3s/disk/opt/g' ./object_detection/android/shell/cxx/picodet_detection/build.sh
    cd object_detection/android/shell/cxx/picodet_detection && bash build.sh
}

function main {
  # step1. compile paddle-lite-demo android lib
  compile_lib

}

main