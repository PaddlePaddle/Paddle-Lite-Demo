# 目标检测 C++ API Demo 使用指南

在 ARMLinux 上实现实时的目标检测功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍目标检测 Demo 运行方法和如何在更新模型/输入/输出处理下，保证目标检测 Demo 仍可继续运行。

## 如何运行目标检测 Demo

### 环境准备

* 准备 ARMLiunx 开发版，用于 Demo 运行。
* Paddle Lite 当前已验证 RK3399（[Ubuntu 18.04](http://www.t-firefly.com/doc/download/page/id/3.html)） 或 树莓派 3B（[Raspbian Buster with desktop](https://www.raspberrypi.org/downloads/raspbian/)），这两个软、硬件环境，其它平台用户可自行尝试；
* 支持树莓派 3B 摄像头采集图像，具体参考[树莓派 3B 摄像头安装与测试](/PaddleLite-armlinux-demo/enable-camera-on-raspberry-pi.md)
* gcc g++ opencv cmake 的安装（以下所有命令均在设备上操作）

```bash
$ sudo apt-get update
$ sudo apt-get install gcc g++ make wget unzip libopencv-dev pkg-config
$ wget https://www.cmake.org/files/v3.10/cmake-3.10.3.tar.gz
$ tar -zxvf cmake-3.10.3.tar.gz
$ cd cmake-3.10.3
$ ./configure
$ make
$ sudo make install
```

### 部署步骤

1. 目标检测 Demo 位于 `Paddle-Lite-Demo/PaddleLite-armlinux-demo/object_detection_demo` 目录
2. 终端中执行 `download_models_and_libs.sh` 脚本自动下载模型和 Paddle Lite 预测库

```shell
cd PaddleLite-armlinux-demo          # 1. 终端中进入 Paddle-Lite-Demo\PaddleLite-armlinux-demo
sh download_models_and_libs.sh       # 2. 执行脚本下载依赖项 （需要联网）
```

下载完成后会出现提示： `Download successful!`
3. 执行用例(保证 ARMLinux 环境准备完成)

```shell
cd object_detection_demo    # 1. 终端中进入
sh run.sh                   # 2. 执行脚本编译并执行物体检测 demo，输出预测数据和运行时间
```

Demo 结果如下:
<p align="center">
<img src="https://user-images.githubusercontent.com/50474132/82852558-da228580-9f35-11ea-837c-e4d71066da57.png">
</p>

## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 ARMLinux 预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.armLinux.xxx.gcc/inference_lite_lib.armLinux.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-armlinux-demo/Paddle-Lite/include`
        * armv7hf
          将生成的 `build.lite.armLinux.armv7hf.gcc/inference_lite_lib.android.armv7hf/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-armlinux-demo/Paddle-Lite/libs/armv7hf/libpaddle_light_api_shared.so`
        * armv8
          将生成的 `build.lite.armLinux.armv8.gcc/inference_lite_lib.armLinux.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-armlinux-demo/Paddle-Lite/libs/armv8/libpaddle_light_api_shared.so`

## Demo 内容介绍

先整体介绍下目标检测 Demo 的代码结构，然后再从 Java 和 C++ 两部分简要的介绍 Demo 每部分功能.

1. `object_detection_demo.cc`： C++ 预测代码

```shell
# 位置：
Paddle-Lite-Demo/PaddleLite-armlinux-demo/object_detection_demo/object_detection_demo.cc
```

2. `model.nb` : 模型文件 (opt 工具转化后 Paddle Lite 模型), `pascalvoc_label_list`：训练模型时的 `labels` 文件

```shell
# 位置：
object_detection_demo/models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb
object_detection_demo/labels/pascalvoc_label_list
```

3. `libpaddle_lite_api_shared.so`：Paddle Lite C++ 预测动态库

```shell
# 位置
../PaddleLite/libs/armv8/libpaddle_lite_api_shared.so
../PaddleLite/libs/armv7hf/libpaddle_lite_api_shared.so
# 如果要替换动态库 so，则将新的动态库 so 更新到此目录下
```

4. `CMakeLists.txt` : C++ 预测代码的编译脚本，用于生成可执行文件

```shell
# 位置
object_detection_demo/CMakeLists.txt
# 如果有cmake 编译选项更新，可以在 CMakeLists.txt 进行修改即可，默认编译 armv8 可执行文件；
```

5. `run.sh` : 运行脚本，包含可执行文件生成和执行

```shell
# 位置
object_detection_demo/run.sh
# 可通过修改 TARGET_ARCH_ABI 变量，生成 armv8 或 armv7hf 可执行文件
```

## 代码讲解 （使用 Paddle Lite `C++ API` 执行预测）

ARMLinux 示例基于 C++ API 开发，调用 Paddle Lite `C++s API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite C++ API ](https://paddle-lite.readthedocs.io/zh/latest/api_reference/c++_api_doc.html)。

```c++
#include <iostream>
// 引入 C++ API
#include "include/paddle_api.h"
#include "include/paddle_use_ops.h"
#include "include/paddle_use_kernels.h"

// 1. 设置 MobileConfig
MobileConfig config;
config.set_model_from_file(modelPath); // 设置 NaiveBuffer 格式模型路径
config.set_power_mode(LITE_POWER_NO_BIND); // 设置 CPU 运行模式
config.set_threads(4); // 设置工作线程数

// 2. 创建 PaddlePredictor
std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

// 3. 设置输入数据
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
}
// 如果输入是图片，则可在第三步时将预处理后的图像数据赋值给输入 Tensor

// 4. 执行预测
predictor->run();

// 5. 获取输出数据
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
std::cout << "Output shape " << output_tensor->shape()[1] << std::endl;
for (int i = 0; i < ShapeProduction(output_tensor->shape()); i += 100) {
  std::cout << "Output[" << i << "]: " << output_tensor->data<float>()[i]
            << std::endl;
}

// 例如目标检测：输出后处理，输出检测结果=
auto outputData = outputTensor->data<float>();
auto outputShape = outputTensor->shape();
int outputSize = ShapeProduction(outputShape);
for (int i = 0; i < outputSize; i += 6) {
  // Class id
  auto class_id = static_cast<int>(round(outputData[i]));
  // Confidence score
  auto score = outputData[i + 1];
  if (score < scoreThreshold_)
    continue;
  RESULT object;
  object.class_name = class_id >= 0 && class_id < labelList_.size()
                          ? labelList_[class_id]
                          : "Unknow";
  object.fill_color = class_id >= 0 && class_id < colorMap_.size()
                            ? colorMap_[class_id]
                            : cv::Scalar(0, 0, 0);
  object.score = score;
  object.x = MIN(MAX(outputData[i + 2], 0.0f), 1.0f);
  object.y = MIN(MAX(outputData[i + 3], 0.0f), 1.0f);
  object.w = MIN(MAX(outputData[i + 4] - outputData[i + 2], 0.0f), 1.0f);
  object.h = MIN(MAX(outputData[i + 5] - outputData[i + 3], 0.0f), 1.0f);
  results->push_back(object);
}
```

## 使用 Python 接口预测

1. Python 预测库编译参考[编译 ARMLinux](https://paddle-lite.readthedocs.io/zh/latest/source_compile/linux_x86_compile_arm_linux.html)，建议在开发版上编译。
2. [Paddle Lite Python API](https://paddle-lite.readthedocs.io/zh/latest/api_reference/python_api_doc.html)。
3. 代码参考，[Python 完整示例](https://paddle-lite.readthedocs.io/zh/latest/user_guides/python_demo.html)


## 如何更新模型和输入/输出预处理

### 更新模型
1. 将优化后的模型存放到目录 `object_detection_demo/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb`，则代码不需更新；否则话，需要修改 `object_detection_demo/rush.sh` 中执行命令；

以更新 ssd_mobilenet_v3 模型为例，则先将优化后的模型存放到 `object_detection_demo/models/ssd_mobilenet_v3_for_cpu/ssd_mv3.nb` 下，然后更新执行脚本

```shell
# 代码文件 `object_detection_demo/rush.sh`
# run
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./object_detection_demo ../models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb ../labels/pascalvoc_label_list ../images/dog.jpg ./result.jpg
# updata
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./object_detection_demo ../models/ssd_mobilenet_v3_for_cpu/ssd_mv3.nb ../labels/pascalvoc_label_list ../images/dog.jpg ./result.jpg
```

**注意：**
- 如果更新的模型中输入tensor、shape、和 Dtype 发生更新，需要更新 `object_detection_demo/object_detection_demo.cc` 中输入数据处理内容；

<p align="center">
<img src="https://paddlelite-data.bj.bcebos.com/doc_images/ARMLinux_demo/model_input_change.png"/>
</p>

- 如果更新的模型中输出tensor 和 Dtype 发生更新，需要更新 `object_detection_demo/object_detection_demo.cc` 中输出数据处理内容；

<p align="center">
<img src="https://paddlelite-data.bj.bcebos.com/doc_images/ARMLinux_demo/model_output_change.png"/>
</p>

- 如果需要更新 `pascalvoc_label_list` 标签文件，则需要将新的标签文件存放在目录 `object_detection_demo/labels/` 下，并参考模型更新方法更新 `object_detection_demo/rush.sh` 中执行命令；

```shell
# 代码文件 `object_detection_demo/rush.sh`
# run
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./object_detection_demo ../models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb ../labels/pascalvoc_label_list ../images/dog.jpg ./result.jpg
# updata
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./object_detection_demo ../models/ssd_mobilenet_v3_for_cpu/ssd_mv3.nb ../labels/lable.txt ../images/dog.jpg ./result.jpg
```

### 更新输入/输出预处理

1. 更新输入数据

- 将更新的图片存放在 `object_detection_demo/images/` 下；
- 更新文件 `object_detection_demo/rush.sh` 中执行命令；

以更新 `cat.jpg` 为例，则先将 `cat.jpg` 存放在 `object_detection_demo/images/` 下，然后更新脚本

```shell
# 代码文件 `object_detection_demo/rush.sh`
# run
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./object_detection_demo ../models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb ../labels/pascalvoc_label_list ../images/dog.jpg ./result.jpg
# updata
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} ./object_detection_demo ../models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb ../labels/pascalvoc_label_list ../images/cat.jpg ./result.jpg
```

2. 更新输入预处理
此处需要更新 `object_detection_demo/object_detection_demo.cc` 中的 `preprocess` 方法

<p align="center">
<img src="https://paddlelite-data.bj.bcebos.com/doc_images/ARMLinux_demo/input_change.png"/>
</p>

3. 更新输出预处理
此处需要更新 `object_detection_demo/object_detection_demo.cc` 中的 `postprocess` 方法

<p align="center">
<img src="https://paddlelite-data.bj.bcebos.com/doc_images/ARMLinux_demo/output_change.png"/>
</p>

