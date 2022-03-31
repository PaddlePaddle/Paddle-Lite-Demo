# 图像分类 C++ API Demo 使用指南

在 ARMLinux 上实现实时的图像分类功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍图像分类 Demo 运行方法和如何在更新模型/输入/输出处理下，保证图像分类 Demo 仍可继续运行。

## 如何运行图像分类 Demo

### 环境准备

* 准备 ARMLiunx 开发版，用于 Demo 运行。
* Paddle Lite 当前已验证 RK3399（[Ubuntu 18.04](http://www.t-firefly.com/doc/download/page/id/3.html)） 或 树莓派 3B（[Raspbian Buster with desktop](https://www.raspberrypi.org/downloads/raspbian/)），这两个软、硬件环境，其它平台用户可自行尝试；
* 支持树莓派 3B 摄像头采集图像，具体参考[树莓派 3B 摄像头安装与测试](https://github.com/PaddlePaddle/Paddle-Lite-Demo/blob/master/PaddleLite-armlinux-demo/enable-camera-on-raspberry-pi.md)
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

1. 图像分类 Demo 位于 `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification` 目录
2. cd `Paddle-Lite-Demo/libs` 目录，终端中执行 `download.sh` 脚本自动下载 Paddle Lite 预测库
下载完成后会出现提示： `Download successful!`
3. cd `Paddle-Lite-Demo/image_classification/assets` 目录，终端中执行 `download.sh` 脚本自动下载图像分类模型和图片及标签文件
4. cd `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification` 目录，终端中执行 `run.sh` 脚本完成和编译和运行推理(保证 ARMLinux 环境准备完成)

```shell
cd Paddle-Lite-Demo/libs          # 1. 终端中进入 Paddle-Lite-Demo\libs
sh download.sh                    # 2. 执行脚本下载依赖项 （需要联网）
cd ../image_classification/assets
sh download.sh                    # 3. 下载图像分类模型和图片及标签文件
cd ../linux/app/cxx/image_classification
sh run.sh                         # 4. 执行脚本编译并执行图像分类 demo，输出预测数据和运行时间
```

Demo 运行结果如下:

<p align="center">
<img src="https://paddlelite-demo.bj.bcebos.com/demo/image_classification/docs_img/linux/armlinux_image_classification.jpg">
</p>

## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 ARMLinux 预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.armLinux.xxx.gcc/inference_lite_lib.armLinux.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/libs/linux/include`
        * armv7hf
          将生成的 `build.lite.armLinux.armv7hf.gcc/inference_lite_lib.armLinux.armv7hf/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/libs/linux/libs/armv7hf/libpaddle_light_api_shared.so`
        * armv8
          将生成的 `build.lite.armLinux.armv8.gcc/inference_lite_lib.armLinux.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/libs/linux/libs/armv8/libpaddle_light_api_shared.so`

## Demo 内容介绍

1. `Paddle-Lite-Demo/libs/` : 存放不同端的预测库和 OpenCV 库，如 Android、iOS 等

**备注：**
  如需更新预测库，例如更新 linux CXX v8 动态库 `so`，则将新的动态库 `so` 更新到 `Paddle-Lite-Demo/libs/linux/libs/armv8` 目录

2. `Paddle-Lite-Demo/image_classification/assets/` : 存放图像分类 demo 的模型、测试图片、标签文件

3. `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/image_classification.cc` : 图像分类的预测代码
     - `pre_process(...)` : 完成图像分类的预处理功能
     - `post_process(...)` : 完成图像分类的后处理功能
     - `process(...)` : 完成图像分类的预测全流程功能
     - `run_model(...)` : 完成预测器初始化和输入数据源选取功能
     - `load_labels(...)` : 完成标签文件读取功能
     - `neon_mean_scale(...)` : 完成图像数据赋值给Tensor的加速处理功能

4. `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/CMakeLists.txt` :  CMake 文件，约束可执行文件的编译方法

5. `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh` : 用于可执行文件的编译和运行

```shell
 # 位置
 Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh # 脚本默认编译 armv8 可执行文件
 # 如果要编译 armv7hf 可执行文件，可以将 build.sh 脚本中的 ARM_ABI 变量改为 armv7hf 即可
 # run.sh 脚本中可执行文件的参数含义：
 ./image_classification ${MODELS_DIR}/mobilenet_v1_for_cpu/model.nb ${LABELS_DIR}/labels.txt 3 ${IMAGES_DIR}/tabby_cat.jpg ./result.jpg

 第一个参数：image_classification 可执行文件，属于必选项
 第二个参数：${MODELS_DIR}//models/mobilenet_v1_for_cpu/model.nb 优化后的分类模型文件，属于必选项
 第三个参数：${LABELS_DIR}/labels.txt  label 文件，属于必选项
 第四个参数：3 top-k 大小，属于可选项，默认是 1
 第五个参数：${IMAGES_DIR}/tabby_cat.jpg  测试图片，属于可选项。如果不提供输入图片，默认从摄像头获取输入数据
 第六个参数：./result.jpg 保存图片位置及名称，属于可选项，默认是空
```

## 代码讲解 （使用 Paddle Lite `C++ API` 执行预测）

Linux 示例基于 C++ API 开发，调用 Paddle Lite `C++s API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite C++ API ](https://paddle-lite.readthedocs.io/zh/latest/api_reference/c++_api_doc.html)。

```c++
#include <iostream>
// 引入 C++ API
#include "include/paddle_api.h"
#include "include/paddle_use_ops.h"
#include "include/paddle_use_kernels.h"

// 1. 设置 MobileConfig
MobileConfig config;
config.set_model_from_file(<modelPath>); // 设置 NaiveBuffer 格式模型路径
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

// 例如图像分类：输出后处理，输出检测结果=
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

## 更新模型、输入和输出预处理

### 更新模型

1. 将优化后的模型存放到目录 `Paddle-Lite-Demo/image_classification/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `mobilenet_v1_for_cpu/model.nb`，则代码不需更新；否则话，需要修改 `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh` 脚本。

例子：假设更新 mobilenet_v2 模型为例，则先将优化后的模型存放到 `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh` 下，然后更新脚本

```shell
# path: Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh
# old
./image_classification ${MODELS_DIR}/mobilenet_v1_for_cpu/model.nb ${LABELS_DIR}/labels.txt 3 ${IMAGES_DIR}/tabby_cat.jpg ./result.jpg

# now
./image_classification ${MODELS_DIR}/mobilenet_v2_for_cpu/mv2.nb ${LABELS_DIR}/labels.txt 3 ${IMAGES_DIR}/tabby_cat.jpg ./result.jpg
```

**注意：**
-  如果更新模型的输入/输出 Tensor 个数、shape 和 Dtype 发生更新，需要更新文件 `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/image_classification.cc` 的 `pre_process` 预处理和 `pre_process` 后处理代码即可。

- 如果需要更新 `labels.txt` 标签文件，则需要将新的标签文件存放在目录 `Paddle-Lite-Demo/image_classification/assets/labels/` 下，并更新 `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh` 脚本。

```shell
# path: Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh
# old
./image_classification ${MODELS_DIR}/mobilenet_v1_for_cpu/model.nb ${LABELS_DIR}/labels.txt 3 ${IMAGES_DIR}/tabby_cat.jpg ./result.jpg

# now
./image_classification ${MODELS_DIR}/mobilenet_v1_for_cpu/model.nb ${LABELS_DIR}/labels_new.txt 3 ${IMAGES_DIR}/tabby_cat.jpg ./result.jpg

```

### 更新输入/输出预处理
1. 更新输入数据

- 将更新的图片存放在 `Paddle-Lite-Demo/image_classification/assets/images/` 下；
- 更新文件 `Paddle-Lite-Demo/image_classification/linux/app/image_classification/run.sh` 脚本

以更新 `dog.jpg` 为例，则先将 `dog.jpg` 存放在 `Paddle-Lite-Demo/image_classification/assets/images/` 下，然后更新脚本

```shell
# path: Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/run.sh
# old
./image_classification ${MODELS_DIR}/mobilenet_v1_for_cpu/model.nb ${LABELS_DIR}/labels.txt 3 ${IMAGES_DIR}/tabby_cat.jpg ./result.jpg

# now
./image_classification ${MODELS_DIR}/mobilenet_v1_for_cpu/model.nb ${LABELS_DIR}/labels.txt 3 ${IMAGES_DIR}/dog.jpg ./result.jpg

```
**注意：**
  本 Demo 支持摄像头输入，如果想更新输入，可通过摄像头进行获取输入数据。


2. 更新输入预处理
此处需要更新 `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/image_classification.cc` 的 `pre_process` 预处理实现就行。

3. 更新输出预处理
此处需要更新 `Paddle-Lite-Demo/image_classification/linux/app/cxx/image_classification/image_classification.cc` 的 `post_process` 后处理代码实现就行。

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。
