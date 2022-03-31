# OOCR 文字识别 Demo 使用指南
在 Android Shell 环境下，实现实时的 OCR 文字识别功能。此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍 OCR 文字识别 Demo 的运行方法和如何在更新模型/输入/输出处理下，保证 Demo 仍可继续运行。

## 如何运行 OCR 文字识别 Demo

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

1. OCR 文字识别 Demo 位于 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ppocr_demo` 目录
2. cd `Paddle-Lite-Demo/libs` 目录，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库
3. cd `Paddle-Lite-Demo/ocr/assets` 目录，运行 `download.sh` 脚本，下载OPT 优化后模型、测试图片和标签文件
4. cd `Paddle-Lite-Demo/ocr/linux/shell/cxx/ppocr_demo` 目录，运行 `build.sh` 脚本， 完成可执行文件的编译
5. 在当前目录运行 `run.sh` 脚本，进行推理，推理结果将会在当前窗口显示和结果写回图片（在当前目录可找到）。其效果如下图所示：
<p align="center"><img width="350" height="500"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/linux/run_app.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="350" height="500"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/linux/run_result.jpg"/></p>

**注意：**
  这些操作均在 ARMLinux 设备上运行，且保证设备可联网（用于预测库和模型下载）

```shell
cd Paddle-Lite-Demo/libs
# 下载所需要的 Paddle Lite 预测库
sh download.sh
cd ../ocr/assets
# 下载OPT 优化后模型、测试图片、标签文件及 config 文件
sh download.sh
cd ../linux/shell/cxx/ppocr_demo
# 完成可执行文件的编译, 默认编译 V8 可执行文件； 如需 V7 可执行文件，可修改 build.sh 脚本中 ARM_ABI 变量即可
sh build.sh
# 进行推理，推理结果将会在当前窗口显示，并将结果写回图片（在当前目录可找到）
sh run.sh
```

## 如何更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 Android 预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.armLinux.xxx.gcc/inference_lite_lib.armLinux.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-armlinux-demo/Paddle-Lite/include`
        * armv7hf
          将生成的 `build.lite.armLinux.armv7hf.gcc/inference_lite_lib.armLinux.armv7hf/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/libs/linux/cxx/libs/armv7hf/libpaddle_light_api_shared.so`
        * armv8
          将生成的 `build.lite.armLinux.armv8.gcc/inference_lite_lib.armLinux.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/libs/linux/cxx/libs/armv8/libpaddle_lite_api_shared.so`

**注意：**
如果预测库有版本升级，建议同步更新 OPT 优化后的模型。例如，预测库升级至 2.10—rc 版本，需要做以下操作：

```shell
# 下载 PaddleOCR V2.0 版本的中英文 inference 模型
wget  https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_slim_infer.tar && tar xf  ch_ppocr_mobile_v2.0_det_slim_infer.tar
wget  https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_rec_slim_infer.tar && tar xf  ch_ppocr_mobile_v2.0_rec_slim_infer.tar
wget  https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_cls_slim_infer.tar && tar xf  ch_ppocr_mobile_v2.0_cls_slim_infer.tar
# 获取 2.10 版本的 MAC 系统的 OPT 工具
wget https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.10-rc/opt_mac
# 转换 V2.0 检测模型
./opt --model_file=./ch_ppocr_mobile_v2.0_det_slim_infer/inference.pdmodel  --param_file=./ch_ppocr_mobile_v2.0_det_slim_infer/inference.pdiparams  --optimize_out=./ch_ppocr_mobile_v2.0_det_slim_opt --valid_targets=arm  --optimize_out_type=naive_buffer
# 转换 V2.0 识别模型
./opt --model_file=./ch_ppocr_mobile_v2.0_rec_slim_infer/inference.pdmodel  --param_file=./ch_ppocr_mobile_v2.0_rec_slim_infer/inference.pdiparams  --optimize_out=./ch_ppocr_mobile_v2.0_rec_slim_opt --valid_targets=arm  --optimize_out_type=naive_buffer
# 转换 V2.0 方向分类器模型
./opt --model_file=./ch_ppocr_mobile_v2.0_cls_slim_infer/inference.pdmodel  --param_file=./ch_ppocr_mobile_v2.0_cls_slim_infer/inference.pdiparams  --optimize_out=./ch_ppocr_mobile_v2.0_cls_slim_opt --valid_targets=arm  --optimize_out_type=naive_buffer
```

## Demo 代码介绍

Demo 的整体目录结构如下图所示：

<p align="center"><img src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/android/predict_android_shell.jpg"/></p>

1. `Paddle-Lite-Demo/libs/` : 存放不同端的预测库和OpenCL库，如android、iOS等

**备注：**
  如需更新预测库，例如更新 linux CXX v8 动态库 `so`，则将新的动态库 `so` 更新到 `Paddle-Lite-Demo/libs/linux/cxx/libs/armv8` 目录

2. `Paddle-Lite-Demo/ocr/assets/` : 存放 OCR demo 的模型、测试图片、标签文件及 config 文件

**备注：**

 - `Paddle-Lite-Demo/ocr/assets/labels/ppocr_keys_v1.txt` 是中文字典文件，如果使用的 nb 模型是英文数字或其他语言的模型，需要更换为对应语言的字典.
 - 其他语言的字典文件，可从 PaddleOCR 仓库下载：https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.3/ppocr/utils

3. `Paddle-Lite-Demo/ocr/android/shell/ppocr_demo/src` : 存放预测代码
    - `cls_process.cc` : 方向分类器的推理全流程，包含预处理、预测和后处理三部分
    - `rec_process.cc` : 识别模型 CRNN 的推理全流程，包含预处理、预测和后处理三部分
    - `det_process.cc` : 检测模型 CRNN 的推理全流程，包含预处理、预测和后处理三部分
    - `det_post_process` : 检测模型 DB 的后处理文件
    - `pipeline.cc` : OCR 文字识别 Demo 推理全流程代码
    - `CMakeLists.txt` : 预测代码的 MakeFile 文件

4. `Paddle-Lite-Demo/ocr/liunx/shell/cxx/ppocr_demo/build.sh` : 用于可执行文件的编译

```shell
# 位置
Paddle-Lite-Demo/ocr/linux/shell/cxx/ppocr_demo/build.sh # 脚本默认编译 armv8 可执行文件
# 如果要编译 armv7 可执行文件，可以将 build.sh 脚本中的 ARM_ABI 变量改为 armv7hf 即可
```

7. `Paddle-Lite-Demo/ocr/linux/shell/cxx/ppocr_demo/run.sh` : 预测脚本，获取返回结果

```shell
# 位置
Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_demo/run.sh
# 脚本中可执行文件的参数含义：
"./ppocr_demo \
./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
./images/test.jpg \
./test_img_result.jpg \
./labels/ppocr_keys_v1.txt \
./config.txt"

第一个参数：ppocr_demo 可执行文件
第二个参数：./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb 优化后的检测模型文件
第三个参数：./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb 优化后的识别模型文件
第四个参数：./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb 优化后的文字方向分类器模型文件
第五个参数：./images/test.jpg  测试图片
第六个参数：./test_img_result.jpg  结果保存文件
第七个参数：./labels/ppocr_keys_v1.txt  label 文件，中文字典文件
第八个参数：./config.txt  配置文件，模型的超参数配置文件，包含了检测器、分类器的超参数
# config.txt 具体参数 List：
max_side_len  960         # 输入图像长宽大于 960 时，等比例缩放图像，使得图像最长边为 960
det_db_thresh  0.3        # 用于过滤 DB 预测的二值化图像，设置为 0.3 对结果影响不明显
det_db_box_thresh  0.5    # DB 后处理过滤 box 的阈值，如果检测存在漏框情况，可酌情减小
det_db_unclip_ratio  1.6  # 表示文本框的紧致程度，越小则文本框更靠近文本
use_direction_classify  0  # 是否使用方向分类器，0 表示不使用，1 表示使用
```

## 代码讲解 （使用 Paddle Lite `C++ API` 执行预测）

该示例基于 C++ API 开发，调用 Paddle Lite `C++s API` 包括以下五步。
更详细的 `API` 描述参考：[Paddle Lite C++ API ](https://paddle-lite.readthedocs.io/zh/latest/api_reference/c++_api_doc.html)。

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
```

## 如何更新模型、输入/输出预处理

### 更新模型

1. 将优化后的新模型存放到目录 `Paddle-Lite-Demo/ocr/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，则 `run.sh` 脚本不需更新；否则话，需要修改 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_demo/run.sh` 中执行命令；

以将检测模型更新为例，则先将优化后的模型存放到 `Paddle-Lite-Demo/ocr/assetss/models/ssd_mv3.nb` 下，然后更新执行脚本

```shell
# 代码文件 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ppocr_demo/run.sh`
# old
"./ppocr_demo \
./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
./images/test.jpg \
./test_img_result.jpg \
./labels/ppocr_keys_v1.txt \
./config.txt"
# update
"./ppocr_demo \
./models/ssd_mv3.nb \
./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
./images/test.jpg \
./test_img_result.jpg \
./labels/ppocr_keys_v1.txt \
./config.txt"
```

**注意：**

- 如果更新模型中的输入 Tensor、Shape、和 Dtype 发生更新:

  - 更新文字方向分类器模型，则需要更新 `ppocr_demo/src/cls_process.cc` 中 `ClsPredictor::Preprocss` 函数
  - 更新检测模型，则需要更新 `ppocr_demo/src/det_process.cc` 中 `DetPredictor::Preprocss` 函数
  - 更新识别器模型，则需要更新 `ppocr_demo/src/rec_process.cc` 中 `RecPredictor::Preprocss` 函数

- 如果更新模型中的输出 Tensor 和 Dtype 发生更新:

  - 更新文字方向分类器模型，则需要更新 `ppocr_demo/src/cls_process.cc` 中 `ClsPredictor::Postprocss` 函数
  - 更新检测模型，则需要更新 `ppocr_demo/src/det_process.cc` 中 `DetPredictor::Postprocss` 函数
  - 更新识别器模型，则需要更新 `ppocr_demo/src/rec_process.cc` 中 `RecPredictor::Postprocss` 函数


- 如果需要更新 `ppocr_keys_v1.txt` 标签文件，则需要将新的标签文件存放在目录 `Paddle-Lite-Demo/ocr/assets/labels/` 下，并参考模型更新 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_demo/run.sh` 中执行命令；

```shell
# 代码文件 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ppocr_demo/run.sh`
# old
"./ppocr_demo \
./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
./images/test.jpg \
./test_img_result.jpg \
./labels/ppocr_keys_v1.txt \
./config.txt"
# update
"./ppocr_demo \
./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
./images/test.jpg \
./test_img_result.jpg \
./labels/new_labels.txt \
./config.txt"
```

### 更新输入/输出预处理

1. 更新输入数据

- 将更新的图片存放在 `Paddle-Lite-Demo/ocr/assets/images/` 下；
- 更新文件 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_demo/run.sh` 中执行命令；

以更新 `new_pics.jpg` 为例，则先将 `new_pics.jpg` 存放在 `Paddle-Lite-Demo/ocr/assets/images/` 下，然后更新脚本

```shell
# 代码文件 `Paddle-Lite-Demo/ocr/assets/images/run.sh`
## old
"./ppocr_demo \
./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
./images/test.jpg \
./test_img_result.jpg \
./labels/ppocr_keys_v1.txt \
./config.txt"
# update
"./ppocr_demo \
./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
./images/new_pic.jpg \
./test_img_result.jpg \
./labels/ppocr_keys_v1.txt \
./config.txt"
```

2. 更新输入预处理
  - 更新文字方向分类器模型，则需要更新 `ocr_demo/src/cls_process.cc` 中 `ClsPredictor::Preprocss` 函数
  - 更新检测模型，则需要更新 `ocr_demo/src/det_process.cc` 中 `DetPredictor::Preprocss` 函数
  - 更新识别器模型，则需要更新 `ocr_demo/src/rec_process.cc` 中 `RecPredictor::Preprocss` 函数

3. 更新输出预处理

  - 更新文字方向分类器模型，则需要更新 `ocr_demo/src/cls_process.cc` 中 `ClsPredictor::Postprocss` 函数
  - 更新检测模型，则需要更新 `ocr_demo/src/det_process.cc` 中 `DetPredictor::Postprocss` 函数
  - 更新识别器模型，则需要更新 `ocr_demo/src/rec_process.cc` 中 `RecPredictor::Postprocss` 函数

## OCR 文字识别 Demo 工程详解

OCR 文字识别 Demo 由三个模型一起完成 OCR 文字识别功能，对输入图片先通过 `ch_ppocr_mobile_v2.0_det_slim_opt.nb` 模型做检测处理，然后通过 `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` 模型做文字方向分类处理，最后通过 `ch_ppocr_mobile_v2.0_rec_slim_opt.nb` 模型完成文字识别处理。

1. `pipeline.cc` : OCR 文字识别 Demo 预测全流程代码
  该文件完成了三个模型串行推理的全流程控制处理，包含整个处理过程的调度处理。

  - `Pipeline::Pipeline(...)` 方法完成调用三个模型类构造函数，完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `Pipeline::Process(...)` 方法用于完成这三个模型串行推理的全流程控制处理
  
2. `cls_process.cc` 方向分类器的预测文件
  该文件完成了方向分类器的预处理、预测和后处理过程

  - `ClsPredictor::ClsPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `ClsPredictor::Preprocess()` 方法用于模型的预处理
  - `ClsPredictor::Postprocess()` 方法用于模型的后处理

3. `rec_process.cc` 识别模型 CRNN 的预测文件
  该文件完成了识别模型 CRNN 的预处理、预测和后处理过程

  - `RecPredictor::RecPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `RecPredictor::Preprocess()` 方法用于模型的预处理
  - `RecPredictor::Postprocess()` 方法用于模型的后处理

4. `det_process.cc` 检测模型 DB 的预测文件
  该文件完成了检测模型 DB 的预处理、预测和后处理过程

  - `DetPredictor::DetPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `DetPredictor::Preprocess()` 方法用于模型的预处理
  - `DetPredictor::Postprocess()` 方法用于模型的后处理

5. `db_post_process` 检测模型 DB 的后处理函数，包含 clipper 库的调用
  该文件完成了检测模型 DB 的第三方库调用和其他后处理方法实现

  - `std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(...)` 方法从 Bitmap 图中获取检测框
  - `std::vector<std::vector<std::vector<int>>> FilterTagDetRes(...)` 方法根据识别结果获取目标框位置

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。
