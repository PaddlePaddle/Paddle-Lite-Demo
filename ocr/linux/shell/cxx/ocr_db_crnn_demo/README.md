# OOCR 文字识别 Demo 使用指南
在 Android Shell 环境下，实现实时的 OCR 文字识别功能。此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍 OCR 文字识别 Demo 的运行方法和如何在更新模型/输入/输出处理下，保证 Demo 仍可继续运行。

## 如何运行 OCR 文字识别 Demo

### 环境准备

1. 在本地环境安装好 CMAKE 编译工具，并在 [Android NDK 官网](https://developer.android.google.cn/ndk/downloads)下载当前系统的某个版本的 NDK 软件包。例如，在 Mac 上开发，需要在 Android NDK 官网下载 Mac 平台的 NDK 软件包。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`
3. 电脑上安装 adb 工具，用于调试。 adb安装方式如下：

    3.1. Mac 电脑安装 ADB:

    ```shell
    brew cask install android-platform-tools
    ```

    3.2. Linux 安装 ADB

    ```shell
    sudo apt update
    sudo apt install -y wget adb
    ```

    3.3. Window 安装 ADB

    win 上安装需要去谷歌的安卓平台下载 ADB 软件包进行安装：[链接](https://developer.android.com/studio)

    打开终端，手机连接电脑，在终端中输入

    ```shell
    adb devices
    ```

    如果有 device 输出，则表示安装成功。

    ```shell
       List of devices attached
       744be294    device
    ```

### 部署步骤

1. OCR 文字识别 Demo 位于 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_db_crnn_demo` 目录
2. cd `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_db_crnn_demo` 目录，运行 `prepare.sh` 脚本，下载所需要的 Paddle Lite 预测库、OPT 优化后模型，并完成可执行文件的编译
3. 在当前目录运行 `run.sh` 脚本，进行推理，推理结果将会在当前窗口显示和结果写回图片（在当前目录可找到）。其效果如下图所示：
<p align="center"><img width="350" height="500"  src="https://paddlelite-demo.bj.bcebos.com/doc/ocr/linux/shell/run_app.jpg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="350" height="500"  src="https://paddlelite-demo.bj.bcebos.com/doc/ocr/linux/shell/run_result.jpg"/></p>

```shell
cd Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_db_crnn_demo
# 下载所需要的 Paddle Lite 预测库、OPT 优化后模型，并完成可执行文件的编译
sh prepare.sh
# 进行推理，推理结果将会在当前窗口显示，并将结果写回图片（在当前目录可找到）
sh run.sh
```

## 如何更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 Android 预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.android.xxx.clang/inference_lite_lib.android.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_db_crnn_demo/PaddleLite/cxx/include`
        * armeabi-v7a
          将生成的 `build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_db_crnn_demo/PaddleLite/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so`
        * arm64-v8a
          将生成的 `build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/ocr/linux/shell/cxx/ocr_db_crnn_demo/PaddleLite/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so`

注意：
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

<p align="center"><img src="https://paddlelite-demo.bj.bcebos.com/doc/ocr/linux/shell/predict.jpg"/></p>

1. `ocr_db_crnn_code` : 存放预测代码
    - `cls_process.cc` : 方向分类器的预处理文件
    - `crnn_process.cc` : 识别模型 CRNN 的预处理和后处理文件
    - `db_post_process` : 检测模型 DB 的后处理文件
    - `ocr_db_crnn.cc` : OCR 文字识别 Demo 预测处理代码
    - `MakeFile` : 预测代码的 MakeFile 文件

```shell
# 位置：
ocr_db_crnn_demo/ocr_db_crnn_code
```

3. `images/` : 测试图片目录，用于存放测试图片

```shell
# 位置：
ocr_db_crnn_demo/images
```

4. `models/` : 模型文件目录 (存放 opt 工具转化后 Paddle Lite 模型), `labels/ppocr_keys_v1.txt`：训练模型时的 `labels` 文件

```shell
# 位置：
ocr_db_crnn_demo/models/
ocr_db_crnn_demo/labels/ppocr_keys_v1.txt
# ppocr_keys_v1.txt是中文字典文件，如果使用的 nb 模型是英文数字或其他语言的模型，需要更换为对应语言的字典.
# 其他语言的字典文件，可从 PaddleOCR 仓库下载：https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.3/ppocr/utils
# 其他语言的字典内容：
dict/french_dict.txt     # 法语字典
dict/german_dict.txt     # 德语字典
ic15_dict.txt       # 英文字典
dict/japan_dict.txt      # 日语字典
dict/korean_dict.txt     # 韩语字典
ppocr_keys_v1.txt   # 中文字典
```

5. `libpaddle_lite_api_shared.so`：Paddle Lite C++ 预测库

```shell
# 位置
ocr_db_crnn_demo/PaddleLite/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so
ocr_db_crnn_demo/PaddleLite/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so
# 如果要替换动态库 so，则将新的动态库 so 更新到此目录下
```

6. `prepare.sh` : 下载 Paddle Lite 预测库、OPT 优化后模型脚本，并完成可执行文件的编译

```shell
# 位置
ocr_db_crnn_demo/prepare.sh # 脚本默认编译 armv8 可执行文件
# 如果要编译 armv7 可执行文件，可以将 prepare.sh 脚本中的 ARM_ABI 变量改为 armeabi-v7a 即可
```

7. `run.sh` : 预测脚本，获取返回结果

```shell
# 位置
ocr_db_crnn_demo/run.sh
# 脚本中可执行文件的参数含义：
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt"

第一个参数：ocr_db_crnn 可执行文件
第二个参数：./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb 优化后的检测模型文件
第三个参数：./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb 优化后的识别模型文件
第四个参数：./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb 优化后的文字方向分类器模型文件
第五个参数：./images/test.jpg  测试图片
第六个参数：./test_img_result.jpg  结果保存文件
第七个参数：./labels/ppocr_keys_v1.txt  label 文件，中文字典文件
```

8. `config.txt` : 超参数配置文件，包含了检测器、分类器的超参数

```shell
# 位置
ocr_db_crnn_demo/config.txt
# 具体参数 List：
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

1. 将优化后的新模型存放到目录 `ocr_db_crnn_demo/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，则 `run.sh` 脚本不需更新；否则话，需要修改 `ocr_db_crnn_demo/run.sh` 中执行命令；

以将检测模型更新为例，则先将优化后的模型存放到 `ocr_db_crnn_demo/models/ssd_mv3.nb` 下，然后更新执行脚本

```shell
# 代码文件 `ocr_db_crnn_demo/run.sh`
# old
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt"
# update
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ssd_mv3.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt"
```

**注意：**
- 如果更新模型中的输入 tensor、shape、和 Dtype 发生更新，需要更新 `ocr_db_crnn_demo/ocr_db_crnn_code/ocr_db_crnn.cc` 中各模型的输入 tensor 数据处理；

- 如果更新模型中的输出 tensor 和 Dtype 发生更新，需要更新 `ocr_db_crnn_demo/ocr_db_crnn_code/ocr_db_crnn.cc` 中各模型的输出 tensor 处理；


- 如果需要更新 `ppocr_keys_v1.txt` 标签文件，则需要将新的标签文件存放在目录 `ocr_db_crnn_demo/labels/` 下，并参考模型更新方法更新 `ocr_db_crnn_demo/rush.sh` 中执行命令；

```shell
# 代码文件 `ocr_db_crnn_demo/run.sh`
# old
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt"
# update
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/new_labels.txt"
```

### 更新输入/输出预处理

1. 更新输入数据

- 将更新的图片存放在 `ocr_db_crnn_demo/images/` 下；
- 更新文件 `ocr_db_crnn_demo/rush.sh` 中执行命令；

以更新 `new_pics.jpg` 为例，则先将 `new_pics.jpg` 存放在 `ocr_db_crnn_demo/images/` 下，然后更新脚本

```shell
# 代码文件 `ocr_db_crnn_demo/rush.sh`
## old
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt"
# update
adb shell "cd ${ocr_demo_path} \
           && chmod +x ./ocr_db_crnn \
           && export LD_LIBRARY_PATH=${ocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ocr_db_crnn \
                ./models/ch_ppocr_mobile_v2.0_det_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_rec_slim_opt.nb \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/new_pics.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt"
```

2. 更新输入预处理

- 更新方向分类器的输入预处理
  更新 `ocr_db_crnn_demo/ocr_db_crnn_code/cls_process.cc` 中的 `ClsResizeImg` 方法即可
- 识别模型 CRNN 的输入预处理
  更新 `ocr_db_crnn_demo/ocr_db_crnn_code/crnn_process.cc` 中的 `CrnnResizeImg` 方法即可
- 检测模型 DB 的输入预处理
  更新 `ocr_db_crnn_demo/ocr_db_crnn_code/ocr_db_crnn.cc` 中的 `DetResizeImg` 方法即可

3. 更新输出预处理

- 更新方向分类器的输出预处理
 更新 `ocr_db_crnn_demo/ocr_db_crnn_code/ocr_db_crnn.cc` 中的 `RunClsModel` 方法中 `std::unique_ptr<const Tensor> softmax_out(std::move(predictor_cls->GetOutput(0)));` 改行代码之后实现即可
- 识别模型 CRNN 的输出预处理
  更新 `ocr_db_crnn_demo/ocr_db_crnn_code/ocr_db_crnn.cc` 中的 `RunRecModel` 方法中 `std::unique_ptr<const Tensor> output_tensor0(std::move(predictor_crnn->GetOutput(0)));` 改行代码之后实现即可
- 检测模型 DB 的输出预处理
  更新 `ocr_db_crnn_demo/ocr_db_crnn_code/ocr_db_crnn.cc` 中的 `RunDetModel` 方法中 `std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));` 改行代码之后实现即可

## OCR 文字识别 Demo 工程详解

OCR 文字识别 Demo 由三个模型一起完成 OCR 文字识别功能，对输入图片先通过 `ch_ppocr_mobile_v2.0_det_slim_opt.nb` 模型做检测处理，然后通过 `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` 模型做文字方向分类处理，最后通过 `ch_ppocr_mobile_v2.0_rec_slim_opt.nb` 模型完成文字识别处理。

1. `ocr_db_crnn.cc` : OCR 文字识别 Demo 预测处理代码
  该文件完成了三个模型串行推理的全流程控制处理，包含整个处理过程的调度处理。

  - `cv::Mat process(...)` 方法是 Demo 的主入口，完成整个预测过程的调度处理。
  - `std::shared_ptr<PaddlePredictor> loadModel(std::string model_file)` 方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `std::map<std::string, double> LoadConfigTxt(std::string config_path)` 方法用于读取超参数配置文件
  - `void RunDetModel(...)` 方法是检测模型预测处理，包含输入预处理、预测和输出预处理
  - `void RunRecModel(...)` 方法是文字分类模型和文字识别模型的预测处理，包含两个的模型串连处理、输入预处理、预测和输出预处理
  - `void RunClsModel(...)` 方法是文字分类模型预测处理，包含输入预处理、预测和输出预处理

2. `cls_process.cc` 方向分类器的预处理文件
  该文件完成了方向分类器的预处理

  - `cv::Mat ClsResizeImg(cv::Mat img)`  方法是方向分类器的预处理，完成输入图片的 resize 处理

3. `crnn_process.cc` 识别模型 CRNN 的预处理文件
  该文件完成了识别模型 CRNN 的预处理

  - `cv::Mat CrnnResizeImg(cv::Mat img, float wh_ratio)`  方法是方向分类器的预处理，完成输入图片的 resize 处理
  - `cv::Mat GetRotateCropImage(cv::Mat srcimage, std::vector<std::vector<int>> box)`  方法是方向分类器的预处理，完成输入图片的 `rotate` 和 `crop` 处理

4. `db_post_process` 检测模型 DB 的后处理文件
  该文件完成了检测模型 DB 的后处理文件，包含多个后处理操作

  - `std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(...)` 方法从 Bitmap 图中获取检测框
  - `std::vector<std::vector<std::vector<int>>> FilterTagDetRes(...)` 方法根据识别结果获取目标框位置