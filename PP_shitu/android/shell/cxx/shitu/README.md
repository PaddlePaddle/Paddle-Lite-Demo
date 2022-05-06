# PP识图 Demo 使用指南
在 Android 上实现识图功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍识图 Demo 运行方法和如何在更新模型/输入/输出处理下，保证识图 demo 仍可继续运行。

## 如何运行识图 Demo

### 环境准备

1. 在本地环境安装好 CMAKE 编译工具，并在 [Android NDK 官网](https://developer.android.google.cn/ndk/downloads)下载当前系统的某个版本的 NDK 软件包。例如，在 Mac 上开发，需要在 Android NDK 官网下载 Mac 平台的 NDK 软件包

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

1. 识图 Demo 位于 `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/shitu` 目录
2. cd `Paddle-Lite-Demo/libs` 目录，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库
3. cd `Paddle-Lite-Demo/PP_shitu/assets` 目录，运行 `download.sh` 脚本，下载 OPT 优化后的模型
4. cd `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/shitu` 目录，先在 `build.sh` 脚本中，完成 NDK_ROOT 路径设置；然后运行 `build.sh` 脚本，完成可执行文件的编译和运行。
> **注意事项：**
>> - 如果是在 Linux 主机编译，请选择 Linux 版本的 NDK 进行设置
>> - 如果是在 Mac 主机编译，请选择 Mac 版本的 NDK 进行设置；另外，同步更新 `CMakeList.txt` 里的 `CMAKE_SYSTEM_NAME` 变量，更新为 `drawn`

5. 运行结果如下所示：

```shell
./images/wu_ling.jpg:
result0: bbox[253, 275, 1146, 872], score: 0.624021, label:  五菱宏光MINI
=============benchmark summary============
ObjectDetect Preprocess:  18.509ms
ObjectDetect inference :  262.841ms
ObjectDetect Postprocess: 0.079ms
Recongnise   Preprocess:  0.418ms
Recongnise   inference:   186.239ms
Recongnise   Postprocess: 0.026ms
nms                     : 0.01ms
==========================================
``` 
输出结果保存为图片, 命名为 result/原始图片名+result.jpg , 这一部分可在 demo 函数中修改

```shell
 cd Paddle-Lite-Demo/libs
 # 下载所需要的 Paddle Lite 预测库
 sh download.sh
 cd ../PP_shitu/assets
 # 下载OPT 优化后模型、测试图片、标签文件
 sh download.sh
 cd ../android/app/shell/cxx/shitu
 # 更新 NDK_ROOT 路径，然后完成可执行文件的编译和运行
 sh build.sh
 # CMakeList.txt 里的 System 默认设置是linux；如果在Mac 运行，则需将 CMAKE_SYSTEM_NAME 变量设置为 drawn
```

## 如何更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
  * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 Android 预测库
  * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/libs/android/cxx/include`
        * armeabi-v7a
          将生成的 `build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/libs/android/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so`
        * arm64-v8a
          将生成的 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/libs/android/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so`

## Demo 代码介绍

1. `Paddle-Lite-Demo/libs/` : 存放不同端的预测库和 OpenCV 库，如 Android、iOS 等

**备注：**
  如需更新预测库，例如更新 Android CXX v8 动态库 `so`，则将新的动态库 `so` 更新到 `Paddle-Lite-Demo/libs/android/cxx/libs/arm64-v8a` 目录

2. `Paddle-Lite-Demo/PP_shitu/assets/` : 存放图像分类 demo 的模型、测试图片、标签文件

3. `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/PP_shitu/src` : 图像分类的预测代码
     - `object_detector.cc` : 完成目标的检查
     - `recognition.cc` : 完成目标框的识图
     - `utils.cc` : 公共基础函数
     - `pipeline.cc` : 串联检测与识图

4. `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/PP_shitu/CMakeLists.txt` :  CMake 文件，约束可执行文件的编译方法

5. `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/shitu/build.sh` : 用于可执行文件的编译和运行

```shell
 # 位置
 Paddle-Lite-Demo/PP_shitu/android/shell/cxx/shitu/build.sh # 脚本默认编译 armv7 可执行文件
 # 如果要编译 armv8 可执行文件，可以将 build.sh 脚本中的 ARM_ABI 变量改为 arm64i-v8a 即可
 # build.sh 中包含了可执行文件的编译和运行功能，其中运行是调用run.sh 脚本进行完成
 # run.sh 脚本中可执行文件的参数含义：
adb shell "cd ${ADB_DIR} \
           && chmod +x ./shitu \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./shitu  \
              ./images \
              ./models/mainbody_PPLCNet_x2_5_640_quant_v1.0_lite.nb \
              ./models/general_PPLCNet_x2_5_quant_v1.0_lite.nb \
              ./labels/label.txt \
              4 0 1"

 第一个参数：shitu 可执行文件，属于必选项
 第二个参数：./images 预测图片文件夹，属于必选项
 第三个参数：./models/mainbody_PPLCNet_x2_5_640_quant_v1.0_lite.nb 目标检查模型，属于必选项
 第四个参数：./models/general_PPLCNet_x2_5_quant_v1.0_lite.nb 识图模型，属于必选项
 第五个参数：./labels/label.txt 标签文件，属于必选项
 第六个参数：cpu线程数，属于可选项
 第七个参数：warmup次数，属于可选项
 第八个参数：repeat次数，属于可选项
```

## 代码讲解 （使用 Paddle Lite `C++ API` 执行预测）

Android 示例基于 C++ API 开发，调用 Paddle Lite `C++s API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite C++ API ](https://paddle-lite.readthedocs.io/zh/latest/api_reference/c++_api_doc.html)。

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
```

## 更新模型、输入和输出预处理

### 更新模型

1. 将优化后的模型存放到目录 `Paddle-Lite-Demo/PP_shitu/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，则代码不需更新；否则话，需要修改 `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/shitu/run.sh` 脚本。

### 更新输入/输出预处理
1. 如果更新模型的输入/输出 Tensor 个数、shape 和 Dtype 发生更新，需要更新文件 `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/shitu/src/object_detector.cc``recognition.cc` 的 `preprocess` 预处理和 `postprocess` 后处理代码即可。

2. 如果需要修改目标检测的参数，可以修改`Paddle-Lite-Demo/PP_shitu/android/shell/cxx/shitu/src/object_detector.h` 180行处构造函数，识图参数修改同理，参见`recognition.h`。

```
    // Init preprocess param
    // Normalisze
    pre_param_.mean = std::vector<float>({0.485, 0.456, 0.406});
    pre_param_.std = std::vector<float>({0.229, 0.224, 0.225});
    pre_param_.is_scale = true;
    // resize
    pre_param_.interp = 2;
    pre_param_.keep_ratio = false;
    pre_param_.target_size =
        std::vector<int>({det_input_shape[2], det_input_shape[3]});
    // Pad
    pre_param_.stride = 0;
    // TopDownEvalAffine
    pre_param_.trainsize =
        std::vector<int>({det_input_shape[2], det_input_shape[3]});
    // op
    preprocess_op_func_ = std::vector<std::string>(
        {"DetResize", "DetNormalizeImage", "DetPermute"});
```

3. 如果需要更新 `labels.txt` 标签文件，则需要将新的标签文件存放在目录 `Paddle-Lite-Demo/PP_shitu/assets/labels/` 下，并更新 `Paddle-Lite-Demo/PP_shitu/android/shell/cxx/PP_shitu/run.sh` 脚本。

4. 如果需要增加预测图片，直接将图片拷贝到images文件夹即可，不用修改代码。
