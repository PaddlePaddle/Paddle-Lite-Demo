# 图像分类 Demo 使用指南
在 Android 上实现实时的图像分类功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍图像分类 Demo 运行方法和如何在更新模型/输入/输出处理下，保证图像分类 demo 仍可继续运行。

## 如何运行图像分类 Demo

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

1. 图像分类 Demo 位于 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification` 目录
2. cd `Paddle-Lite-Demo/libs` 目录，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库
3. cd `Paddle-Lite-Demo/image_classification/assets` 目录，运行 `download.sh` 脚本，下载OPT 优化后模型、测试图片和标签文件
4. cd `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification` 目录，运行 `build.sh` 脚本，完成可执行文件的编译和运行。
5. 运行结果如下所示：

```shell
======= benchmark summary =======
input_shape(s) (NCHW): {1, 3, 224, 224}
model_dir:./models/mobilenet_v1_for_cpu/model.nb
warmup:10
repeats:100
power_mode:1
thread_num:0
*** time info(ms) ***
1st_duration:33.87
max_duration:30.657
min_duration:28.5
avg_duration:29.046

====== output summary ======
i: 0,  index: 285,  name: n02124075 Egyptian cat,  score: 0.482869
i: 1,  index: 281,  name: n02123045 tabby, tabby cat,  score: 0.471595
i: 2,  index: 282,  name: n02123159 tiger cat,  score: 0.039779
``` 

```shell
 cd Paddle-Lite-Demo/libs
 # 下载所需要的 Paddle Lite 预测库
 sh download.sh
 cd ../image_classification/assets
 # 下载OPT 优化后模型、测试图片、标签文件
 sh download.sh
 cd ../android/app/shell/cxx/image_classification
 # 完成可执行文件的编译和运行
 sh build.sh
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

2. `Paddle-Lite-Demo/image_classification/assets/` : 存放图像分类 demo 的模型、测试图片、标签文件

3. `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/image_classification.cc` : 图像分类的预测代码
     - `pre_process(...)` : 完成图像分类的预处理功能
     - `post_process(...)` : 完成图像分类的后处理功能
     - `run_model(...)` : 完成图像分类的预测全流程功能
     - `load_labels(...)` : 完成标签文件读取功能
     - `neon_mean_scale(...)` : 完成图像数据赋值给Tensor的加速处理功能

4. `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/CMakeLists.txt` :  CMake 文件，约束可执行文件的编译方法

5. `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/build.sh` : 用于可执行文件的编译和运行

```shell
 # 位置
 Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/build.sh # 脚本默认编译 armv7 可执行文件
 # 如果要编译 armv8 可执行文件，可以将 build.sh 脚本中的 ARM_ABI 变量改为 arm64i-v8a 即可
 # build.sh 中包含了可执行文件的编译和运行功能，其中运行是调用run.sh 脚本进行完成
 # run.sh 脚本中可执行文件的参数含义：
 adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
              ./models/mobilenet_v1_for_cpu/model.nb \
              ./images/tabby_cat.jpg \
              ./labels/labels.txt \
              3 224 224 \
              0 1 100 10 \
           "

 第一个参数：image_classification 可执行文件，属于必选项
 第二个参数：./models/mobilenet_v1_for_cpu/model.nb 优化后的分类模型文件，属于必选项
 第三个参数：./images/tabby_cat.jpg  测试图片，属于必选项
 第四个参数：./labels/labels.txt  label 文件，属于必选项
 第五个参数：3 top-k 大小，属于可选项，默认是 1
 第六个参数：224 输入图片宽度，属于可选项，默认是 224
 第七个参数：224 输入图片高度，属于可选项，默认是 224
 第八个参数：0 是否绑核，0-绑定大核， 1-绑定小核，2-绑定所有核，3-不绑核，属于可选项，默认是 0
 第九个参数：1 线程数，属于可选项，默认是 1
 第十个参数：100 repeats 数目，属于可选项，默认是 1
 第十一个参数：10 warmup 数目，属于可选项，默认是 0
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
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `mobilenet_v1_for_cpu/model.nb`，则代码不需更新；否则话，需要修改 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/run.sh` 脚本。

例子：假设更新 mobilenet_v2 模型为例，则先将优化后的模型存放到 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/run.sh` 下，然后更新脚本

```shell
# path: Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/run.sh
# old
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
              ./models/mobilenet_v1_for_cpu/model.nb \
              ./images/tabby_cat.jpg \
              ./labels/labels.txt \
              3 224 224 \
              0 1 100 10 \
          "
# now
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
              ./models/mobilenet_v2_for_cpu/mv2.nb \
              ./images/tabby_cat.jpg \
              ./labels/labels.txt \
              3 224 224 \
              0 1 100 10 \
          "
```

**注意：**
-  如果更新模型的输入/输出 Tensor 个数、shape 和 Dtype 发生更新，需要更新文件 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/image_classification.cc` 的 `pre_process` 预处理和 `pre_process` 后处理代码即可。

- 如果需要更新 `labels.txt` 标签文件，则需要将新的标签文件存放在目录 `Paddle-Lite-Demo/image_classification/assets/labels/` 下，并更新 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/run.sh` 脚本。

```shell
# path: Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/run.sh
# old
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
              ./models/mobilenet_v1_for_cpu/model.nb \
              ./images/tabby_cat.jpg \
              ./labels/labels.txt \
              3 224 224 \
              0 1 100 10 \
          "
# now
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
              ./models/mobilenet_v1_for_cpu/model.nb \
              ./images/tabby_cat.jpg \
              ./labels/labels_new.txt \
              3 224 224 \
              0 1 100 10 \
          "
```

### 更新输入/输出预处理
1. 更新输入数据

- 将更新的图片存放在 `Paddle-Lite-Demo/image_classification/assets/images/` 下；
- 更新文件 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/run.sh` 脚本

以更新 `dog.jpg` 为例，则先将 `dog.jpg` 存放在 `Paddle-Lite-Demo/image_classification/assets/images/` 下，然后更新脚本

```shell
# path: Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/run.sh
# old
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
              ./models/mobilenet_v1_for_cpu/model.nb \
              ./images/tabby_cat.jpg \
              ./labels/labels.txt \
              3 224 224 \
              0 1 100 10 \
          "
# now
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           && ./image_classification \
              ./models/mobilenet_v1_for_cpu/model.nb \
              ./images/dog.jpg \
              ./labels/labels.txt \
              3 224 224 \
              0 1 100 10 \
          "
```


2. 更新输入预处理
此处需要更新 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/image_classification.cc` 的 `pre_process` 预处理实现就行。

3. 更新输出预处理
此处需要更新 `Paddle-Lite-Demo/image_classification/android/shell/cxx/image_classification/image_classification.cc` 的 `post_process` 后处理代码实现就行。
