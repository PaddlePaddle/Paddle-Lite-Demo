# 目标检测 Demo 使用指南
在 Android 上实现实时的目标检测功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍目标检测 Demo 运行方法和如何在更新模型/输入/输出处理下，保证目标检测 demo 仍可继续运行。

## 如何运行目标检测 Demo

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

1. 目标检测 Demo 位于 `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3` 目录
2. cd `Paddle-Lite-Demo/libs` 目录，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库
3. cd `Paddle-Lite-Demo/object_detection/assets` 目录，运行 `download.sh` 脚本，下载OPT 优化后模型、测试图片和标签文件
4. cd `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3` 目录，先在 `build.sh` 脚本中，完成 NDK_ROOT 路径设置；然后运行 `build.sh` 脚本，完成可执行文件的编译和运行。
> **注意事项：**
>> - 如果是在 Linux 主机编译，请选择 Linux 版本的 NDK 进行设置
>> - 如果是在 Mac 主机编译，请选择 Mac 版本的 NDK 进行设置；另外，同步更新 `CMakeList.txt` 里的 `CMAKE_SYSTEM_NAME` 变量，更新为 `drawn`

5. 运行结果如下所示：

```shell
======= benchmark summary =======
input_shape(s) (NCHW): {1, 3, 320, 320}
model_dir:./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10//model.nb
warmup:1
repeats:10
power_mode:1
thread_num:0
*** time info(ms) ***
1st_duration:362.273
max_duration:231.867
min_duration:227.434
avg_duration:229.626

====== output summary ======
detection, image size: 768, 576, detect object: bicycle, score: 0.918155, location: x=121, y=107, width=430, height=332
detection, image size: 768, 576, detect object: car, score: 0.705025, location: x=471, y=81, width=221, height=88
detection, image size: 768, 576, detect object: dog, score: 0.624054, location: x=122, y=211, width=191, height=317
result has been saved to picture: dog_yolo_v3_mobilenetv3_detection_result.jpg
``` 

```shell
 cd Paddle-Lite-Demo/libs
 # 下载所需要的 Paddle Lite 预测库
 sh download.sh
 cd ../object_detection/assets
 # 下载OPT 优化后模型、测试图片、标签文件
 sh download.sh
 cd ../android/app/shell/cxx/yolov3_mobilenet_v3
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

2. `Paddle-Lite-Demo/object_detection/assets/` : 存放目标检测 demo 的模型、测试图片、标签文件

3. `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/yolov3_mobilenet_v3.cc` : 目标检测的预测代码
     - `pre_process(...)` : 完成目标检测的预处理功能
     - `post_process(...)` : 完成目标检测的后处理功能
     - `run_model(...)` : 完成目标检测的预测全流程功能
     - `load_labels(...)` : 完成标签文件读取功能
     - `neon_mean_scale(...)` : 完成图像数据赋值给Tensor的加速处理功能

4. `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/CMakeLists.txt` :  CMake 文件，约束可执行文件的编译方法

5. `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/build.sh` : 用于可执行文件的编译和运行

```shell
 # 位置
 Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/build.sh # 脚本默认编译 armv7 可执行文件
 # 如果要编译 armv8 可执行文件，可以将 build.sh 脚本中的 ARM_ABI 变量改为 arm64i-v8a 即可
 # build.sh 中包含了可执行文件的编译和运行功能，其中运行是调用run.sh 脚本进行完成
 # run.sh 脚本中可执行文件的参数含义：
 adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb \
               ./images/dog.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 100 5 0 \
           "

 第一个参数：yolov3_mobilenet_v3 可执行文件，属于必选项
 第二个参数：./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb 优化后的分类模型文件，属于必选项
 第三个参数：./images/dog.jpg  测试图片，属于必选项
 第四个参数：./labels/coco_label_list.txt  label 文件，属于必选项
 第五个参数：0.5 thresh 域值大小，属于可选项，默认是 0.5
 第六个参数：320 输入图片宽度，属于可选项，默认是 320
 第七个参数：320 输入图片高度，属于可选项，默认是 320
 第八个参数：0 是否绑核，0-绑定大核， 1-绑定小核，2-绑定所有核，3-不绑核，属于可选项，默认是 0
 第九个参数：1 线程数，属于可选项，默认是 1
 第十个参数：100 repeats 数目，属于可选项，默认是 1
 第十一个参数：10 warmup 数目，属于可选项，默认是 0
 第十二个参数：0 是否使用 GPU，属于可选项，默认是 0
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

## 更新模型、输入和输出预处理

### 更新模型

1. 将优化后的模型存放到目录 `Paddle-Lite-Demo/object_detection/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `yolov3_mobilenet_v3/model.nb`，则代码不需更新；否则话，需要修改 `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/run.sh` 脚本。

例子：假设更新 yolov3_mobilenet_v1 模型为例，更新脚本

```shell
# path: Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/run.sh
# old
adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb \
               ./images/dog.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 100 5 0 \
           "
# now
 adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v1/model.nb \
               ./images/dog.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 100 5 0 \
           "
```

**注意：**
-  如果更新模型的输入/输出 Tensor 个数、shape 和 Dtype 发生更新，需要更新文件 `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/yolov3_mobilenet_v3.cc` 的 `pre_process` 预处理和 `pre_process` 后处理代码即可。

- 如果需要更新 `labels.txt` 标签文件，则需要将新的标签文件存放在目录 `Paddle-Lite-Demo/object_detection/assets/labels/` 下，并更新 `Paddle-Lite-Demo/yolov3_mobilenet_v3/android/shell/cxx/yolov3_mobilenet_v3/run.sh` 脚本。

```shell
# path: Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/run.sh
# old
adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb \
               ./images/dog.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 100 5 0 \
           "
# now
adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb \
               ./images/dog.jpg \
               ./labels/new_label_list.txt \
               0.5 320 320 \
               0 1 100 5 0 \
           "
```

- 如果想在 GPU 上推理，则将run.sh 中 use_gpu 设为True 即可：

```shell
adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb \
               ./images/dog.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 100 5 1 \
           "
```

### 更新输入/输出预处理
1. 更新输入数据

- 将更新的图片存放在 `Paddle-Lite-Demo/object_detection/assets/images/` 下；
- 更新文件 `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/run.sh` 脚本

以更新 `cat.jpg` 为例，则先将 `cat.jpg` 存放在 `Paddle-Lite-Demo/yolov3_mobilenet_v3/assets/images/` 下，然后更新脚本

```shell
# path: Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/run.sh
# old
adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb \
               ./images/dog.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 100 5 0 \
           "
# now
adb shell "cd ${ADB_DIR} \
           && chmod +x ./yolov3_mobilenet_v3 \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./yolov3_mobilenet_v3 \
               ./models/yolov3_mobilenet_v3_prune86_FPGM_320_fp32_fluid_for_cpu_v2_10/model.nb \
               ./images/cat.jpg \
               ./labels/coco_label_list.txt \
               0.5 320 320 \
               0 1 100 5 0 \
           "
```


2. 更新输入预处理
此处需要更新 `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/yolov3_mobilenet_v3.cc` 的 `pre_process` 预处理实现就行。

3. 更新输出预处理
此处需要更新 `Paddle-Lite-Demo/object_detection/android/shell/cxx/yolov3_mobilenet_v3/yolov3_mobilenet_v3.cc` 的 `post_process` 后处理代码实现就行。

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。
