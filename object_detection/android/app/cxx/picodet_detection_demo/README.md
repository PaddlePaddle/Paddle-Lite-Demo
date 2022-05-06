# 目标检测 C++ API Demo 使用指南

在 Android 上实现实时的目标检测功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍目标检测 Demo 运行方法和如何在更新模型/输入/输出处理下，保证目标检测 demo 仍可继续运行。

## 如何运行目标检测 Demo

### 环境准备

1. 在本地环境安装好 Android Studio 工具，详细安装方法请见[Android Stuido 官网](https://developer.android.com/studio)。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

**注意**：如果您的 Android Studio 尚未配置 NDK ，请根据 Android Studio 用户指南中的[安装及配置 NDK 和 CMake ](https://developer.android.com/studio/projects/install-ndk)内容，预先配置好 NDK 。您可以选择最新的 NDK 版本，或者使用
Paddle Lite 预测库版本一样的 NDK

### 部署步骤

1. 目标检测 Demo 位于 `Paddle-Lite-Demo/object_detection/android/app/cxx/picodet_detection_demo` 目录
2. 用 Android Studio 打开 picodet_detection_demo 工程
3. 手机连接电脑，打开 USB 调试和文件传输模式，并在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

<p align="center">
<img width=250, height=300, src="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/docs_img/android/run_app.jpg"/>
</p>

> **注意：**
>> 如果您在导入项目、编译或者运行过程中遇到 NDK 配置错误的提示，请打开 ` File > Project Structure > SDK Location`，修改 `Andriod NDK location` 为您本机配置的 NDK 所在路径。
>> 如果您是通过 Andriod Studio 的 SDK Tools 下载的 NDK (见本章节"环境准备")，可以直接点击下拉框选择默认路径。
>> 还有一种 NDK 配置方法，你可以在 `picodet_detection_demo/local.properties` 文件中手动完成 NDK 路径配置，如下图所示
>> 如果以上步骤仍旧无法解决 NDK 配置错误，请尝试根据 Andriod Studio 官方文档中的[更新 Android Gradle 插件](https://developer.android.com/studio/releases/gradle-plugin?hl=zh-cn#updating-plugin)章节，尝试更新Android Gradle plugin版本。

4. 点击 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载 Paddle Lite 预测库和模型，需要联网)
成功后效果如下，图一：APP 安装到手机        图二： APP 打开后的效果，会自动识别图片中的物体并标记

  | APP 图标 | APP 效果 |
  | ---     | --- |
  | ![app_pic ](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/docs_img/android/app_pic.jpg)    | ![app_res ](https://paddlelite-demo.bj.bcebos.com/demo/object_detection/docs_img/android/app_run_res.jpg) |

## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 Android 预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 java 库
        * jar 包
          将生成的 `build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/java/jar/PaddlePredictor.jar` 替换 Demo 中的 `Paddle-Lite-Demo/object_detection/andrdoid/app/cxx/picodet_detection_demo/app/PaddleLite/java/PaddlePredictor.jar`
        * Java so
            * armeabi-v7a
              将生成的 `build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/java/so/libpaddle_lite_jni.so` 库替换 Demo 中的 `Paddle-Lite-Demo/object_detection/andrdoid/app/cxx/picodet_detection_demo/app/PaddleLite/java/libs/armeabi-v7a/libpaddle_lite_jni.so`
            * arm64-v8a
              将生成的 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so` 库替换 Demo 中的 `Paddle-Lite-Demo/object_detection/andrdoid/app/cxx/picodet_detection_demo/app/PaddleLite/java/libs/arm64-v8a/libpaddle_lite_jni.so`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/object_detection/andrdoid/app/cxx/picodet_detection_demo/app/PaddleLite/cxx/include`
        * armeabi-v7a
          将生成的 `build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/object_detection/andrdoid/app/cxx/picodet_detection_demo/app/PaddleLite/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so`
        * arm64-v8a
          将生成的 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/object_detection/andrdoid/app/cxx/picodet_detection_demo/app/PaddleLite/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so`

## Demo 内容介绍

先整体介绍下目标检测 Demo 的代码结构，然后再从 Java 和 C++ 两部分简要的介绍 Demo 每部分功能.

<p align="center"><img width=250, height=300, src="https://paddlelite-demo.bj.bcebos.com/demo/object_detection/docs_img/android/predict.jpg"/></p>

1. `Native.java`： Java 预测代码

```shell
# 位置：
picodet_detection_demo/app/src/main/java/com/baidu/paddle/lite/demo/object_detection/Native.java
```

2. `Native.cc`： Jni 预测代码用于 Java 与 C++ 语言传递信息

```shell
# 位置：
picodet_detection_demo/app/src/main/cpp/Native.cc
```

3. `Pipeline.cc`： C++ 预测代码

```shell
# 位置：
picodet_detection_demo/app/src/main/cpp/Pipeline.cc
```

4. `model.nb` : 模型文件 (opt 工具转化后 Paddle Lite 模型), `pascalvoc_label_list`：训练模型时的 `labels` 文件

```shell
# 位置：
picodet_detection_demo/app/src/main/assets/models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb
picodet_detection_demo/app/src/main/assets/labels/pascalvoc_label_list
```

5. `libpaddle_lite_api_shared.so`：Paddle Lite C++ 预测库

```shell
# 位置
picodet_detection_demo/app/PaddleLite/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so
# 如果要替换动态库 so，则将新的动态库 so 更新到此目录下
```

6. `build.gradle` : 定义编译过程的 gradle 脚本。（不用改动，定义了自动下载 Paddle Lite 预测和模型的过程）

```shell
# 位置
picodet_detection_demo/app/build.gradle
# 如果需要手动更新模型和预测库，则可将 gradle 脚本中的 `download*` 接口注释即可
```

7. `CMakeLists.txt` : C++ 预测库代码的编译脚本，用于生成 jni 的动态库 `lib_Native.so`

```shell
# 位置
picodet_detection_demo/app/cpp/CMakeLists.txt
# 如果有cmake 编译选项更新，可以在 CMakeLists.txt 进行修改即可
```
### Java 端

* 模型存放，将下载好的模型解压存放在 `app/src/assets/models` 目录下
* common Java 包
    在 `app/src/java/com/baidu/paddle/lite/demo/common` 目录下，实现摄像头和框架的公共处理，一般不用修改。其中，Utils.java 用于存放一些公用的且与 Java 基类无关的功能，例如模型拷贝、字符串类型转换等
* object_detection Java 包
   在 `app/src/java/com/baidu/paddle/lite/demo/object_detection` 目录下，实现 APP 界面消息事件和 Java/C++ 端代码互传的桥梁功能
* MainActivity
    实现 APP 的创建、运行、释放功能
    重点关注 `checkAndUpdateSettings`和 `onTextureChanged` 函数，实现 APP 界面值向 C++ 端值互传及预测处理流程
    
    ```java
      public void checkAndUpdateSettings() {
             if (SettingsActivity.checkAndUpdateSettings(this)) {
                 String realModelDir = getCacheDir() + "/" + SettingsActivity.modelDir;
                 Utils.copyDirectoryFromAssets(this, SettingsActivity.modelDir, realModelDir);
                 String realLabelPath = getCacheDir() + "/" + SettingsActivity.labelPath;
                 Utils.copyFileFromAssets(this, SettingsActivity.labelPath, realLabelPath);
                 // 初始化
                 predictor.init(
                         realModelDir,
                         realLabelPath,
                         SettingsActivity.cpuThreadNum,
                         SettingsActivity.cpuPowerMode,
                         SettingsActivity.inputWidth,
                         SettingsActivity.inputHeight,
                         SettingsActivity.inputMean,
                         SettingsActivity.inputStd,
                         SettingsActivity.scoreThreshold);
             }
         }
      
      public boolean onTextureChanged(Bitmap ARGB8888ImageBitmap) {
          String savedImagePath = "";
          synchronized (this) {
              savedImagePath = MainActivity.this.savedImagePath;
          }
          // 预测
          boolean modified = predictor.process(ARGB8888ImageBitmap, savedImagePath);
          if (!savedImagePath.isEmpty()) {
              synchronized (this) {
                  MainActivity.this.savedImagePath = "";
              }
          }
          lastFrameIndex++;
          if (lastFrameIndex >= 30) {
              final int fps = (int) (lastFrameIndex * 1e9 / (System.nanoTime() - lastFrameTime));
              runOnUiThread(new Runnable() {
                  public void run() {
                      tvStatus.setText(Integer.toString(fps) + "fps");
                  }
              });
              lastFrameIndex = 0;
              lastFrameTime = System.nanoTime();
          }
          return modified;
      }
    ```
   
* SettingActivity
    实现设置界面各个元素的更新与显示，如果新增/删除界面的某个元素，均在这个类里面实现
    备注：

    - 参数的默认值可在 `app/src/main/res/values/strings.xml` 查看
    - 每个元素的 ID 和 value 是对应 `app/src/main/res/xml/settings.xml` 和 `app/src/main/res/values/string.xml` 文件中的值
    - 这部分内容不建议修改，如果有新增属性，可以按照此格式进行添加

* Native
    实现 Java 与 C++ 端代码互传的桥梁功能
    包含三个功能：`init`初始化、 `process`预测处理 和 `release`释放
    备注：
        Java 的 native 方法和 C++ 的 native 方法要一一对应
     
 ### C++ 端（native）

 * Native
  实现 Java 与 C++ 端代码互传的桥梁功能，将 Java 数值转换为 c++ 数值，调用 c++ 端的完成人脸关键点检测功能
  **注意：**
  Native 文件生成方法：
   
  ```shell
   cd app/src/java/com/baidu/paddle/lite/demo/face_keypoints_detection
   # 在当前目录会生成包含 Native 方法的头文件，用户可以将其内容拷贝至 `cpp/Native.cc` 中
   javac -classpath D:\dev\android-sdk\platforms\android-29\android.jar -encoding utf8 -h . Native.java 
  ```

 * Pipeline
  实现输入预处理、推理执行和输出后处理的流水线处理，支持多个模型的串行处理

 * Utils
  实现其他辅助功能，如 `NHWC` 格式转 `NCHW` 格式、字符串处理等

 * 新增模型支持
    - 在 Pipeline 文件中新增模型的预测类，实现图像预处理、预测和图像后处理功能
    - 在 Pipeline 文件中 `Pipeline` 类添加该模型预测类的调用和处理

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

## 如何更新模型和输入/输出预处理

### 更新模型

1. 将优化后的模型存放到目录 `picodet_detection_demo/app/src/main/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `picodet_s_320_coco_for_cpu/model.nb`，则代码不需更新；否则话，需要修改 `picodet_detection_demo/app/src/main/java/com.baidu.paddle.lite.demo.object_detection/MainActivity.java` 中代码：

以更新 ssd_mobilenet_v3 模型为例，则先将优化后的模型存放到 `picodet_detection_demo/app/src/main/assets/models/ssd_mobilenet_v3_for_cpu/ssd_mv3.nb` 下，然后更新代码

```java
// 代码文件 `picodet_detection_demo/app/src/main/java/com.baidu.paddle.lite.demo.object_detection/MainActivity.java`
public void checkAndUpdateSettings() {
        if (SettingsActivity.checkAndUpdateSettings(this)) {
            // old
            // String realModelDir = getCacheDir() + "/" + SettingsActivity.modelDir;
            // now
            String realModelDir = getCacheDir() + "/" + "models/ssd_mobilenet_v3_for_cpu/"; // change modelDir
            Utils.copyDirectoryFromAssets(this, SettingsActivity.modelDir, realModelDir);
            String realLabelPath = getCacheDir() + "/" + SettingsActivity.labelPath;
            Utils.copyFileFromAssets(this, SettingsActivity.labelPath, realLabelPath);
            predictor.init(
                    realModelDir,
                    realLabelPath,
                    SettingsActivity.cpuThreadNum,
                    SettingsActivity.cpuPowerMode,
                    SettingsActivity.inputWidth,
                    SettingsActivity.inputHeight,
                    SettingsActivity.inputMean,
                    SettingsActivity.inputStd,
                    SettingsActivity.scoreThreshold);
        }
}
```
**注意：**

- 如果优化后的模型名字不是 `model.nb`，则需要将优化后的模型名字更新为 `model.nb` 或修改 `picodet_detection_demo/app/src/main/cpp/Pipeline.cc` 中代码

```c++
// 代码文件 `picodet_detection_demo/app/src/main/cpp/Pipeline.cc`
Detector::Detector(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
      inputStd_(inputStd), scoreThreshold_(scoreThreshold) {
  paddle::lite_api::MobileConfig config;
  // old
  // config.set_model_from_file(modelDir + "/model.nb");
  // now
  config.set_model_from_file(modelDir + "/ssd_mv3.nb"); // change model_name
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  labelList_ = LoadLabelList(labelPath);
  colorMap_ = GenerateColorMap(labelList_.size());
}
```


-  如果更新模型的输入/输出 Tensor 个数、shape 和 Dtype 发生更新，需要更新文件 `picodet_detection_demo/app/src/main/cpp/Pipeline.cc` 的 `Detector::Preprocess` 预处理和 `Detector::Postprocess` 后处理代码即可。


- 如果需要更新 `pascalvoc_label_list` 标签文件，则需要将新的标签文件存放在目录 `picodet_detection_demo/app/src/main/assets/labels/` 下，并参考模型更新方法更新 `picodet_detection_demo/app/src/main/java/com.baidu.paddle.lite.demo.object_detection/MainActivity.java` 代码文件

```java
// 代码文件 `picodet_detection_demo/app/src/main/java/com.baidu.paddle.lite.demo.object_detection/MainActivity.java`
public void checkAndUpdateSettings() {
        if (SettingsActivity.checkAndUpdateSettings(this)) {
            String realModelDir = getCacheDir() + "/" + SettingsActivity.modelDir;
            Utils.copyDirectoryFromAssets(this, SettingsActivity.modelDir, realModelDir);
            // old
            // String realLabelPath = getCacheDir() + "/" + SettingsActivity.labelPath;
            // now
            String realLabelPath = getCacheDir() + "/" + "new_label_path.txt";
            Utils.copyFileFromAssets(this, SettingsActivity.labelPath, realLabelPath);
            predictor.init(
                    realModelDir,
                    realLabelPath,
                    SettingsActivity.cpuThreadNum,
                    SettingsActivity.cpuPowerMode,
                    SettingsActivity.inputWidth,
                    SettingsActivity.inputHeight,
                    SettingsActivity.inputMean,
                    SettingsActivity.inputStd,
                    SettingsActivity.scoreThreshold);
        }
}
```

### 更新输入/输出预处理

1. 更新输入数据
  - 将更新的图片存放在 `picodet_detection_demo/app/src/main/assets/images/` 下；
  - 更新文件 `picodet_detection_demo/app/src/main/java/com.baidu.paddle.lite.demo.object_detection/MainActivity.java` 中的代码

  以更新 `cat.jpg` 为例，则先将 `cat.jpg` 存放在 `picodet_detection_demo/app/src/main/assets/images/` 下，然后更新代码

```java
// 代码文件 `picodet_detection_demo/app/src/main/java/com.baidu.paddle.lite.demo.object_detection/MainActivity.java`
public boolean onTextureChanged(Bitmap ARGB8888ImageBitmap) {
        String savedImagePath = "";
        synchronized (this) {
            savedImagePath = MainActivity.this.savedImagePath;
        }
        // update image
        Bitmap new_bit;
        ARGB8888ImageBitmap = new_bit;
        boolean modified = predictor.process(ARGB8888ImageBitmap, savedImagePath);
        if (!savedImagePath.isEmpty()) {
            synchronized (this) {
                MainActivity.this.savedImagePath = "";
            }
        }
        lastFrameIndex++;
        if (lastFrameIndex >= 30) {
            final int fps = (int) (lastFrameIndex * 1e9 / (System.nanoTime() - lastFrameTime));
            runOnUiThread(new Runnable() {
                public void run() {
                    tvStatus.setText(Integer.toString(fps) + "fps");
                }
            });
            lastFrameIndex = 0;
            lastFrameTime = System.nanoTime();
        }
        return modified;
    }

```

**注意：** 本 Demo 是以视频流做输入数据，如果要用图片，可以通过摄像头将图片输入，不用修改代码；或者修改输入 image 参数，将图片以 cv::mat 或 Bitmap 方式传进去


2. 更新输入预处理
此处需要更新 `picodet_detection_demo/app/src/main/cpp/Pipeline.cc` 中的 `Detector::Preprocess(const cv::Mat &rgbaImage)` 方法

**注意：** 如果模型的的输入 tensor 个数、输入 shape 和数据类型 Dtype 有更新，可以在 `Detector::Preprocess(const cv::Mat &rgbaImage)` 方法中更新模型的输入


3. 更新输出预处理
此处需要更新 `picodet_detection_demo/app/src/main/cpp/Pipeline.cc` 中的 `Detector::Postprocess(std::vector<Object> *results)` 方法

**注意：**

- 如果需要更新输出显示效果，可以更新 `picodet_detection_demo/app/src/main/cpp/Pipeline.cc`中的 `Pipeline::VisualizeStatus(double preprocessTime, double predictTime, double postprocessTime, cv::Mat *rgbaImage)` 方法 和 `Pipeline::VisualizeResults(const std::vector<Object> &results, cv::Mat *rgbaImage)` 方法即可

- 如果模型的的输出 tensor 个数、输出 shape 和数据类型 Dtype 有更新，可以在 `Detector::Postprocess(std::vector<Object> *results)` 方法中更新模型的输出

## 介绍 Pipeline 文件中的方法

代码文件：`picodet_detection_demo/app/src/main/cpp/Pipeline.cc`
`Pipeline.cc` 包含两个类：Detector 和 Pipeline 类

- Detector 用于检测模型的全流程处理，即输入图片预处理、预测处理和输出图片后处理
- Pipeline 用于检测 Demo 全流程处理，即初始化赋值、模型间信息交换、输出结果的显示处理（将结果返回Java/如何在界面回显）

```c++
// 检测类的构造函数
Detector::Detector(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold)；
// 检测类的输入预处理函数
void Detector::Preprocess(const cv::Mat &rgbaImage)；

// 检测类的输出预处理函数
void Detector::Postprocess(std::vector<Object> *results)；

// 检测类的预测函数
void Detector::Predict(const cv::Mat &rgbaImage, std::vector<Object> *results,
                       double *preprocessTime, double *predictTime,
                       double *postprocessTime)；
// Pipeline 的构造函数
Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold)；
// Pipeline 的输出结果显示函数
void Pipeline::VisualizeResults(const std::vector<Object> &results,
                                cv::Mat *rgbaImage)；
// Pipeline 的预测时间、前后处理时间等状态显示函数
void Pipeline::VisualizeStatus(double preprocessTime, double predictTime,
                               double postprocessTime, cv::Mat *rgbaImage)；
// Pipeline 的处理函数，用于模型间前后处理衔接
bool Pipeline::Process(cv::Mat &rgbaImage, std::string savedImagePath)；
```

### setting 界面参数介绍

可通过 APP 上的 Settings 按钮，实现目标检测 demo 中些许参数的更新，目前支持以下参数的更新：
参数的默认值可在 `app/src/main/res/values/strings.xml` 查看

- model setting：（需要提前将模型/图片/标签放在 assets 目录，或者通过 adb push 将其放置手机目录）
    - model_path 默认是 `models/ssd_mobilenet_v1_pascalvoc_for_cpu`
    - label_path 默认是 `labels/pascalvoc_label_list`

- CPU setting：
    - power_mode 默认是 `LITE_POWER_HIGH`
    - thread_num 默认是 1

- input setting：
    - input_height 默认是 `300`
    - input_width 默认是 `300`
    - input_mean 默认是 `0.5,0.5,0.5`
    - input_std  默认是 `0.5,0.5,0.5`
    - score_threshold 默认是 `0.5`

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。
