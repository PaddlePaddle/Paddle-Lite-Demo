## Harmony demo部署方法

### 概述

本教程基于[ Paddle Lite] 的目标检测示例[object_detection_demo]程序，演示端侧部署的流程。

本章将详细说明如何在端侧利用 Paddle Lite  `Java API` 和 Paddle Lite  图像检测模型完成端侧推理。

### 部署应用

**目的**：将基于 Paddle Lite 预测库的 Harmony APP 部署到手机，实现物体检测

**需要的环境**： DevEco Studio、鸿蒙手机（开启 USB 调试模式）、下载到本地的[ Object Detection Demo ]工程

**部署步骤**：
1、搭建鸿蒙开发环境流程，可参考[DevEco下载](https://developer.harmonyos.com/cn/docs/documentation/doc-guides/software_install-0000001053582415)、
[配置开发环境](https://developer.harmonyos.com/cn/docs/documentation/doc-guides/environment_config-0000001052902427)

2、用 DevEco Studio 打开 object_detection_demo 工程 （本步骤需要联网，下载 Paddle Lite 预测库和模型）。

3、手机连接电脑，打开**USB调试**和**文件传输模式**，在 DevEco Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

![studio](https://github.com/helwens/object_detection_demo/blob/main/docs/images/studio.png)

4.真机运行前需要配置签名（如果使用模拟器运行可跳过这一步）
请根据华为官网[使用真机进行调试](https://developer.harmonyos.com/cn/docs/documentation/doc-guides/ide_debug_device-0000001053822404#section20493196162)
进行配置
>**注意：**
> 如果真机运行未配置签名可能会出现错误提示，
Failure[INSTALL_FAILED_NO_BUNDLE_SIGNATURE]
Sign the app before running it on a real device.
the instructions to configure the signature information.

5、按下 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载 Paddle Lite 预测库和模型，需要联网)


成功后效果如下，图一：APP 安装到手机        图二： APP 打开后的效果，会自动识别图片中的物体并标记

<p align="center"><img width="300" height="450"  src="https://github.com/helwens/object_detection_demo/blob/main/docs/images/app install.jpg"/>
&#8194;&#8194;&#8194;&#8194;&#8194;<img width="300" height="450"  src="https://github.com/helwens/object_detection_demo/blob/main/docs/images/eason.png"/></p>


## Harmony demo 结构讲解

Harmony 示例的代码结构如下图所示：

<p align="center"><img width="695" height="900"  src="https://github.com/helwens/object_detection_demo/blob/main/docs\images\code_structure.png"/></p>


1、 `Predictor.java`： 预测代码

```shell
# 位置：
object_detection_demo\entry\src\main\java\com\baidu\paddle\lite\demo\object_detection\Predictor.java
```

2、 `model.nb` : 模型文件 (opt 工具转化后 Paddle Lite 模型), `pascalvoc_label_list`：训练模型时的 `labels` 文件

```shell
# 位置：
object_detection_demo\entry\src\main\resources\rawfile\models\ssd_mobilenet_v1_pascalvoc_for_cpu\model.nb
object_detection_demo\entry\src\main\resources\rawfile\labels\pascalvoc_label_list
# 如果要替换模型，可以将新模型放到 `object_detection_demo\entry\src\main\resources\rawfile\models\ssd_mobilenet_v1_pascalvoc_for_cpu` 目录下
# 同时模型对应的标签文件也可能不同，需要替换到`object_detection_demo\entry\src\main\resources\rawfile\labels` 目录下
```

3、 `libpaddle_lite_jni.so、PaddlePredictor.jar`：Paddle Lite Java 预测库与 Jar 包

```shell
# 位置
object_detection_demo\entry\libs\arm64-v8a\libpaddle_lite_jni.so
object_detection_demo\entry\libs\PaddlePredictor.jar
# 如果要替换动态库 so 和 jar 文件，则将新的动态库 so 更新到 `object_detection_demo\entry\libs\arm64-v8a\` 目录下，新的 jar 文件更新至 `object_detection_demo\entry\libs\` 目录下
```

4、`build.gradle` : 定义编译过程的 gradle 脚本。（不用改动，定义了自动下载 Paddle Lite 预测和模型的过程）

```shell
# 位置
object_detection_demo\build.gradle
# 如果需要手动更新模型和预测库，则可将 gradle 脚本中的 `download*` 接口注释即可
```

## 代码讲解 （使用 Paddle Lite `Java API` 执行预测）

Harmony 示例基于 Java API 开发，调用 Paddle Lite `Java API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite Java API ](../api_reference/java_api_doc)。

```c++
// 导入 Java API
import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.Predictor;
import com.baidu.paddle.lite.PowerMode;

// 1. 写入配置：设置 MobileConfig
MobileConfig config = new MobileConfig();
config.setModelFromFile(<modelPath>); // 设置 Paddle Lite 模型路径
config.setPowerMode(PowerMode.LITE_POWER_NO_BIND); // 设置 CPU 运行模式
config.setThreads(4); // 设置工作线程数

// 2. 创建 PaddlePredictor
PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

// 3. 设置输入数据
long[] dims = {100, 100};
float[] inputBuffer = new float[10000];
for (int i = 0; i < 10000; ++i) {
    inputBuffer[i] = i;
}
// 如果输入是图片，则可在第三步时将预处理后的图像数据赋值给输入 Tensor
Tensor input = predictor.getInput(0);
input.resize(dims);
input.setData(inputBuffer);

// 4. 执行预测
predictor.run();

// 5. 获取输出数据
Tensor result = predictor.getOutput(0);
float[] output = result.getFloatData();
for (int i = 0; i < 1000; ++i) {
    System.out.println(output[i]);
}

// 例如目标检测：输出后处理，输出检测结果
// Fetch output tensor
Tensor outputTensor = getOutput(0);

// Post-process
 long outputShape[] = outputTensor.shape();
 long outputSize = 1;
 for (long s : outputShape) {
   outputSize *= s;
 }

 int objectIdx = 0;
 for (int i = 0; i < outputSize; i += 6) {
   float score = outputTensor.getFloatData()[i + 1];
   if (score < scoreThreshold) {
      continue;
   }
   int categoryIdx = (int) outputTensor.getFloatData()[i];
   String categoryName = "Unknown";
   if (wordLabels.size() > 0 && categoryIdx >= 0 && categoryIdx < wordLabels.size()) {
     categoryName = wordLabels.get(categoryIdx);
   }
   float rawLeft = outputTensor.getFloatData()[i + 2];
   float rawTop = outputTensor.getFloatData()[i + 3];
   float rawRight = outputTensor.getFloatData()[i + 4];
   float rawBottom = outputTensor.getFloatData()[i + 5];
   float clampedLeft = Math.max(Math.min(rawLeft, 1.f), 0.f);
   float clampedTop = Math.max(Math.min(rawTop, 1.f), 0.f);
   float clampedRight = Math.max(Math.min(rawRight, 1.f), 0.f);
   float clampedBottom = Math.max(Math.min(rawBottom, 1.f), 0.f);
   // detect_box coordinate
   float imgLeft = clampedLeft * imgWidth;
   float imgTop = clampedTop * imgWidth;
   float imgRight = clampedRight * imgHeight;
   float imgBottom = clampedBottom * imgHeight;
   objectIdx++;
}

```

### Q&A:
问题：
- 提示 `in_dims().size() == 4 || in_dims.size() == 5 test error`
  - 如果你是基于我们的 demo 工程替换模型以后出现这个问题，有可能是替换模型以后模型的输入和 Paddle Lite 接收的输入不匹配导致，可以参考[ issue 6406 ](https://github.com/PaddlePaddle/Paddle-Lite/issues/6406)来解决该问题。
- 如果想进一步提高 APP 速度：
  - 可以将 APP 的默认线程数由线程数 1 更新为多线程，如 4 线程--modelConfig.setThreads(4);
  - 多线程使用限制：线程数最大值是手机大核处理器的个数，如小米 9，它由 4 个 A76 大核组成，即最大运行 4 个线程。
  - 多线程预测库：GCC 编译，V7/V8 多线程均支持；clang 编译下，只支持V8 多线程，V7 多线程编译受限于 NDK，当前 NDK >= 17, 编译报错，问题来源 NDK 内部 clang 编译的寄存器数目限制。
- 如果想用 FP16 模型推理：
  - 更新预测库：包含FP16 kernel的预测库，可以在 [release 官网](https://github.com/PaddlePaddle/Paddle-Lite/tags)下载，也可以参考[源码编译文档](../source_compile/macos_compile_android.rst)，自行编译。
  - 更新 nb 模型：需要使用 OPT 工具，将 `enable_fp16` 设置为 ON，重新转换模型。
  - FP16 预测库和 FP16 模型只在**V8.2 架构以上的手机**上运行，即高端手机，如小米 9，华为 P30 等
