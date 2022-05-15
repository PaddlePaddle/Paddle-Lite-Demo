# 人脸检测 Java API Demo 使用指南

在 Android 上实现实人脸检测功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等.
本文主要介绍人脸检测 Demo 运行方法和如何在更新模型/输入/输出处理下，保证人脸检测 demo 仍可继续运行。

## 如何运行人脸检测 Demo

### 环境准备

1. 在本地环境安装好 Android Studio 工具，详细安装方法请见[Android Stuido 官网](https://developer.android.com/studio)。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

### 部署步骤

1. 人脸检测 Demo 位于 `Paddle-Lite-Demo/face_detection/android/app/java/face_detection` 目录
2. 用 Android Studio 打开 face_detection 工程
3. 手机连接电脑，打开 USB 调试和文件传输模式，并在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

> **注意：**

4. 点击 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载 Paddle Lite 预测库和模型，需要联网)
成功后效果如下:

| APP 图标 | APP 效果 |
  | ---     | --- |
  | <img width="750" height="750"  src="https://paddlelite-demo.bj.bcebos.com/demo/face_detection/docs_img/android_app_pic.jpg"/>    | <img width="750" height="750"  src="https://paddlelite-demo.bj.bcebos.com/demo/face_detection/docs_img/android_app_run_res.jpg"/> |


## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 Android 预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 java 库
        * jar 包
          将生成的 `build.lite.android.xxx.clang/inference_lite_lib.android.xxx/java/jar/PaddlePredictor.jar` 替换 Demo 中的 `Paddle-Lite-Demo/face_detection/android/java/face_detection/app/PaddleLite/java/PaddlePredictor.jar`
        * Java so
            * armeabi-v7a
              将生成的 `build.lite.android.armv7.clang/inference_lite_lib.android.armv7/java/so/libpaddle_lite_jni.so` 库替换 Demo 中的 `Paddle-Lite-Demo/face_detection/android/java/face_detection/app/PaddleLite/java/libs/armeabi-v7a/libpaddle_lite_jni.so`
            * arm64-v8a
              将生成的 `build.lite.android.armv8.clang/inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so` 库替换 Demo 中的 `Paddle-Lite-Demo/face_detection/android/java/face_detection/app/PaddleLite/java/libs/arm64-v8a/libpaddle_lite_jni.so`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.android.xxx.clang/inference_lite_lib.android.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/Pface_detection/android/java/face_detection/app/PaddleLite/cxx/include`
        * armeabi-v7a
          将生成的 `build.lite.android.armv7.clang/inference_lite_lib.android.armv7/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/face_detection/android/java/face_detection/app/PaddleLite/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so`
        * arm64-v8a
          将生成的 `build.lite.android.armv8.clang/inference_lite_lib.android.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/face_detection/android/java/face_detection/app/PaddleLite/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so`

## Demo 内容介绍

先整体介绍下目标检测 Demo 的代码结构，然后介绍 Java 各功能模块的功能。

### 重点关注内容

1. `Predictor.java`： 预测代码

```shell
# 位置：
face_detection/app/src/main/java/com/baidu/paddle/lite/demo/face_detection/Predictor.java
```

2. `model.nb` : 模型文件 (opt 工具转化后 Paddle Lite 模型)

```shell
# 位置：
face_detection/app/src/main/assets/models/facedetection_for_cpu/model.nb
```

3. `libpaddle_lite_jni.so、PaddlePredictor.jar`：Paddle Lite Java 预测库与 Jar 包

```shell
# 位置
face_detection/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so
face_detection/app/libs/PaddlePredictor.jar
# 如果要替换动态库 so 和 jar 文件，则将新的动态库 so 更新到 `face_detection/app/src/main/jniLibs/arm64-v8a/` 目录下，新的 jar 文件更新至 `face_detection/app/libs/` 目录下
```

4. `build.gradle` : 定义编译过程的 gradle 脚本。（不用改动，定义了自动下载 Paddle Lite 预测和模型的过程）

```shell
# 位置
face_detection/app/build.gradle
# 如果需要手动更新模型和预测库，则可将 gradle 脚本中的 `download*` 接口注释即可, 将新的预测库替换至相应目录下
```

### Java 端

* 模型存放，将下载好的模型解压存放在 `app/src/assets/models` 目录下
* face_detection Java 包
   在 `app/src/java/com/baidu/paddle/lite/demo/face_detection` 目录下，实现 APP 界面消息事件
* MainActivity
     实现 APP 的创建、运行、释放功能
     重点关注 `onLoadModel` 和 `onRunModel` 函数，实现 APP 界面值传递和推理处理
     
     ```
     public boolean onLoadModel() {
             return predictor.init(MainActivity.this, modelPath, labelPath, cpuThreadNum,
                     cpuPowerMode,
                     inputColorFormat,
                     inputShape, inputMean,
                     inputStd);
     }
     
     public boolean onRunModel() {
             return predictor.isLoaded() && predictor.runModel();
     }
     ```java
   
* SettingActivity
     实现设置界面各个元素的更新与显示如模型地址、线程数、输入shape大小等，如果新增/删除界面的某个元素，均在这个类里面实现
     备注：
         - 参数的默认值可在 `app/src/main/res/values/strings.xml` 查看
         - 每个元素的 ID 和 value 是对应 `app/src/main/res/xml/settings.xml` 和 `app/src/main/res/values/string.xml` 文件中的值
         - 这部分内容不建议修改，如果有新增属性，可以按照此格式进行添加

* Predictor
     使用 Java API 实现人脸检测模型的预测功能
     重点关注 `init`、 `preProcess`、`postProcess`和 `runModel` 函数，实现 Paddle Lite 端侧推理功能
     
     ```
     // 预处理函数
     public boolean preProcess();
     // 后处理函数
     public boolean postProcess();
     // 初始化函数，完成预测器初始化
     public boolean init(Context appCtx, String modelPath, String labelPath, int cpuThreadNum, String cpuPowerMode,
                             String inputColorFormat,
                             long[] inputShape, float[] inputMean,
                             float[] inputStd);
     // 预测处理函数，包含模型的预处理、预测和后处理三个过程
     public boolean runModel();
     ```java
  
## 代码讲解 （使用 Paddle Lite `Java API` 执行预测）

Android 示例基于 Java API 开发，调用 Paddle Lite `Java API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite Java API ](https://paddle-lite.readthedocs.io/zh/latest/api_reference/java_api_doc.html)。

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
long[] dims = {1, 3, 224, 224};
float[] inputBuffer = new float[1*3*224*224];
for (int i = 0; i < 1*3*224*224; ++i) {
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
for (int i = 0; i < 1001; ++i) {
    System.out.println(output[i]);
}

// 例如人脸检测：输出后处理，输出分类结果
// Fetch output tensor
Tensor outputTensor = getOutput(0);

// Post-process
long outputShape[] = outputTensor.shape();
long outputSize = 1;
 for (long s : outputShape) {
   outputSize *= s;
}
int[] max_index = new int[3]; // Top3 indices
double[] max_num = new double[3]; // Top3 scores
for (int i = 0; i < outputSize; i++) {
  float tmp = outputTensor.getFloatData()[i];
  int tmp_index = i;
  for (int j = 0; j < 3; j++) {
     if (tmp > max_num[j]) {
         tmp_index += max_index[j];
         max_index[j] = tmp_index - max_index[j];
         tmp_index -= max_index[j];
         tmp += max_num[j];
          max_num[j] = tmp - max_num[j];
          tmp -= max_num[j];
     }
  }
}
```

## 如何更新模型和输入/输出预处理

### 更新模型
1. 将优化后的模型存放到目录 `face_detection/app/src/main/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `facedetection_for_cpu/model.nb`，则代码不需更新；否则话，需要修改 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/MainActivity.java` 中代码：
例子：假设更新 facedetection_new 模型为例，则先将优化后的模型存放到 `face_detection/app/src/main/assets/models/facedetection_new/` 下，然后更新代码

```c++
// 代码文件 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/MainActivity.java`
public boolean onLoadModel() {
  modelPath = "models/facedetection_new/"; // change modelPath
  return predictor.init(MainActivity.this, modelPath, labelPath, cpuThreadNum,
                cpuPowerMode, inputColorFormat, inputShape, inputMean,
                inputStd);
}
```

**注意：**
- 如果优化后的模型名字不是 `model.nb`，则需要将优化后的模型名字更新为 `model.nb` 或修改 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/Predictor.java` 中代码

```c++
// 代码文件 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/Predictor.java`
config.setModelFromFile(realPath + File.separator + "model.nb");
更新：config.setModelFromFile(realPath + File.separator + "facedetection_new.nb");

```
- 本 Demo 提供了 setting 界面，可以在将新模型 facedetection_new 放在 `assets/models/`后，不用手动更新代码，直接在安装好 APP 的 setting 界面更新模型路径即可

- 如果更新模型的输入/输出 Tensor 个数、shape 和 Dtype 发生更新，需要更新文件 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/Predictor.java` 的 `preprocess` 预处理和 `postProcess` 后处理代码即可。

### 更新输入/输出预处理
1. 更新输入数据

- 将更新的图片存放在 `face_detection/app/src/main/assets/images/` 下；
- 更新文件 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/MainActivity.java`  中的代码

以更新 `face.jpg` 为例，则先将 `face_new.jpg` 存放在 `face_detection/app/src/main/assets/images/` 下，然后更新代码

```c++
// 代码文件 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/MainActivity.java` 中 init 方法的图片路径
public void onLoadModelSuccessed() {
        // Load test image from path and run model
        imagePath = "images/face_new.jpg"; // change image_path
        try {
            if (imagePath.isEmpty()) {
                return;
            }
            Bitmap image = null;
            // Read test image file from custom path if the first character of mode path is '/', otherwise read test
            // image file from assets
            if (!imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(imagePath);
            }
            if (image != null && predictor.isLoaded()) {
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
}
```

**注意：**
>> 本 Demo 支持拍照和从相册加载新图片进行推理。此处想更新图片，可通过拍照或从相册加载图片方式实现。

2. 更新输入预处理
此处需要更新 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/Predictor.java` 中的`preprocess` 预处理代码实现。

3. 更新输出预处理
此处需要更新 `face_detection/app/src/main/java/com.baidu.paddle.lite.demo.face_detection/Predictor.java` 中的 `postProcess` 后处理代码实现。

### setting 界面参数介绍
可通过 APP 上的 Settings 按钮，实现人脸检测 demo 中些许参数的更新，目前支持以下参数的更新：
参数的默认值可在 `app/src/main/res/values/strings.xml` 查看
- model setting：（需要提前将模型/图片放在 assets 目录，或者通过 adb push 将其放置手机目录）
    - model_path 默认是 `models/facedetection_for_cpu`
    - image_path 默认是 `images/face.jpg`

- CPU setting：
    - power_mode 默认是 `LITE_POWER_HIGH`
    - thread_num 默认是 1
- input setting：
    - input_shape 默认是 `1, 3, 240, 320`
    - input_image_format 默认是 `RGB`
    - input_mean 默认是 `0.498,0.498,0.498`
    - input_std  默认是 `0.502,0.502,0.502`

### setting 界面参数更新
打开 APP，点击右上角的 `:` 符合，选择 `Settings..` 选项，打开 setting 界面；
<p align="center">
<img src=https://paddlelite-demo.bj.bcebos.com/demo/face_detection/docs_img/android_app_settings.jpg>
</p>

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。