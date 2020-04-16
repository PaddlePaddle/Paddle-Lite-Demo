# FaceKepypoint Demo 
  在Android上实现实时的人脸关键点的检测功能，支持多人脸的关键点检测
  同时，此Demo还提供了美妆功能，如美白、磨皮等
  另外，此Demo具有很好的易用性和开放性。如在demo中跑自己训练好的模型，新增美妆处理功能。

## 功能
* 人脸框检测
* 人脸关键点检测
* 美妆特效

## 要求
* Android
    * Android Studio 3.4
    * Android手机或开发版，NPU的功能暂时只在nova5、mate30和mate30 5G上进行了测试，用户可自行尝试其它搭载了麒麟810和990芯片的华为手机（如nova5i pro、mate30 pro、荣耀v30，mate40或p40，且需要将系统更新到最新版）

## 安装
$ git clone https://github.com/PaddlePaddle/Paddle-Lite-Demo
 Android
    * 打开Android Studio，在"Welcome to Android Studio"窗口点击"Open an existing Android Studio project"，在弹出的路径选择窗口中进入"face_keypoints_demo"目录，然后点击右下角的"Open"按钮即可导入工程
    * 通过USB连接Android手机或开发板；
    * 载入工程后，点击菜单栏的Run->Run 'App'按钮，在弹出的"Select Deployment Target"窗口选择已经连接的Android设备，然后点击"OK"按钮；
    * 由于Demo所用到的库和模型均通过app/build.gradle脚本在线下载，因此，第一次编译耗时较长（取决于网络下载速度），请耐心等待；
    * 如果库和模型下载失败，建议手动下载并拷贝到相应目录下


## 更新到最新的预测库
* Paddle-Lite项目：https://github.com/PaddlePaddle/Paddle-Lite
* 参考 [Paddle-Lite文档](https://github.com/PaddlePaddle/Paddle-Lite/wiki)，编译Android预测库
* 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换java 库
        * jar包
          将生成的build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/java/jar/PaddlePredictor.jar替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/face_keypoints_detection_demo/app/PaddleLite/java/PaddlePredictor.jar
        * Java so
            * armeabi-v7a
              将生成build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/java/so/libpaddle_lite_jni.so库替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/face_keypoints_detection_demo/app/PaddleLite/java/libs/armeabi-v7a/libpaddle_lite_jni.so
            * arm64-v8a
              将生成build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so库替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/face_keypoints_detection_demo/app/PaddleLite/java/libs/arm64-v8a/libpaddle_lite_jni.so
    * 替换c++ 库
        * armeabi-v7a
          将生成build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/cxx/libs/libpaddle_lite_api_shared.so库替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/face_keypoints_detection_demo/app/PaddleLite/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so
        * arm64-v8a
          将生成build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/libs/libpaddle_lite_api_shared.so库替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/face_keypoints_detection_demo/app/PaddleLite/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so
   

## 效果展示
先用人脸检测模型检测出人脸，然后用人脸关键点模型检测出人脸68个关键点。
* 基于视频流的人脸检测
  原始图片：![origin](https://paddlelite-demo.bj.bcebos.com/doc/android_face_keypoints_detection_origin_face_cpu.jpg)
  用人脸检测模型检测出人脸，并将人脸用红色矩形框显示出来
  - CPU预测结果（测试环境：华为mate30）

    ![android_face_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_face_keypoints_detection_face_detect_cpu.jpg)
  - NPU预测结果

    待支持

* 基于视频流的人脸关键点检测
  用人脸检测模型检测出人脸，然后用口罩检测模型检测是否佩戴口罩，并用文本显示是否有口罩及其概率值
  - CPU预测结果（测试环境：华为mate30）

    ![android_face_keypoints_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_face_keypoints_detection_face_keypoints1_cpu.jpg)
  - NPU预测结果

    待支持
* 基于视频流的人脸美妆
  用人脸检测模型检测出人脸，然后用人脸关键点模型检测出人脸68个关键点。根据人脸68个关键点对它做瘦脸、美白等处理。如瘦脸功能，利用其中3号点到5号点距离作为瘦左脸距离，13号点到15号点距离作为瘦右脸距离，同时利用局部平移算法完成瘦脸。
  - CPU预测结果（测试环境：华为mate30）
    美白特效结果：

    ![android_face_keypoints_detection_face_beauty_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_face_keypoints_detection_face_beauty_cpu.jpg)
  - NPU预测结果

    待支持

## Demo内容介绍
主要从Java和C++两部分简要的介绍Demo每部分功能，更多详细的内容请见：![Demo使用指南](https://paddlelite-demo.bj.bcebos.com/doc/Introduction_to_face_keypoints_detection_demo.docx)

### Java端
* 模型存放，将下载好的模型解压存放在`app/src/assets/models`目录下
* common Java包
  在`app/src/java/com.baidu.paddle.lite.demo/common`目录下，实现摄像头和框架的公共处理，一般不用修改。其中，Utils.java用于存放一些公用的且与Java基类无关的功能，如果模型拷贝、字符串类型转换等
* mask_detection Java 包
  在`app/src/java/com.baidu.paddle.lite.demo/face_keypoints_detection`目录下，实现APP界面消息事件和Java/C++端代码互传的桥梁功能
* MainActivity
    实现APP的创建、运行、释放功能
    重点关注`checkAndUpdateSettings`函数，实现APP界面值向C++端值互传
    ```
    public void checkAndUpdateSettings() {
        if (SettingsActivity.checkAndUpdateSettings(this)) {
            String fdtRealModelDir = getCacheDir() + "/" + SettingsActivity.fdtModelDir;
            Utils.copyDirectoryFromAssets(this, SettingsActivity.fdtModelDir, fdtRealModelDir);
            String fkpRealModelDir = getCacheDir() + "/" + SettingsActivity.fkpModelDir;
            Utils.copyDirectoryFromAssets(this, SettingsActivity.fkpModelDir, fkpRealModelDir);
            predictor.init(
                    fdtRealModelDir,
                    SettingsActivity.fdtCPUThreadNum,
                    SettingsActivity.fdtCPUPowerMode,
                    SettingsActivity.fdtInputScale,
                    SettingsActivity.fdtInputMean,
                    SettingsActivity.fdtInputStd,
                    SettingsActivity.fdtScoreThreshold,
                    fkpRealModelDir,
                    SettingsActivity.fkpCPUThreadNum,
                    SettingsActivity.fkpCPUPowerMode,
                    SettingsActivity.fkpInputWidth,
                    SettingsActivity.fkpInputHeight,
                    SettingsActivity.fkpInputMean,
                    SettingsActivity.fkpInputStd);
        }
    }
   ```java

* SettingActivity
    实现设置界面各个元素的更新与显示，如果新增/删除界面的某个元素，均在这个类里面实现
    备注：
        每个元素的ID和value是与`res/values/string.xml` 中的字符串一一对应，便于更新元素的value

* Native
    实现Java与C++端代码互传的桥梁功能
    备注：
        Java 的native方法和C++的native方法要一一对应
    
### C++端（native）
* Native
  实现Java与C++端代码互传的桥梁功能，将Java数值转换为c++数值，调用c++端的完成人脸关键点检测功能

* Pipeline
  实现输入预处理、推理执行和输出后处理的流水线处理，支持多个模型的串行处理

* Utils
  实现其他辅助功能，如NHWC格式转NCHW格式、字符串处理等

* FaceProcess
  实现人脸美妆处理， 如美白、瘦脸、摇头等功能

* 新增模型支持
  - 在Pipeline文件中新增模型的预测类，实现图像预处理、预测和图像后处理功能
  - 在Pipeline文件中`Pipeline`类添加该模型预测类的调用和处理

* 新增美妆功能
  - 在FaceProcess文件中，添加新增的美妆功能
  - 在Pipeline文件中的`VisualizeResults`函数中，添加新增美妆功能的调用



