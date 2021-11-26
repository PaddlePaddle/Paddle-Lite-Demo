# Paddle-Lite-Demo

Paddle Lite 提供了图像分类、目标检测、人像分割、人脸关键点检测、口罩识别、OCR 检测和识图这几大类应用案例，每类应用案例将从三个 OS 端：Android、IOS、Linux（ArmLinux 和 Shell） 分别提供预测 Demo 案例。

## 应用 Demo 示例

* 图像分类
  * Android
    * 基于MobileNetV1的图像分类
  * IOS
    * 基于MobileNetV1的图像分类
  * ARMLinux
    * 基于MobileNetV1的图像分类
  * Shell
    * 基于MobileNetV1的图像分类
* 目标检测
  * Android
    * 基于MobileNetV1-SSD的目标检测
    * 基于YOLOV3-MobileNetV3的目标检测
    * 基于Ultra-Light-Fast-Generic-Face-Detector-1MB的人脸检测
  * IOS
    * 基于MobileNetV1-SSD的目标检测
  * ARMLinux
    * 基于MobileNetV1-SSD的目标检测
  * Shell
    * 基于MobileNetV1-SSD的目标检测
* 人像分割
  * Android
    * 基于DeeplabV3+MobilNetV2的人像分割
  * IOS
    * 待提供
  * ARMLinux
    * 待提供
  * Shell
    * 待提供
* 人脸关键点检测
  * Android
    * 基于视频流的人脸关键点检测
  * IOS
    * 待提供
  * ARMLinux
    * 待提供
  * Shell
    * 待提供
* 口罩识别
  * Android
    * 基于视频流的人脸检测+口罩识别
  * IOS
    * 待提供
  * ARMLinux
    * 待提供
  * Shell
    * 待提供
* OCR 检测
  * Android
    * 待提供
  * IOS
    * 待提供
  * ARMLinux
    * 待提供
  * Shell
    * 待提供
* 识图
  * Android
    * 待提供
  * IOS
    * 待提供
  * ARMLinux
    * 待提供
  * Shell
    * 待提供

## Demo 示例文档
关于 Paddle Lite 和示例，请参考本文剩余章节和如下文档链接：
- [文档官网](https://paddle-lite.readthedocs.io/zh/latest/index.html)
- [Android 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/android_app_demo.html)
  APK 下载文件：下载 APK，然后再手机 install，即可运行相关 APP
  - [[图像分类]](https://paddlelite-demo.bj.bcebos.com/apps/android/mobilenet_classification_demo.apk)  
  - [[目标检测]](https://paddlelite-demo.bj.bcebos.com/apps/android/yolo_detection_demo.apk) 
  - [[口罩检测]](https://paddlelite-demo.bj.bcebos.com/apps/android/mask_detection_demo.apk)  
  - [[人脸关键点]](https://paddlelite-demo.bj.bcebos.com/apps/android/face_keypoints_detection_demo.apk) 
  - [[人像分割]](https://paddlelite-demo.bj.bcebos.com/apps/android/human_segmentation_demo.apk)
- [iOS 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/ios_app_demo.html)
- [ARMLinux 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/linux_arm_demo.html)
- [X86 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/x86.html)
- [OpenCL 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/opencl.html)
- [FPGA 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/fpga.html)
- [华为 NPU 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/huawei_kirin_npu.html)
- [百度 XPU 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/baidu_xpu.html)
- [瑞芯微 NPU 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/rockchip_npu.html)
- [联发科 APU 示例](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/mediatek_apu.html)

## 环境要求

* iOS
    * macOS+Xcode，已验证的环境：Xcode Version 11.5 (11E608c) on macOS Catalina(10.15.5)
    * Xcode 11.3会报"Invalid bitcode version ..."的编译错误，请将Xcode升级到11.4及以上的版本后重新编译
    * 对于ios 12.x版本，如果提示“xxx.  which may not be supported by this version of Xcode”，请下载对应的[工具包]( https://github.com/iGhibli/iOS-DeviceSupport), 下载完成后解压放到/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/DeviceSupport目录，重启xcode

* Android
    * Android Studio 4.2；
    * adb调试工具；
    * Android手机或开发版；
    * 华为手机支持NPU的[ Demo](https://paddlelite-demo.bj.bcebos.com/devices/huawei/kirin/PaddleLite-android-demo_v2_9_0.tar.gz)（NPU的功能暂时只在nova5、mate30和mate30 5G上进行了测试，用户可自行尝试其它搭载了麒麟810和990芯片的华为手机（如nova5i pro、mate30 pro、荣耀v30，mate40或p40，且需要将系统更新到最新版）

* ARMLinux
    * RK3399（[Ubuntu 18.04](http://www.t-firefly.com/doc/download/page/id/3.html)） 或 树莓派3B（[Raspbian Buster with desktop](https://www.raspberrypi.org/downloads/raspbian/)），暂时验证了这两个软、硬件环境，其它平台用户可自行尝试；
    * 支持树莓派3B摄像头采集图像，具体参考[树莓派3B摄像头安装与测试](/PaddleLite-armlinux-demo/enable-camera-on-raspberry-pi.md)
    * gcc g++ opencv cmake的安装（以下所有命令均在设备上操作）
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

## 更新到最新的预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
* 参考 [Paddle Lite 文档](https://paddle-lite.readthedocs.io/zh/latest/guide/introduction.html)，编译 IOS 预测库、Android 和ARMLinux 预测库
* 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`

### IOS 更新预测库

* 替换库文件：产出的 `lib` 目录替换 `Paddle-Lite-Demo/PaddleLite-ios-demo/ios-classification_demo/classification_demo/lib` 目录
* 替换头文件：产出的 `include` 目录下的文件替换 `Paddle-Lite-Demo/PaddleLite-ios-demo/ios-classification_demo/classification_demo/paddle_lite` 目录下的文件

### Android 更新预测库

* 替换 jar 文件：将生成的 `build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/java/jar/PaddlePredictor.jar` 替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-android-demo/image_classification_demo/app/libs/PaddlePredictor.jar`
* 替换 arm64-v8a jni 库文件：将生成 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so` 库替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-android-demo/image_classification_demo/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so`
* 替换 armeabi-v7a jni 库文件：将生成的 `build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/java/so/libpaddle_lite_jni.so` 库替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-android-demo/image_classification_demo/app/src/main/jniLibs/armeabi-v7a/libpaddle_lite_jni.so`.

### ARMLinux 更新预测库

* 替换头文件目录，将生成的 cxx 中的 `include` 目录替换 `Paddle-Lite-Demo/PaddleLite-armlinux-demo/Paddle-Lite/include` 目录；
* 替换 armv8 动态库，将生成的 `cxx/libs` 中的 `libpaddle_light_api_shared.so`替换`Paddle-Lite-Demo/PaddleLite-armlinux-demo/Paddle-Lite/libs/armv8/libpaddle_light_api_shared.so`；
* 替换 armv7hf 动态库，将生成的 `cxx/libs` 中的 `libpaddle_light_api_shared.so` 替换 `Paddle-Lite-Demo/PaddleLite-armlinux-demo/Paddle-Lite/libs/armv7hf/libpaddle_light_api_shared.so`；

## 效果展示

* iOS
    * 基于 MobileNetV1 的图像分类

    ![ios_static](https://paddlelite-demo.bj.bcebos.com/doc/ios_static.jpg)      
    ![ios_video](https://paddlelite-demo.bj.bcebos.com/doc/ios_video.jpg)

    * 基于 MobileNetV1-SSD 的目标检测

    ![ios_static](https://paddlelite-demo.bj.bcebos.com/doc/ios-image-detection.jpg)      
    ![ios_video](https://paddlelite-demo.bj.bcebos.com/doc/ios-video-detection.jpg)

* Android
    * 基于 MobileNetV1 的图像分类（CPU 预测结果，测试环境：华为 nova5）

      ![android_image_classification_cat_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_image_classification_cat_cpu.jpg)
      ![android_image_classification_keyboard_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_image_classification_keyboard_cpu.jpg)

    * 基于 MobileNetV1-SSD 的目标检测（CPU 预测结果，测试环境：华为 nova5）

      ![android_object_detection_npu](https://paddlelite-demo.bj.bcebos.com/doc/android_object_detection_cpu.jpg)

    * 基于 Ultra-Light-Fast-Generic-Face-Detector-1MB 的人脸检测（CPU 预测结果，测试环境：华为 nova5）

      ![android_face_detection_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_face_detection_cpu.jpg)

    * 基于 DeeplabV3+MobilNetV2 的人像分割（CPU 预测结果，测试环境：华为 nova5）
      
      ![android_human_segmentation_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_human_segmentation_cpu.jpg)

    * 基于视频流的人脸检测+口罩识别（CPU 预测结果，测试环境：华为 mate30）
      
      ![android_mask_detection_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_mask_detection_cpu.jpg)

    * 基于视频流的人脸关键点检测（CPU 预测结果，测试环境：OnePlus 7）
      
      ![android_face_keypoints_detection_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_face_keypoints_detection_cpu.jpg)

    * 基于 YOLOV3-MobileNetV3 的目标检测（CPU 预测结果，测试环境：华为 p40）
      
      ![android_yolo_detection_cpu](https://paddlelite-demo.bj.bcebos.com/doc/android_yolo_detection_cpu.jpg)

* ARMLinux
     * 基于 MobileNetV1 的图像分类

     ![armlinux_image_classification_raspberry_pi](https://paddlelite-demo.bj.bcebos.com/doc/armlinux_image_classification.jpg)

     * 基于 MobileNetV1-SSD 的目标检测

     ![armlinux_object_detection_raspberry_pi](https://paddlelite-demo.bj.bcebos.com/doc/armlinux_object_detection.jpg)
