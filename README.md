# Paddle-Lite-Demo


## 功能
* iOS示例: 静态图像目标分类和视频流目标分类；
* Android示例: 基于MobileNetV1的图像分类示例程序；

## 要求

* iOS
    * Mac机器，需要有xcode环境（已验证：Xcode Version 10.1 (10B61)
    * 对于ios 12.x版本，如果提示“xxx.  which may not be supported by this version of Xcode”，请下载对应的[工具包]( https://github.com/iGhibli/iOS-DeviceSupport), 下载完成后解压放到/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/DeviceSupport目录，重启xcode

* Android
    * Android Studio 3.4
    * Android手机或开发版，NPU功能暂时只在麒麟810芯片的华为手机（如Nova5系列）进行了测试，使用前请将EMUI更新到最新版本；


## 安装
$ git clone https://github.com/PaddlePaddle/Paddle-Lite-Demo

* iOS
    * 打开xcode，点击“Open another project…”打开Paddle-Lite-Demo/ios-classification_demo/目录下的xcode工程；
    * 在选中左上角“project navigator”，选择“classification_demo”，修改“General”信息；
    * 插入ios真机（已验证：iphone8， iphonexr），选择Device为插入的真机；
    * 点击左上角“build and run”按钮；

* Android
    * 打开Android Studio，在"Welcome to Android Studio"窗口点击"Open an existing Android Studio project"，在弹出的路径选择窗口中进入"PaddleLite-android-demo"目录，然后点击右下角的"Open"按钮即可导入工程
    * 通过USB连接Android手机或开发版；
    * 载入工程后，点击菜单栏的Run->Run 'App'按钮，在弹出的"Select Deployment Target"窗口选择已经连接的Android设备，然后点击"OK"按钮；
    * 手机上会出现Demo的主界面，选择第一个"Image Classification"图标，进入基于MobileNetV1的图像分类Demo，注："Object Detection"的Demo正在开发中，请忽略；
    * 在图像分类Demo中，默认会载入一张猫的图像，并会在图像下方给出CPU的预测结果，如果你使用的是麒麟810芯片的华为手机（如Nova5系列），可以通过按下右上角的"NPU"按钮切换成NPU进行预测；
    * 在图像分类Demo中，你还可以通过上方的"Gallery"和"Take Photo"按钮从相册或相机中加载测试图像；

## 效果展示

* iOS
    * 静态图识别

    ![](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite-Demo/master/doc/ios_static.jpg)

    * 动态图识别

    ![](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite-Demo/master/doc/ios_video.jpg)

* Android
    * CPU预测结果（测试环境：华为nova5）

    ![](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite-Demo/master/doc/android_cat_cpu.jpg)      ![](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite-Demo/master/doc/android_keyboard_cpu.jpg)

    * NPU预测结果（测试环境：华为nova5）

    ![](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite-Demo/master/doc/android_cat_npu.jpg)      ![](https://raw.githubusercontent.com/PaddlePaddle/Paddle-Lite-Demo/master/doc/android_keyboard_npu.jpg)
