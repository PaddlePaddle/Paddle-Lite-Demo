# PP-Tiny-Pose Demo 使用指南
  在 Android 上实现实时的Pose关键点的检测功能
  
  
## 如何运行 Demo

## 要求
* Android
    * 在本地环境安装好 Android Studio 工具，详细安装方法请见[Android Stuido 官网](https://developer.android.com/studio)
    * Android手机或开发版，NPU的功能暂时只在nova5、mate30和mate30 5G上进行了测试，用户可自行尝试其它搭载了麒麟810和990芯片的华为手机（如nova5i pro、mate30 pro、荣耀v30，mate40或p40，且需要将系统更新到最新版）
    * 打开 Android 手机的 USB 调试模式，开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

**注意**：如果您的 Android Studio 尚未配置 NDK ，请根据 Android Studio 用户指南中的[安装及配置 NDK 和 CMake ](https://developer.android.com/studio/projects/install-ndk)内容，预先配置好 NDK 。您可以选择最新的 NDK 版本，或者使用
Paddle Lite 预测库版本一样的 NDK

## 安装

 * Android
    * 打开 Android Studio，在 "Welcome to  Android Studio" 窗口点击 "Open an existing Android Studio project"，在弹出的路径选择窗口中进入 "face_keypoints_detection_demo" 目录，然后点击右下角的 "Open" 按钮即可导入工程
    * 通过 USB 连接 Android 手机或开发板；
    * 载入工程后，点击菜单栏的 Run->Run `App` 按钮，在弹出的 "Select Deployment Target" 窗口选择已经连接的 Android 设备，然后点击 "OK" 按钮；
    * 由于 Demo 所用到的库和模型均通过 `app/build.gradle` 脚本在线下载，因此，第一次编译耗时较长（取决于网络下载速度），请耐心等待；
    * 如果库和模型下载失败，建议手动下载并拷贝到相应目录下
    
    <p align="center">
    <img src="./images/run_app.jpg"/>
    </p>
    
    > **注意：**
    >> 如果您在导入项目、编译或者运行过程中遇到 NDK 配置错误的提示，请打开 ` File > Project Structure > SDK Location`，修改 `Andriod NDK location` 为您本机配置的 NDK 所在路径。
    >> 如果您是通过 Andriod Studio 的 SDK Tools 下载的 NDK (见本章节"环境准备")，可以直接点击下拉框选择默认路径。
    >> 还有一种 NDK 配置方法，你可以在 `face_keypoints_detection_demo/local.properties` 文件中手动完成 NDK 路径配置，如下图所示
    >> 如果以上步骤仍旧无法解决 NDK 配置错误，请尝试根据 Andriod Studio 官方文档中的[更新 Android Gradle 插件](https://developer.android.com/studio/releases/gradle-plugin?hl=zh-cn#updating-plugin)章节，尝试更新Android Gradle plugin版本。
    

## 更新到PaddleLite-2.11-rc

下载PaddleLite-2.11-rc.zip，并解压到项目目录。解压后目录应该如下所示。

```
├── app
│   ├── build.gradle
│   ├── PaddleLite-2.11-rc
│   │   ├── include
│   │   │   ├── paddle_api.h
│   │   │   ├── paddle_image_preprocess.h
│   │   │   ├── paddle_lite_factory_helper.h
│   │   │   ├── paddle_place.h
│   │   │   ├── paddle_use_kernels.h
│   │   │   ├── paddle_use_ops.h
│   │   │   └── paddle_use_passes.h
│   │   └── lib
│   │       └── libpaddle_light_api_shared.so
│   └── src
...
```
