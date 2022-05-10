# 人像分割 C++ API Demo 使用指南
在 IOS 上实现人像分割功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍人像分割 Demo 运行方法和如何在更新模型/输入/输出处理下，保证人像分割 Demo 仍可继续运行。

## 如何运行人像分割 Demo

### 环境准备

1. 在本地环境安装好 Xcode 工具，详细安装方法请见[Xcode 官网](https://developer.apple.com/cn/xcode/resources/)。
2. 准备一部 Iphone 手机，并在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的 `设置->通用->设备管理` 中选择本电脑并信任）

<p align="center">
<img width="500" height="300"  src="https://paddlelite-demo.bj.bcebos.com/demo/image_classification/docs_img/ios/Xcode-phone.jpg"/>
</p>

### 部署步骤

1. 人像分割 Demo 位于 `Paddle-Lite-Demo/human_segmentation/ios/human_segmentation`  目录
2.  cd `Paddle-Lite-Demo/libs` 目录，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库
3.  cd `Paddle-Lite-Demo/human_segmentation/assets` 目录，运行 `download.sh` 脚本，下载 OPT 优化后模型

```shell
cd Paddle-Lite-Demo/libs
# 下载所需要的 Paddle Lite 预测库
sh download.sh
cd ../human_segmentation/assets
# 下载OPT 优化后模型
sh download.sh
cd ..
```

4.  用 Xcode 打开  `human_segmentation/human_segmentation.xcodeproj`  文件，修改工程配置。依次修改  `General/Identity`  和 `Signing&Capabilities`  属性，替换为自己的工程代号和团队名称。（必须修改，不然无法通过编译）

    <p align="center">
    <img width="500" height="150" src="https://paddlelite-demo.bj.bcebos.com/demo/image_classification/docs_img/ios/Xcode1.png"/>
    </p>

    <p align="center">
    <img width="500" height="150" src="https://paddlelite-demo.bj.bcebos.com/demo/image_classification/docs_img/ios/Xcode2.png"/>
    </p>

5.  选中 `human_segmentation/third-party` 目录 ，右击选择 `Add Files to "third-party" ...`  选项，将预测库、Opencv库和 assets内容（模型、测试图片及标签文件）添加到工程中。操作过程如下图：
     
     <p align="center">
     <img width="300" height="400"  src="https://paddlelite-demo.bj.bcebos.com/demo/image_classification/docs_img/ios/ios_add_file.jpg"/>
     </p>

    - 添加  `assets ` 案例
    
    <p align="center">
    <img width="400" height="100"  src="https://paddlelite-demo.bj.bcebos.com/demo/human_segmentation/doc_images/ios/ios_add_assets.jpg"/>
    </p>
   
    - 添加预测库案例
      
      <p align="center">
      <img width="400" height="80" src="https://paddlelite-demo.bj.bcebos.com/demo/human_segmentation/doc_images/ios/ios_add_lib.jpg"/>
      </p>

    - 添加完成后，工程目录如下：
      
      <p align="center">
      <img width="200" height="180" src="https://paddlelite-demo.bj.bcebos.com/demo/human_segmentation/doc_images/ios/ios_add_finish.jpg"/>
      </p>
     
     **注意：**
        如果觉得上述方法比较麻烦，可以使用工程下的 `prepare.sh` 脚本，完成上述资源的拷贝
        
      ```shell
        # path = Paddle-Lite-Demo/ios/human_segmentation
        sh prepare.sh
      ```
        
6.  IPhone 手机连接电脑，在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的 `设置->通用->设备管理` 中选择本电脑并信任）

<p align="center">
<img width="500" height="300"  src="https://paddlelite-demo.bj.bcebos.com/demo//image_classification/docs_img/ios/Xcode-phone.jpg"/>
</p>


7. 按下左上角的 Run 按钮，自动编译 APP 并安装到手机。在苹果手机中设置信任该 APP（进入 `设置->通用->设备管理`，选中新安装的 APP 并 `验证该应用`）
成功后效果如下：

  | APP 效果 |
  | ---     |
<p align="center">
<img width="300" height="400"  src="https://paddlelite-demo.bj.bcebos.com/demo/human_segmentation/doc_images/ios/app_interface.jpg"/>
</p>

## 更新预测库

*Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
* 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 iOS 预测库
* 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
  * 替换 c++ 库
       * 头文件
         将生成的 `build.lite.ios.xxx.clang/inference_lite_lib.ios64.xxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/human_segmentation/ios/human_segmentation/human_segmentation/third-party/PaddleLite/include`
       * 替换 arm64-v8a 库
         将生成的 `build.lite.ios.ios64.armv8/inference_lite_lib.ios64.armv8/libs/libpaddle_api_light_bundled.a` 库替换 Demo 中的 `Paddle-Lite-Demo/human_segmentation/ios/human_segmentation/human_segmentation/third-party/PaddleLite/lib/libpaddle_api_light_bundled.a`

>**注意：**
> 如果要使用 armv7 库，则可将 armv7 库替换至相应目录下：
> * armeabi-v7a
>  将生成的 `build.lite.ios.ios.armv7/inference_lite_lib.ios.armv7/libs/libpaddle_api_light_bundled.a` 库替换 Demo 中的 `Paddle-Lite-Demo/human_segmentation/ios/human_segmentation/human_segmentation/third-party/PaddleLite/lib/libpaddle_api_light_bundled.a`
  
  
## Demo 内容介绍

先整体介绍下目标检测 Demo 的代码结构，然后再介绍 Demo 每部分功能.

### 整体结构介绍
1.  `third-party`： 存放预测库、模型、测试图片等相关信息
      * `assets`: 存放预测资源
        - models：模型文件，opt 工具转化后 Paddle Lite 模型
        - images：测试图片
        - labels：标签文件
      * `PaddleLite`：存放 Paddle Lite 预测库和头文件
        - lib
        - include
      * `opencv2.framework`：opencv  库和头文件

    ```shell
    # 位置：
    human_segmentation/third-party/
    example：
    # IOS 预测库
    human_segmentation/third-party/PaddleLite/lib/libpaddle_api_light_bundled.a
    # 预测库头文件
    human_segmentation/third-party/PaddleLite/include/paddle_api.h
    human_segmentation/third-party/PaddleLite/include/paddle_use_kernels.h
    human_segmentation/third-party/PaddleLite/include/paddle_use_ops.h
    ```

 3.  `ViewController.mm`：主要预测代码

  ```shell
    # 位置
    human_segmentation/ViewController.mm
  ``` 

### `ViewController.mm`  文件内容介绍

 * `viewDidLoad`  方法
    APP 界面初始化、推理引擎 predictor 创建和运行方法，这个方法包含界面参数获取、predictor 构建和运行、图像前/后处理等内容
   
 * `processImage` 方法
   实现图像输入变化时，进行新的推理，并获取相应的输出结果

* `preprocess` 方法
   输入预处理操作

* `postprocess` 方法
   输出后处理操作中

## 代码讲解 （使用 Paddle Lite `C++ API` 执行预测）

IOS 示例基于 C++ API 开发，调用 Paddle Lite `C++s API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite C++ API ](https://paddle-lite.readthedocs.io/zh/latest/api_reference/c++_api_doc.html)。

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

## 如何更新模型和输入/输出预处理

### 更新模型
1. 将优化后的模型存放到目录 `third-party/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `third-party/assets/models/model.nb`，则代码不需更新；否则话，需要修改 `./ViewController.mm` 中代码

  以更新 mobilenet_v2 模型为例，则先将优化后的模型存放到 `third-party/assets/models/mobilenet_v2_for_cpu/mv2.nb` 下，然后更新代码

```c++
// 代码文件 `human_segmentation/ViewController.mm`
- (void)viewDidLoad {
...
MobileConfig config;
// old
// config.set_model_from_file(app_dir+ "/models/mobilenet_v1_for_cpu/model.nb");
// update now
config.set_model_from_file(app_dir+ "/models/mobilenet_v2_for_cpu/mv2.nb");
predictor = CreatePaddlePredictor<MobileConfig>(config);
...
}
```

**注意：**

 - 如果更新后模型的输入信息如Shape、Tensor个数等发生改变，需要更新 `ViewController.mm` 文件中 `preprocess(...)` 输入预处理方法，完成模型输入更新
 - 如果更新后模型的输出信息发生改变，需要更新 `ViewController.mm` 文件中 `postprocess(...)` 输出后处理方法，完成模型输出更新即可
 
- 如果需要更新 `label.txt`，则需将更新后的标签文件，存放至`third-party/assets/labels/` 目录下。
若更新后标签名字不一样，应修改代码文件 `./ViewController.mm` 中代码 

```c++
// 代码文件 `human_segmentation/ViewController.mm`
- (void)viewDidLoad {
...
// old
// std::string label_file_str = app_dir+"/labels/labels.txt";
// update now
std::string label_file_str = app_dir+"/labels/labels_new.txt";
self.labels = [self load_labels:label_file_str];
...
}
```

### 更新输入/输出预处理
1. 更新输入数据

- 将更新的图片存放在 `third-party/assets/images/` 下；
- 更新文件 `detection_demo/ViewController.mm` 中的代码

以更新 `dog.jpg` 为例，则先将 `dog.jpg` 存放在 `third-party/assets/images/` 下，然后更新代码

```c++
// 代码文件 `human_segmentation/ViewController.mm`
- (void)viewDidLoad {
...
// old
// _image = [UIImage imageNamed:@"third-party/assets/images/tabby_cat.jpg"];
// now
_image = [UIImage imageNamed:@"third-party/assets/images/human.jpg"];
if (_image != nil) {
    printf("load image successed\n");
    imageView.image = _image;
} else {
    printf("load image failed\n");
}
...
}
```

**注意：** 本 Demo 是支持图片/视频流/拍照三种输入方式，如果需更新输入图片建议通过 APP 的拍照或视频流方式进行更新，这样不用修改代码，则能正常推理。


2. 更新输入预处理
此处需要更新 `human_segmentation/ViewController.mm` 中的 `preprocess(...)` 输入预处理方法

3. 更新输出预处理
此处需要更新 `human_segmentation/ViewController.mm` 中的 `postprocess(...)` 输出后处理方法

### 其他文件
* `time.h` 包含常见的计时处理函数，用于计时处理

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。
