# 目标检测 C++ API Demo 使用指南
 在 IOS 上实现实时的目标检测功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
 本文主要介绍目标检测 Demo 运行方法和如何在更新模型/输入/输出处理下，保证目标检测 Demo 仍可继续运行。

 ## 如何运行目标检测 Demo
 ### 环境准备

 1. 在本地环境安装好 Xcode 工具，详细安装方法请见[Xcode 官网](https://developer.apple.com/cn/xcode/resources/)。
 2. 准备一部 Iphone 手机，并在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的 `设置->通用->设备管理` 中选择本电脑并信任）

 <p align="center"><img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode-phone.jpg"/>

 ### 部署步骤

 1. 目标检测 Demo 位于 `Paddle-Lite-Demo/PaddleLite-ios-demo/ios-detection_demo`  目录
 2. 在终端中执行  `download_dependencies.sh`  脚本自动下载模型和 Paddle Lite 预测库

 ```shell
 cd PaddleLite-ios-demo          # 1. 终端中进入 Paddle-Lite-Demo\PaddleLite-ios-demo
 sh download_dependencies.sh     # 2. 执行脚本下载依赖项 （需要联网）
 ```

 下载完成后会出现提示： `Extract done `

 3. 用 Xcode 打开  `ios-detection_demo/detection_demo.xcodeproj`  文件，修改工程配置。依次修改  `General/Identity`  和 `Signing&Capabilities`  属性，替换为自己的工程代号和团队名称。（必须修改，不然无法通过编译）

 ![Xcode1](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode1.png)

 ![Xcode2](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode2.png)

 4.  IPhone 手机连接电脑，在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的 `设置->通用->设备管理` 中选择本电脑并信任）

 <p align="center"><img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode-phone.jpg"/>

 5. 按下左上角的 Run 按钮，自动编译 APP 并安装到手机。在苹果手机中设置信任该 APP（进入 `设置->通用->设备管理`，选中新安装的 APP 并 `验证该应用`）

 成功后效果如下，图一：APP安装到手机        图二： APP打开后的效果，会自动识别图片中的物体并标记

 <p align="center"><img width="350" height="500"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/IOS2.jpeg"/>&#8194;&#8194;&#8194;&#8194;&#8194;<img width="350" height="500"  src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/IOS3.jpeg"/></p>

 ## 更新预测库

 * Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
  * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)，编译 IOS 预测库
  * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
         * 头文件
           将生成的 `build.lite.ios.xxx.clang/inference_lite_lib.ios64.xxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-ios-demo/ios-detection_demo/detection_demo/include`
         * 替换 arm64-v8a 库
           将生成的 `build.lite.ios.ios64.armv8/inference_lite_lib.ios64.armv8/libs/libpaddle_api_light_bundled.a` 库替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-ios-demo/ios-detection_demo/detection_demo/lib/libpaddle_api_light_bundled.a`

 >**注意：**
 >> 如果要使用 armv7 库，则可将 armv7 库替换至相应目录下：
 >> * armeabi-v7a
 >>  将生成的 `build.lite.ios.ios.armv7/inference_lite_lib.ios.armv7/libs/libpaddle_api_light_bundled.a` 库替换 Demo 中的 `Paddle-Lite-Demo/PaddleLite-ios-demo/ios-detection_demo/detection_demo/lib/libpaddle_api_light_bundled.a`
   
 ## Demo 内容介绍

 先整体介绍下目标检测 Demo 的代码结构，然后再介绍 Demo 每部分功能.

 <p align="center"><img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/IOS-struct.png"/>

 ### 整体结构介绍
 1.  `mobilenetv1-ssd`： 模型文件( opt 工具转化后 Paddle Lite 模型)

 ```shell
 # 位置：
 ios-detection_demo/detection_demo/models/mobilenetv1-ssd
 ```

  2. `libpaddle_api_light_bundled.a`、`paddle_api.h`：Paddle-Lite C++ 预测库和头文件

 ```shell
 # 位置：
 # IOS 预测库
 ios-detection_demo/detection_demo/lib/libpaddle_api_light_bundled.a
 # 预测库头文件
 ios-detection_demo/detection_demo/include/paddle_api.h
 ios-detection_demo/detection_demo/include/paddle_use_kernels.h
 ios-detection_demo/detection_demo/include/paddle_use_ops.h
 ```

  3.  `ViewController.mm`：主要预测代码

 ```shell
 # 位置
 ios-detection_demo/detection_demo/ViewController.mm
 ``` 

 ### `ViewController.mm`  文件内容介绍

  * `viewDidLoad`  方法
    APP 界面初始化、推理引擎 predictor 创建和运行方法，这个方法包含界面参数获取、predictor 构建和运行、图像前/后处理等内容

  * `processImage` 方法
    实现图像输入变化时，进行新的推理，并获取相应的输出结果

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

 // 例如目标检测：输出后处理，输出检测结果
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
 1. 将优化后的模型存放到目录 `detection_demo/models/` 下；
 2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `detection_demo/models/mobilenetv1-ssd/model.nb`，则代码不需更新；否则话，需要修改 `detection_demo/ViewController.mm` 中代码和将新模型路径添加到 `Build Phases-> Copy Bundle Resource` 中

 <p align="centet">
 <img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/model_change_0.png"/>
 </p>

 <p align="centet">
 <img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/model_change_1.png"/>
 </p>

 以更新 ssd_mobilenet_v3 模型为例，则先将优化后的模型存放到 `detection_demo/models/ssd_mobilenet_v3_for_cpu/ssd_mv3.nb` 下，然后更新代码

 ```c++
 // 代码文件 `detection_demo/ViewController.mm`
 - (void)viewDidLoad {
 ...
 NSString *path = [[NSBundle mainBundle] bundlePath];
 std::string paddle_mobilenetv1_dir = std::string([path UTF8String]);
 MobileConfig config;
 // config.set_model_from_file(paddle_mobilenetv1_dir + "/model.nb");
 config.set_model_from_file(paddle_mobilenetv1_dir + "/ssd_mv3.nb");
 net_mbv1 = CreatePaddlePredictor<MobileConfig>(config);
 ...
 }
 ```
**注意：**

- 如果模型的输入和输出个数、shape、数据类型有更新，也需要更新 `detection_demo/ViewController.mm` 文件中 `viewDidLoad` 方法

<p align="centet">
<img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/model_inpute_change.png"/>
</p>

 - 如果需要更新 `label`，则需要修改代码文件 `detection_demo/ViewController.mm` 中的  `class_names`  常量

 ```c++
 // 代码文件 `sdetection_demo/ViewController.mm`
 const char* class_names[] = {
     "background",
     "aeroplane",
     "bicycle",
     "bird",
     "boat",
     "bottle",
     "bus",
     "car",
     "cat",
     "chair",
     "cow",
     "diningtable",
     "dog",
     "horse",
     "motorbike",
     "person",
     "pottedplant",
     "sheep",
     "sofa",
     "train",
     "tvmonitor"
 };
 ```

 ### 更新输入/输出预处理
 1. 更新输入数据

 - 将更新的图片存放在 `detection_demo/images/` 下；
 - 将新图片的路径添加到 `Build Phases-> Copy Bundle Resource` 中
 - 更新文件 `detection_demo/ViewController.mm` 中的代码

 以更新 `dog.jpg` 为例，则先将 `dog.jpg` 存放在 `detection_demo/images/` 下，然后更新代码

 <p align="centet">
 <img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/input_change_pic.png"/>
 </p>

 <p align="centet">
 <img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/input_change_0.png"/>
 </p>

 ```c++
 // 代码文件 `detection_demo/ViewController.mm`
 - (void)viewDidLoad {
 ...
 // _image = [UIImage imageNamed:@"cat.jpg"];
 _image = [UIImage imageNamed:@"dog.jpg"];
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
 此处需要更新 `detection_demo/ViewController.mm` 中的输入预处理方法

 <p align="centet">
 <img src="https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/input_process.png"/>
 </p>

 3. 更新输出预处理
 此处需要更新 `detection_demo/ViewController.mm` 中的 `detect_object(const float* data, int count,const std::vector<std::vector<uint64_t>>& lod, const float thresh,Mat& image)` 方法

 ### 其他文件
 * `time.h` 包含常见的计时处理函数，用于计时处理
