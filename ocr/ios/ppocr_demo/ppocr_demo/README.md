# OCR 文字识别 Demo 使用指南
在 iOS 上实现实时的 OCR 文字识别功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。
本文主要介绍 OCR 文字识别 Demo 运行方法和如何在更新模型/输入/输出处理下，保证 OCR 文字识别 Demo 仍可继续运行。

## 如何运行目标检测 Demo
### 环境准备

1. 在本地环境安装好 Xcode 工具，详细安装方法请见[Xcode 官网](https://developer.apple.com/cn/xcode/resources/)。
2. 准备一部 Iphone 手机，并在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的 `设置->通用->设备管理` 中选择本电脑并信任）

<p align="center"><img width="600" height="250"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/ios/Xcode-phone.jpg"/>

### 部署步骤

1.  OCR 文字识别 Demo 位于 `Paddle-Lite-Demo/ocr/ios/ppocr_demo`  目录
2.  cd `Paddle-Lite-Demo/libs` 目录，运行 `download.sh` 脚本，下载所需要的 Paddle Lite 预测库
3.  cd `Paddle-Lite-Demo/ocr/assets` 目录，运行 `download.sh` 脚本，下载 OPT 优化后模型、测试图片和标签文件
4. cd  `Paddle-Lite-Demo/ocr/ios/ppocr_demo/` 目录，运行 `prepare.sh` 脚本，将工程所需的资源拷贝至当前工程目录下

```shell
cd Paddle-Lite-Demo/libs
# 下载所需要的 Paddle Lite 预测库
sh download.sh
cd ../ocr/assets
# 下载OPT 优化后模型、测试图片、标签文件及 config 文件
sh download.sh
cd ../ios/ppocr_demo
# 将工程所需的资源拷贝至当前工程目录下
sh prepare.sh
```

5. 用 Xcode 打开  `ppocr_demo/ppocr_demo.xcodeproj`  文件，修改工程配置。依次修改  `General/Identity`  和  `Signing&Capabilities`  属性，替换为自己的工程代号和团队名称。（必须修改，不然无法通过编译）

![Xcode1](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode1.png)

![Xcode2](https://paddlelite-data.bj.bcebos.com/doc_images/Android_iOS_demo/iOS/Xcode2.png)

6.  IPhone 手机连接电脑，在 Xcode 中连接自己的手机 （第一次连接 IPhone 到电脑时，需要在 IPhone 的 `设置->通用->设备管理` 中选择本电脑并信任）

<p align="center"><img width="600" height="250"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/ios/Xcode-phone.jpg"/>

7. 按下左上角的 Run 按钮，自动编译 APP 并安装到手机。在苹果手机中设置信任该 APP（进入 `设置->通用->设备管理`，选中新安装的 APP 并 `验证该应用`）

成功后效果如下，图一：APP安装到手机        图二： APP打开后的效果，会自动识别图片中的物体并标记

  | APP 图标 | APP 效果 |
  | ---     | --- |
  | ![app_pic](https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/ios/IOS2.jpeg)   | ![app_res](https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/ios/run_app.jpeg) |

## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://www.paddlepaddle.org.cn/lite/develop/source_compile/compile_env.html)，编译 iOS 预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
   * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.ios.xxx.clang/inference_lite_lib.ios64.xxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/ocr/iOS/ppocr_demo/ppocr_demo/include`
        * 替换 arm64-v8a 库
          将生成的 `build.lite.ios.ios64.armv8/inference_lite_lib.ios64.armv8/libs/libpaddle_api_light_bundled.a` 库替换 Demo 中的 `Paddle-Lite-Demo/ocr/iOS/ppocr_demo/ppocr_demo/lib/libpaddle_api_light_bundled.a`

>**注意：**
>> 如果要使用 armv7 库，则可将 armv7 库替换至相应目录下：
>> * armeabi-v7a
>>  将生成的 `build.lite.ios.ios.armv7/inference_lite_lib.ios.armv7/libs/libpaddle_api_light_bundled.a` 库替换 Demo 中的 `Paddle-Lite-Demo/ocr/iOS/ppocr_demo/ppocr_demo/lib/libpaddle_api_light_bundled.a`
  
## Demo 内容介绍

Demo 的整体目录结构如下图所示：

<p align="center"><img src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/android/predict_android_shell.jpg"/></p>

1. `ppocr_demo/lib` : 存放编译好的预测库

**备注：**
  如需更新预测库，例如更新 iOS v8 预测库 ，则将新的预测库更新到 `ppocr_demo/lib` 目录

2. `ppocr_demo/assets` : 存放 OCR demo 的模型、测试图片、标签文件及 config 文件

**备注：**

 - `./assets/labels/ppocr_keys_v1.txt` 是中文字典文件，如果使用的 模型是英文数字或其他语言的模型，需要更换为对应语言的字典.
 - 其他语言的字典文件，可从 PaddleOCR 仓库下载：`https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.3/ppocr/utils`
 - `./assets/labels/config.txt` 字段含义:
 
```shell
 max_side_len  960         # 输入图像长宽大于 960 时，等比例缩放图像，使得图像最长边为 960
 det_db_thresh  0.3        # 用于过滤 DB 预测的二值化图像，设置为 0.3 对结果影响不明显
 det_db_box_thresh  0.5    # DB 后处理过滤 box 的阈值，如果检测存在漏框情况，可酌情减小
 det_db_unclip_ratio  1.6  # 表示文本框的紧致程度，越小则文本框更靠近文本
 use_direction_classify  0  # 是否使用方向分类器，0 表示不使用，1 表示使用
```

3. `./ppocr_demo/` :  存放预测代码
    - `cls_process.cc` :  方向分类器的推理全流程，包含预处理、预测和后处理三部分
    - `rec_process.cc` :  识别模型 CRNN 的推理全流程，包含预处理、预测和后处理三部分
    - `det_process.cc` :  检测模型 CRNN 的推理全流程，包含预处理、预测和后处理三部分
    - `det_post_process` :  检测模型 DB 的后处理文件
    - `pipeline.cc` :  OCR 文字识别 Demo 推理全流程代码
    - `utils.cc` :  Tensor 相关处理代码
    - `time.cc` :  计时代码
    - `ViewController.mm` ：功能界面主函数界面，完成 APP 界面初始化、`Pipeline` 类的创建和运行方法及结果显示


## 代码讲解 （使用 Paddle Lite `C++ API` 执行预测）

iOS 示例基于 C++ API 开发，调用 Paddle Lite `C++s API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite C++ API ](https://www.paddlepaddle.org.cn/lite/develop/api_reference/cxx_api_doc.html)。

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
1. 将优化后的模型存放到目录 `ppocr_demo/assets/models/` 下；
2. 如果模型名字跟工程中模型名字一模一样，即均是使用 `ppocr_demo/assets/models/***.nb`，则代码不需更新；否则话，需要修改 `ppocr_demo/ViewController.mm` 中代码和将新模型路径添加到 `Build Phases-> Copy Bundle Resource` 中

<p align="centet">
<img width="600" height="450"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/ios/model_change_0.png"/>
</p>

<p align="centet">
<img width="600" height="450"  src="https://paddlelite-demo.bj.bcebos.com/demo/ocr/docs_img/ios/model_change_1.png"/>
</p>

以更新 ssd_mobilenet_v3 模型为例，则先将优化后的模型存放到 `ppocr_demo/assets/models/ssd_mv3.nb` 下，然后更新代码

```c++
// 代码文件 `detection_demo/ViewController.mm`
- (void)viewDidLoad {
...
NSString *path = [[NSBundle mainBundle] bundlePath];
std::string paddle_mobilenetv1_dir = std::string([path UTF8String]);
#std::string det_model_file = paddle_dir + "/ch_ppocr_mobile_v2.0_det_slim_opt.nb";
std::string det_model_file = paddle_dir + "/ssd_mv3.nb";
...
}
```
**注意：**

- 如果更新模型中的输入 Tensor、Shape、和 Dtype 发生更新:

  - 更新文字方向分类器模型，则需要更新  `ppocr_demo/cls_process.cc` 中 `ClsPredictor::Preprocss` 函数
  - 更新检测模型，则需要更新  `ppocr_demo/det_process.cc` 中 `DetPredictor::Preprocss` 函数
  - 更新识别器模型，则需要更新 `ppocr_demo/rec_process.cc` 中 `RecPredictor::Preprocss` 函数

- 如果更新模型中的输出 Tensor 和 Dtype 发生更新:

  - 更新文字方向分类器模型，则需要更新  `ppocr_demo/cls_process.cc` 中 `ClsPredictor::Postprocss` 函数
  - 更新检测模型，则需要更新  `ppocr_demo/det_process.cc` 中 `DetPredictor::Postprocss` 函数
  - 更新识别器模型，则需要更新  `ppocr_demorec_process.cc` 中 `RecPredictor::Postprocss` 函数

- 如果需要更新  `ppocr_keys_v1.txt` 标签文件，则需要将新的标签文件存放在目录 `./assets/labels/` 下

### 更新输入/输出预处理
1. 更新输入数据

- 将更新的图片存放在 `./ppocr_demo/assets/images/` 下；
- 将新图片的路径添加到 `Build Phases-> Copy Bundle Resource` 中
- 更新文件 `./ppocr_demo/ViewController.mm` 中的代码

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
  - 更新文字方向分类器模型，则需要更新 `ppocr_demo/cls_process.cc` 中 `ClsPredictor::Preprocss` 函数
  - 更新检测模型，则需要更新 `ppocr_demo/det_process.cc` 中 `DetPredictor::Preprocss` 函数
  - 更新识别器模型，则需要更新 `ppocr_demo/rec_process.cc` 中 `RecPredictor::Preprocss` 函数

3. 更新输出预处理

  - 更新文字方向分类器模型，则需要更新 `ppocr_demo/cls_process.cc` 中 `ClsPredictor::Postprocss` 函数
  - 更新检测模型，则需要更新 `ppocr_demo/det_process.cc` 中 `DetPredictor::Postprocss` 函数
  - 更新识别器模型，则需要更新 `ppocr_demo/rec_process.cc` 中 `RecPredictor::Postprocss` 函数

## OCR 文字识别 Demo 工程详解

OCR 文字识别 Demo 由三个模型一起完成 OCR 文字识别功能，对输入图片先通过 `ch_ppocr_mobile_v2.0_det_slim_opt.nb` 模型做检测处理，然后通过 `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` 模型做文字方向分类处理，最后通过 `ch_ppocr_mobile_v2.0_rec_slim_opt.nb` 模型完成文字识别处理。

1. `pipeline.cc` : OCR 文字识别 Demo 预测全流程代码
  该文件完成了三个模型串行推理的全流程控制处理，包含整个处理过程的调度处理。

  - `Pipeline::Pipeline(...)` 方法完成调用三个模型类构造函数，完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `Pipeline::Process(...)` 方法用于完成这三个模型串行推理的全流程控制处理
  
2. `cls_process.cc` 方向分类器的预测文件
  该文件完成了方向分类器的预处理、预测和后处理过程

  - `ClsPredictor::ClsPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `ClsPredictor::Preprocess()` 方法用于模型的预处理
  - `ClsPredictor::Postprocess()` 方法用于模型的后处理

3. `rec_process.cc` 识别模型 CRNN 的预测文件
  该文件完成了识别模型 CRNN 的预处理、预测和后处理过程

  - `RecPredictor::RecPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `RecPredictor::Preprocess()` 方法用于模型的预处理
  - `RecPredictor::Postprocess()` 方法用于模型的后处理

4. `det_process.cc` 检测模型 DB 的预测文件
  该文件完成了检测模型 DB 的预处理、预测和后处理过程

  - `DetPredictor::DetPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
  - `DetPredictor::Preprocess()` 方法用于模型的预处理
  - `DetPredictor::Postprocess()` 方法用于模型的后处理

5. `db_post_process` 检测模型 DB 的后处理函数，包含 clipper 库的调用
  该文件完成了检测模型 DB 的第三方库调用和其他后处理方法实现

  - `std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(...)` 方法从 Bitmap 图中获取检测框
  - `std::vector<std::vector<std::vector<int>>> FilterTagDetRes(...)` 方法根据识别结果获取目标框位置

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。
