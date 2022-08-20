# 目标检测 C++ API Demo 使用指南

在 ARMLinux 上实现实时的目标检测功能，此 Demo 有较好的的易用性和扩展性，如在 Demo 中跑自己训练好的模型等。
 - 如果该开发板使用搭载了芯原 NPU （瑞芯微、晶晨、JQL、恩智浦）的 Soc，将有更好的加速效果。

## 如何运行目标检测 Demo

### 环境准备

* 准备 ARMLiunx 开发版，将系统刷为 Ubuntu，用于 Demo 编译和运行。请注意，本 Demo 是使用板上编译，而非交叉编译，因此需要图形界面的开发板操作系统。
* 如果需要使用 芯原 NPU 的计算加速，对 NPU 驱动版本有严格要求，请务必注意事先参考 [芯原 TIM-VX 部署示例](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/verisilicon_timvx.html#id6)，将 NPU 驱动改为要求的版本。
* Paddle Lite 当前已验证 Khadas VIM3 （芯片为 Amlogic A311d）开发板，其它平台用户可自行尝试；由于 VIM3 出厂自带 Android 系统，请先刷成 Ubuntu 系统，在此提供刷机教程：[VIM3/3L Linux 文档](https://docs.khadas.com/linux/zh-cn/vim3)，其中有详细描述刷机方法。以及系统镜像：VIM3 Linux：VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625：[官方链接](http://dl.khadas.com/firmware/VIM3/Ubuntu/EMMC/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)
* 准备 usb camera。
* 配置开发板的网络。如果是办公网络红区，可以将开发板和PC用以太网链接，然后PC共享网络给开发板。
* gcc g++ opencv cmake 的安装（以下所有命令均在设备上操作）

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

### 部署步骤

1. 将本 repo 上传至 VIM3 开发板，或者直接开发板上下载或者 git clone 本 repo
2. 目标检测 Demo 位于 `Paddle-Lite-Demo/object_detection/linux/picodet_detection` 目录
3. 进入 `Paddle-Lite-Demo/object_detection/linux` 目录, 终端中执行 `download_models_and_libs.sh` 脚本自动下载模型和 Paddle Lite 预测库

```shell
cd Paddle-Lite-Demo/object_detection/linux   # 1. 终端中进入 Paddle-Lite-Demo/object_detection/linux
sh download_models_and_libs.sh               # 2. 执行脚本下载依赖项 （需要联网）
```

下载完成后会出现提示： `Download successful!`
4. 执行用例(保证 ARMLinux 环境准备完成)

```shell
cd picodet_detection        # 1. 终端中进入
sh build.sh                 # 2. 编译 Demo 可执行程序
sh run.sh                   # 3. 执行物体检测（picodet 模型） demo，会直接开启摄像头，启动图形界面并呈现检测结果。
```

### Demo 结果如下:（注意，示例的 picodet 仅使用 coco 数据集，在实际场景中效果一般，请使用实际业务场景重新训练）

  <img src="https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/demo_view.jpg" alt="demo_view" style="zoom: 10%;" />

## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [芯原 TIM-VX 部署示例](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/verisilicon_timvx.html#tim-vx)，编译预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.xxx/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/object_detection/linux/Paddle-Lit/include`
        * armv8
          将生成的 `build.lite.armLinux.armv8.gcc/inference_lite_lib.armLinux.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 Demo 中的 `Paddle-Lite-Demo/object_detection/linux/Paddle-Lit/libs/armv8/libpaddle_light_api_shared.so`

## Demo 内容介绍

先整体介绍下目标检测 Demo 的代码结构，然后再简要地介绍 Demo 每部分功能.

1. `object_detection_demo.cc`： C++ 预测代码

```shell
# 位置：
Paddle-Lite-Demo/object_detection/linux/picodet_detection/object_detection_demo.cc
```

2. `models` : 模型文件夹 (执行 download_models_and_libs.sh 后会下载 picodet Paddle 模型), label 使用 Paddle-Lite-Demo/object_detection/assets/labels 目录下 coco_label_list.txt

```shell
# 位置：
Paddle-Lite-Demo/object_detection/linux/picodet_detection/models/picodetv2_relu6_coco_no_fuse
Paddle-Lite-Demo/object_detection/assets/labels/coco_label_list.txt
```

3. `Paddle-Lite`：内含 Paddle-Lite 头文件和 动态库，默认带有 timvx 加速库，以及第三方库 yaml-cpp 用于解析 yml 配置文件（执行 download_models_and_libs.sh 后会下载）

```shell
# 位置
# 如果要替换动态库 so，则将新的动态库 so 更新到此目录下
Paddle-Lite-Demo/object_detection/linux/Paddle-Lite/libs/armv8
Paddle-Lite-Demo/object_detection/linux/Paddle-Lite/include
```

4. `CMakeLists.txt` : C++ 预测代码的编译脚本，用于生成可执行文件

```shell
# 位置
Paddle-Lite-Demo/object_detection/linux/picodet_detection/CMakeLists.txt
# 如果有cmake 编译选项更新，可以在 CMakeLists.txt 进行修改即可，默认编译 armv8 可执行文件；
```

5. `build.sh` : 编译脚本

```shell
# 位置
Paddle-Lite-Demo/object_detection/linux/picodet_detection/build.sh
```

6. `run.sh` : 运行脚本

```shell
# 位置
Paddle-Lite-Demo/object_detection/linux/picodet_detection/run.sh
```
- 请注意，运行需要5个元素：测试程序、模型、label 文件、异构配置、yaml 文件。

## 代码讲解 （使用 Paddle Lite `C++ API` 执行预测）

ARMLinux 示例基于 C++ API 开发，调用 Paddle Lite `C++s API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite C++ API ](https://paddle-lite.readthedocs.io/zh/latest/api_reference/c++_api_doc.html)。

```c++
#include <iostream>
// 引入 C++ API
#include "include/paddle_api.h"
#include "include/paddle_use_ops.h"
#include "include/paddle_use_kernels.h"

// 使用在线编译模型的方式（等价于使用 opt 工具）

// 1. 设置 CxxConfig
paddle::lite_api::CxxConfig cxx_config;
std::vector<paddle::lite_api::Place> valid_places;
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
// 如果只需要 cpu 计算，那到此结束即可，下面是设置 NPU 的代码段
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
cxx_config.set_valid_places(valid_places);
std::string device = "verisilicon_timvx";
cxx_config.set_nnadapter_device_names({device});
// 设置定制化的异构策略 （如需要）
cxx_config.set_nnadapter_subgraph_partition_config_buffer(
            nnadapter_subgraph_partition_config_string);

// 2. 生成 nb 模型 （等价于 opt 工具的产出）
std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor = nullptr;
predictor = paddle::lite_api::CreatePaddlePredictor(cxx_config);
predictor->SaveOptimizedModel(
        model_path, paddle::lite_api::LiteModelType::kNaiveBuffer);

// 3. 设置 MobileConfig
MobileConfig config;
config.set_model_from_file(modelPath); // 设置 NaiveBuffer 格式模型路径
config.set_power_mode(LITE_POWER_NO_BIND); // 设置 CPU 运行模式
config.set_threads(4); // 设置工作线程数

// 4. 创建 PaddlePredictor
predictor = CreatePaddlePredictor<MobileConfig>(config);

// 5. 设置输入数据，注意，如果是带后处理的 picodet ，则是有两个输入
std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 416, 416});
auto* data = input_tensor->mutable_data<float>();
// 如果输入是图片，则可在第三步时将预处理后的图像数据赋值给输入 Tensor
// scale_factor tensor
auto scale_factor_tensor = predictor->GetInput(1);
scale_factor_tensor->Resize({1, 2});
auto scale_factor_data = scale_factor_tensor->mutable_data<float>();
scale_factor_data[0] = 1.0f;
scale_factor_data[1] = 1.0f;

// 6. 执行预测
predictor->run();

// 7. 获取输出数据
std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));

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
1. 将模型存放到目录 `object_detection_demo/models/` 下；
2. 模型名字跟工程中模型名字一模一样，即均是使用 `model`、`params`；

```shell
# shell 脚本 `object_detection_demo/run.sh`
export LD_LIBRARY_PATH=../Paddle-Lite/libs/armv8/ # 指定库位置
export GLOG_v=0 # Paddle-Lite 日志等级
export VSI_NN_LOG_LEVEL=0 # TIM-VX 日志等级
export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1 # NPU 开启 perchannel 量化模型
export VIV_VX_SET_PER_CHANNEL_ENTROPY=100 # 同上 
build/object_detection_demo models/picodetv2_relu6_coco_no_fuse ../../assets/labels/coco_label_list.txt models/picodetv2_relu6_coco_no_fuse/subgraph.txt models/picodetv2_relu6_coco_no_fuse/picodet.yml  # 执行 Demo 程序，4个 arg 分别为：模型、 label 文件、 自定义异构配置、 yaml
```

- 如果需要更新 `label_list` 或者 `yaml` 文件，则修改 `object_detection_demo/run.sh` 中执行命令的第二个和第四个 arg 指定为新的 label 文件和 yaml 配置文件；

```shell
# 代码文件 `object_detection_demo/rush.sh`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PADDLE_LITE_DIR}/libs/${TARGET_ARCH_ABI} 
build/object_detection_demo {模型} {label} {自定义异构配置文件} {yaml}
```

### 更新输入/输出预处理

1. 更新输入预处理
预处理完全根据 yaml 文件来，如果完全按照 PaddleDetection 中 picodet 重训，只需要替换 yaml 文件即可

2. 更新输出预处理
此处需要更新 `object_detection_demo/object_detection_demo.cc` 中的 `postprocess` 方法

<p align="center">
<img src="https://paddlelite-data.bj.bcebos.com/doc_images/ARMLinux_demo/output_change.png"/>
</p>

## 更新模型后，自定义 NPU-CPU 异构配置（如需使用 NPU 加速）
由于使用芯原 NPU 在 8bit 量化的情况下有最优的性能，因此部署时，我们往往会考虑量化
 - 由于量化可能会引入一定程度的精度问题，所以我们可以通过自定义的异构定制，来将部分有精度问题的 layer 异构至cpu，从而达到最优的精度

### 第一步，确定模型量化后在 arm cpu 上的精度
如果在 arm cpu 上，精度都无法满足，那量化本身就是失败的，此时可以考虑修改训练集或者预处理。
 - 修改 Demo 程序，仅用 arm cpu 计算
```c++
paddle::lite_api::CxxConfig cxx_config;
std::vector<paddle::lite_api::Place> valid_places;
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
// 仅用 arm cpu 计算， 注释如下代码即可
/*
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
*/
```
如果 arm cpu 计算结果精度达标，则继续

### 第二步，获取整网拓扑信息
 - 回退第一步的修改，使用
 - 修改 run.sh ，将其中 export GLOG_v=0 改为 export GLOG_v=5
 - 运行 Demo，等摄像头启动，即可 ctrl+c 关闭 Demo
 - 收集日志，搜索关键字 "subgraph operators" 随后那一段，便是整个模型的拓扑信息，其格式如下：
    - 每行记录由『算子类型:输入张量名列表:输出张量名列表』组成（即以分号分隔算子类型、输入和输出张量名列表），以逗号分隔输入、输出张量名列表中的每个张量名；
    - 示例说明：
    ```
      op_type0:var_name0,var_name1:var_name2          表示将算子类型为 op_type0、输入张量为var_name0 和 var_name1、输出张量为 var_name2 的节点强制运行在 ARM CPU 上
    ```

### 第三步，修改异构配置文件
 - 首先看到示例 Demo 中 Paddle-Lite-Demo/object_detection/linux/picodet_detection/models/picodetv2_relu6_coco_no_fuse 目录下的 subgraph.txt 文件。(feed 和 fetch 分别代表整个模型的输入和输入)
  ```
  feed:feed:scale_factor
  feed:feed:image

  sqrt:tmp_3:sqrt_0.tmp_0
  reshape2:sqrt_0.tmp_0:reshape2_0.tmp_0,reshape2_0.tmp_1

  matmul_v2:softmax_0.tmp_0,auto_113_:linear_0.tmp_0
  reshape2:linear_0.tmp_0:reshape2_2.tmp_0,reshape2_2.tmp_1

  sqrt:tmp_6:sqrt_1.tmp_0
  reshape2:sqrt_1.tmp_0:reshape2_3.tmp_0,reshape2_3.tmp_1

  matmul_v2:softmax_1.tmp_0,auto_113_:linear_1.tmp_0
  reshape2:linear_1.tmp_0:reshape2_5.tmp_0,reshape2_5.tmp_1

  sqrt:tmp_9:sqrt_2.tmp_0
  reshape2:sqrt_2.tmp_0:reshape2_6.tmp_0,reshape2_6.tmp_1

  matmul_v2:softmax_2.tmp_0,auto_113_:linear_2.tmp_0
  ...
  ```
 - 在 txt 中的都是需要异构至 cpu 计算的 layer，在示例 Demo 中，我们把 picodet 后处理的部分异构至 arm cpu 做计算，不必担心，Paddle-Lite 的 arm kernel 性能也是非常卓越。
 - 如果新训练的模型没有额外修改 layer，则直接复制使用示例 Demo 中的 subgraph.txt 即可
 - 此时 ./run.sh 看看精度是否符合预期，如果精度符合预期，恭喜，可以跳过本章节，enjoy it。
 - 如果精度不符合预期，则将上文『第二步，获取整网拓扑信息』中获取的拓扑信息，从 "feed" 之后第一行，直到 "sqrt" 之前，都复制进 sugraph.txt。这一步代表了将大量的 backbone 部分算子放到 arm cpu 计算。
 - 此时 ./run.sh 看看精度是否符合预期，如果精度达标，那说明在 backbone 中确实存在引入 NPU 精度异常的层（再次重申，在 subgraph.txt 的代表强制在 arm cpu 计算）。
 - 逐行删除、成片删除、二分法，发挥开发人员的耐心，找到引入 NPU 精度异常的 layer，将其留在 subgraph.txt 中，按照经验，如果有 NPU 精度问题，可能会有 1~5 层conv layer 需要异构。
 - 剩余没有精度问题的 layer 在 subgraph.txt 中删除即可
