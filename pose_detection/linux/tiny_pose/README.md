# 姿态检测（pose detection） C++ API Demo 使用指南

在 ARMLinux 上实现单帧/实时的单人姿态检测功能，此 Demo 有较好的的易用性和扩展性，如在 Demo 中跑自己训练好的模型、根据使用场景调整检测参数等。
  - 如果该开发板使用搭载了芯原 NPU （瑞芯微、晶晨、JQL、恩智浦）的 Soc，将有更好的加速效果。

## 如何运行姿态检测 Demo
 
### 环境准备

* 准备 ARMLiunx 开发版，将系统刷为 Ubuntu，用于 Demo 编译和运行。请注意，本 Demo 是使用板上编译，而非交叉编译，因此需要图形界面的开发板操作系统。
* 如果需要使用 芯原 NPU 的计算加速，对 NPU 驱动版本有严格要求，请务必注意事先参考 [芯原 TIM-VX 部署示例](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/verisilicon_timvx.html#id6)，将 NPU 驱动改为要求的版本。
* Paddle Lite 当前已验证的开发板为 Khadas VIM3（芯片为 Amlogic A311D）、Khadas VIM3L（芯片为 Amlogic S905D3）、荣品 RV1126、荣品RV1109，其它平台用户可自行尝试：
  - Khadas VIM3：由于 VIM3 出厂自带 Android 系统，请先刷成 Ubuntu 系统，在此提供刷机教程：[VIM3/3L Linux 文档](https://docs.khadas.com/linux/zh-cn/vim3)，其中有详细描述刷机方法。以及系统镜像：VIM3 Linux：VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625：[官方链接](http://dl.khadas.com/firmware/VIM3/Ubuntu/EMMC/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)；[百度云备用链接](https://paddlelite-demo.bj.bcebos.com/devices/verisilicon/firmware/khadas/vim3/VIM3_Ubuntu-gnome-focal_Linux-4.9_arm64_EMMC_V1.0.7-210625.img.xz)
  - 荣品 RV1126、1109：由于出场自带 buildroot 系统，如果使用 GUI 界面的 demo，请先刷成 Ubuntu 系统，在此提供刷机教程：[RV1126/1109 教程](https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/os_img/rockchip/RV1126-RV1109%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC%E6%96%87%E6%A1%A3-V3.0.pdf)，[刷机工具](https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/os_img/rockchip/RKDevTool_Release.zip)，以及镜像：[1126镜像](https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/os_img/update-pro-rv1126-ubuntu20.04-5-720-1280-v2-20220505.img)，[1109镜像](https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/os_img/update-pro-rv1109-ubuntu20.04-5.5-720-1280-v2-20220429.img)。完整的文档和各种镜像请参考[百度网盘链接](https://pan.baidu.com/s/1Id0LMC0oO2PwR2YcYUAaiQ#list/path=%2F&parentPath=%2Fsharelink2521613171-184070898837664)，密码：2345。
* 准备 usb camera，注意使用 openCV capture 图像时，请注意 usb camera 的 video序列号作为入参。
```shell
ls -l /dev/video*  #查看usb camera的video序列号
```
* 请注意，瑞芯微芯片不带有 HDMI 接口，图像显示是依赖 MIPI DSI，所以请准备好 MIPI 显示屏（我们提供的镜像是 720*1280 分辨率，网盘中有更多分辨率选择，注意：请选择 camera-gc2093x2 的镜像）。
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
2. 姿态检测 Demo 位于 `Paddle-Lite-Demo/pose_detection/linux/tiny_pose` 目录
3. 进入 `Paddle-Lite-Demo/pose_detection/linux` 目录, 终端中执行 `download_models_and_libs.sh` 脚本自动下载模型和 Paddle Lite 预测库

```shell
cd Paddle-Lite-Demo/pose_detection/linux        # 1. 终端中进入 Paddle-Lite-Demo/pose_detection/linux
sh download_models_and_libs.sh                  # 2. 执行脚本下载依赖项 （需要联网）
```

下载完成后会出现提示： `Download successful!`

4. 进入 `Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/libs/armv8` 目录，根据部署的芯片替换对应的库（默认为A311D，可跳过此步骤；RV1109、RV1126所依赖库相同，也可跳过此步骤）

```shell
cd Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/libs/armv8   # 1. 终端中进入 Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/libs/armv8
cp -f libs_S905D3/* ./                                            # 2-1. 将S905D3所需库文件复制到当前目录下
cp -f libs_A311D/* ./                                             # 2-2. 将A311D所需库文件复制到当前目录下(当前目录下默认有A311D所需库文件)
```

5. 执行用例(保证 ARMLinux 环境准备完成)

```shell
cd tiny_pose                # 1. 终端中进入。以下如果是 32bit 环境（RV1126、RV1109...），则armv8->armv7hf
sh build.sh armv8           # 2. 编译 Demo 可执行程序
sh run.sh armv8             # 3-1. 执行姿态检测（pptinypose 模型） demo，单帧图片运行，将/image中的图片作为输入，保存处理后图片并呈现检测结果
sh run_rt.sh armv8          # 3-2. 执行姿态检测（pptinypose 模型） demo，实时视频运行，会直接开启摄像头，启动图形界面并呈现检测结果
```

### Demo 结果如下:（注意，示例的 tiny-pose 仅使用 coco 数据集与部分其他数据集，在实际场景中效果一般，请使用实际业务场景重新训练）
<center class="half">
  <img decoding="async" src="https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/demo_result_view/tinpose_view_image.jpg" alt="tinpose_view_image" width="30%">
  <img decoding="async" src="https://paddlelite-demo.bj.bcebos.com/Paddle-Lite-Demo/demo_result_view/tinypose_view_camera.jpg" alt="tinypose_view_camera" width="50%">
</center><center><div>左：单帧检测；右：实时检测</div></center>

## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [芯原 TIM-VX 部署示例](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/verisilicon_timvx.html#tim-vx)，编译预测库
 * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/include` 文件夹替换 Demo 中的 `Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/include`
        * armv8
          将生成的 `build.lite.linux.armv8.gcc/inference_lite_lib.armlinux.armv8.nnadapter/cxx/libs/libpaddle_full_api_shared.so、libnnadapter.so、libtim-vx.so、libverisilicon_timvx.so` 库替换 Demo 中的 `Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/libs/armv8/` 目录下同名 so
        * armv7hf
          将生成的 `build.lite.linux.armv7hf.gcc/inference_lite_lib.armlinux.armv7hf.nnadapter/cxx/libs/libpaddle_full_api_shared.so、libnnadapter.so、libtim-vx.so、libverisilicon_timvx.so` 库替换 Demo 中的 `Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/libs/armv7hf/` 目录下同名 so

## Demo 内容介绍

先整体介绍下目标检测 Demo 的代码结构，然后再简要地介绍 Demo 每部分功能.

1. `pose_detection_demo.cc`： C++ 预测代码

```shell
# 位置：
Paddle-Lite-Demo/pose_detection/linux/tiny_pose/pose_detection_demo.cc
```

2. `models` : 模型文件夹 (执行 download_models_and_libs.sh 后会下载 PP-TinyPose 模型)

```shell
# 位置：
Paddle-Lite-Demo/pose_detection/linux/tiny_pose/models/PP_TinyPose_128x96_qat_dis_nopact
```

3. `Paddle-Lite`：内含 Paddle-Lite 头文件和 动态库，默认带有 timvx 加速库，以及第三方库 yaml-cpp 用于解析 yml 配置文件（执行 download_models_and_libs.sh 后会下载）

```shell
# 位置
# 如果要替换动态库 so，则将新的动态库 so 更新到此目录下（32bit环境更新/armv7hf）
Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/libs/armv8
Paddle-Lite-Demo/pose_detection/linux/Paddle-Lite/include
```

4. `CMakeLists.txt` : C++ 预测代码的编译脚本，用于生成可执行文件

```shell
# 位置
Paddle-Lite-Demo/pose_detection/linux/tiny_pose/CMakeLists.txt
# 如果有cmake 编译选项更新，可以在 CMakeLists.txt 进行修改即可，默认编译 armv8 可执行文件；
```

5. `build.sh` : 编译脚本

```shell
# 位置
Paddle-Lite-Demo/pose_detection/linux/tiny_pose/build.sh
```

6. `run*.sh` : 运行脚本，请注意设置 arm-aarch，armv8 或者 armv7hf。默认为armv8

```shell
# 位置
Paddle-Lite-Demo/pose_detection/linux/tiny_pose/run*.sh
```
- 请注意，运行需要4个元素：测试程序、模型、异构配置、yaml 文件。

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
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kInt8)});
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
// 如果只需要 CPU 计算，那到此结束即可，下面是设置 NPU 的代码段
valid_places.push_back(
      paddle::lite_api::Place{TARGET(kNNAdapter), PRECISION(kInt8)});

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

```

## 如何更新模型和输入/输出预处理

### 更新模型
1. 请参考 PaddleDetection 中 [PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/keypoint/tiny_pose/README.md)，基于用户自己数据集进行重训并且重新全量化
2. 将模型存放到目录 `pose_detection/assets/models/` 下；
3. 模型名字跟工程中模型名字一模一样，即均是使用 `model.pdmodel`、`model.pdiparams`；

```shell
# shell 脚本 `tiny_pose/run*.sh`
TARGET_ABI=armv8 # for 64bit, such as Amlogic A311D
#TARGET_ABI=armv7hf # for 32bit, such as Rockchip 1109/1126
if [ -n "$1" ]; then
    TARGET_ABI=$1
fi
export LD_LIBRARY_PATH=../Paddle-Lite/libs/$TARGET_ABI/
export GLOG_v=0 # Paddle-Lite 日志等级
export VSI_NN_LOG_LEVEL=0 # TIM-VX 日志等级
export VIV_VX_ENABLE_GRAPH_TRANSFORM=-pcq:1 # NPU 开启 perchannel 量化模型
export VIV_VX_SET_PER_CHANNEL_ENTROPY=100 # 同上 
build/pose_detection_demo ../../assets/models/PP_TinyPose_128x96_qat_dis_nopact ../../assets/models/PP_TinyPose_128x96_qat_dis_nopact/subgraph.txt ../../assets/models/PP_TinyPose_128x96_qat_dis_nopact/infer_cfg.yml ../../assets/images/posedet_demo.jpg ../../assets/images/posedet_demo_output.jpg  # 执行 Demo 程序，5个 arg 分别为：模型、 自定义异构配置、 yaml、 input_image、 output_image
```


### 更新输入/输出预处理

1. 更新输入预处理
预处理完全根据 yaml 文件来，如果完全按照教程重训，只需要替换 yaml 文件即可

2. 更新输出后处理
此处需要更新 `pose_detection_demo.cc` 中的 `postprocess` 方法， 可[参考论文](https://arxiv.org/abs/1910.06278)。

```c++
std::vector<RESULT> postprocess(const float *output_data, int64_t output_size,
                                const float score_threshold,
                                cv::Mat &output_image, double time) {
	bool target_detected=true;
	std::vector<RESULT> results;
	float scale_x=output_image.rows/32.f;
	float scale_y=output_image.cols/24.f;
	std::vector<cv::Point> kpts;
	std::vector<std::array<int, 2>> link_kpt = {
		{0,1},{1,3},{3,5},{5,7},{7,9},
							{5,11},{11,13},{13,15},
		{0,2},{2,4},{4,6},{6,8},{8,10},
							{6,12},{12,14},{14,16},
		{11,12}};
	
	//get posted result
	for (int64_t i = 0; i < output_size; i += 768) {
		results.push_back(get_keypoints(output_data+i,i/768));
		if(results[i/768].num_joints<0){
			target_detected=false;
			break;
		}
	}

	//shell&vision show
	if(target_detected){
		//shell show
		printf("results: %ld\n", results.size());
		for (int i = 0; i < results.size(); i++) {
		std::array<float, 2> center=find_center(results[i],scale_x,scale_y);
		kpts.push_back(cv::Point(center[1],center[0]));
		printf("[%d] - %f, %f\n", results[i].num_joints, center[0], center[1]);
		}
            //vision show
		for(auto p:kpts){
			cv::circle(output_image,p,2,cv::Scalar(0, 0, 255),-1);
		}
		for(auto idx:link_kpt){
			cv::line(output_image,kpts[idx[0]],kpts[idx[1]],cv::Scalar(255, 0, 0),1);
		}
	}
	return results;
}
```

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
 - 回退第一步的修改，不再注释，使用 NPU 加速
 - 运行 Demo，如果此时精度良好，则无需参考后面步骤，模型部署和替换非常顺利，enjoy it。
 - 如果精度不行，请参考后续步骤。

### 第三步，获取整网拓扑信息
 - 回退第一步的修改，使用
 - 修改 run*.sh ，将其中 export GLOG_v=0 改为 export GLOG_v=5
 - 运行 Demo，等摄像头启动，即可 ctrl+c 关闭 Demo
 - 收集日志，搜索关键字 "subgraph operators" 随后那一段，便是整个模型的拓扑信息，其格式如下：
    - 每行记录由『算子类型:输入张量名列表:输出张量名列表』组成（即以分号分隔算子类型、输入和输出张量名列表），以逗号分隔输入、输出张量名列表中的每个张量名；
    - 示例说明：
    ```
      op_type0:var_name0,var_name1:var_name2          表示将算子类型为 op_type0、输入张量为var_name0 和 var_name1、输出张量为 var_name2 的节点强制运行在 ARM CPU 上
    ```

### 第四步，修改异构配置文件
 - 首先看到示例 Demo 中 Paddle-Lite-Demo/pose_detection/assets/models/PP_TinyPose_128x96_qat_dis_nopact 目录下的 subgraph.txt 文件。(feed 和 fetch 分别代表整个模型的输入和输出)
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
 - 在 txt 中的都是需要异构至 cpu 计算的 layer，在示例 Demo 中，我们把 tinypose 后处理的部分异构至 arm cpu 做计算，不必担心，Paddle-Lite 的 arm kernel 性能也是非常卓越。
 - 如果新训练的模型没有额外修改 layer，则直接复制使用示例 Demo 中的 subgraph.txt 即可
 - 此时 ./run*.sh 看看精度是否符合预期，如果精度符合预期，恭喜，可以跳过本章节，enjoy it。
 - 如果精度不符合预期，则将上文『第二步，获取整网拓扑信息』中获取的拓扑信息，从 "feed" 之后第一行，直到 "sqrt" 之前，都复制进 sugraph.txt。这一步代表了将大量的 backbone 部分算子放到 arm cpu 计算。
 - 此时 ./run*.sh 看看精度是否符合预期，如果精度达标，那说明在 backbone 中确实存在引入 NPU 精度异常的层（再次重申，在 subgraph.txt 的代表强制在 arm cpu 计算）。
 - 逐行删除、成片删除、二分法，发挥开发人员的耐心，找到引入 NPU 精度异常的 layer，将其留在 subgraph.txt 中，按照经验，如果有 NPU 精度问题，可能会有 1~5 层conv layer 需要异构。
 - 剩余没有精度问题的 layer 在 subgraph.txt 中删除即可
