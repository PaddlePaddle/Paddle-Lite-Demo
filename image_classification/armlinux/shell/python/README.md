# 图像分类 Python API Demo 使用指南

在 ArmLinux 上实现实图像分类功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等.

## 如何运行图像分类 Demo

### 环境准备

1. 安装 Opencv 包
使用 pip 工具安装 opencv 包

```shell
python -m pip install opencv-python
```

### 部署步骤

1. 图像分类 Demo 位于 `Paddle-Lite-Demo/image_classification/linux/shell/python` 目录
3. cd `Paddle-Lite-Demo/image_classification/linux/shell/python` 目录, 运行 `python images_classification.py` 脚本，进行图像分类预测

```shell
cd Paddle-Lite-Demo/image_classification/linux/shell/python
python images_classification.py
```

   3.运行结果如下：在树莓派4B/32G上CPU的占用率始终稳定在30%左右。

![image-20220718145220897](https://s2.loli.net/2022/07/18/ldBfoUTOJM5aFZK.png)



## Demo 内容介绍

整个demo 由 image_classifiction.py 一个 python 脚本组成，开发者可根据自己的开发要求来选择单张图片预测功能或多张图片的预测功能，下面是根据实际情况需要修改的参数：
![image-20220718142407938](https://s2.loli.net/2022/07/18/l1DpfPXYiyGe2Kj.png)

`model_dir`为模型需要更改的路径，默认路径为`Paddle-Lite-Demo/image_classification/assets/models/PPLCNet/model.nb`

`image_path`为图片需要更改的路径，默认为对多张图片进行预测，将多张图片的路径存放在`test_list.txt`下

`labels_path`为标签集需要更改的路径，将其替换为自己的label.txt即可

`input_shape`为数据的输入格式要求，默认无需更改，224，224为图片输入的分辨率，若更改则需要修改对应的源代码

`thread_num`为训练时设置的线程数

`warmup_num`为模型预热的次数，需要根据模型的实际效果来设置，本模型预热前与预热后并无太大差别，因此可以设置为0

`repeat_num`用来计算单张图片在一百次预测中的平均预测速度，及最小预测速度

`topk`为打印单张图片预测时的前k名，默认为前三名，以观察模型的实际效果



在将图片路径image_path修改为xxx.png/jpg后，可观察到在100次预测中单张图片的平均预测速度如下

![image-20220718155834569](https://s2.loli.net/2022/07/18/oCc4yjdrxQUX7Kl.png)



## 更新预测库

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
 * 参考 [Paddle Lite 源码编译文档](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html)准环境
 * 参考[源码编译 (ARMLinux)](https://paddle-lite.readthedocs.io/zh/develop/source_compile/linux_x86_compile_arm_linux.html)编译和安装 Paddle Lite 的 python 包。


## 代码讲解 （使用 Paddle Lite `Python API` 执行预测）

该示例基于 Python API 开发，调用 Paddle Lite `Python API` 包括以下五步。更详细的 `API` 描述参考：[Paddle Lite Python API ](https://paddle-lite.readthedocs.io/zh/develop/api_reference/python_api_doc.html)。

```python
# 1 引入必要的库
from paddlelite.lite import *
import numpy as np
import cv2

# 2 指定模型文件，创建 predictor
config = MobileConfig()
config.set_model_from_file(model_dir)
predictor = create_paddle_predictor(config)

# 3 设置模型输入 (下面以全一输入为例)
# 如果模型有多个输入，每一个模型输入都需要准确设置 shape 和 data。
input_tensor = predictor.get_input(0)
input_tensor.from_numpy(np.ones((1, 3, 224, 224)).astype("float32"))

# 4 执行预测
predictor.run()

# 5 获得预测结果并将预测结果转化为 numpy 数组
output_tensor = predictor.get_output(0)
output_data = output_tensor.numpy()
print(output_data)
```

## 性能优化方法
如果你觉得当前性能不符合需求，想进一步提升模型性能，可参考[首页中性能优化文档](/README.md)完成性能优化。
