# 树莓派摄像头的购买、安装、配置与验证（以树莓派3B为例）

## 1、购买

可以从淘宝、京东等平台购买，直接搜索，树莓派型号 + 摄像头，例如"树莓派3B摄像头"，大概25元左右。

## 2、安装

将树莓派断电后，将摄像头如下图安装在树莓派上。

![](../doc/enable_camera_on_raspberry_pi_step0.png)

## 3、配置

执行命令

```shell
sudo raspi-config
```
进入设置页面后，之后按照下面步骤设置

1. 选择 `Interfacing Options`

![](../doc/enable_camera_on_raspberry_pi_step1.png)

2. 选择 `Camera`

![](../doc/enable_camera_on_raspberry_pi_step2.png)

3. 点击 `Yes`

![](../doc/enable_camera_on_raspberry_pi_step3.png)

4. 点击 `Ok`

![](../doc/enable_camera_on_raspberry_pi_step4.png)

5. 之后重启树莓派

## 4、验证

执行命令

```shell
sudo raspistill -o test.jpg 
```

之后使用树莓派自带的`xdg-open`打开图片

```shell
sudo xdg-open test.jpg
```

## 5、安装V4L2驱动使OpenCV能够识别摄像头

在命令行中输入

```shell
sudo nano /etc/modules  
```

在 `/etc/modules` 中添加 `bcm2835-v4l2`，如下图所示。 之后先按下`Control` + `o`，之后按下`Enter`保存；接着执行`Control` + `x`退出编辑。
