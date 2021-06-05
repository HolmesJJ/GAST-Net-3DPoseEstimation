## 安装必要软件

### Ubuntu 18.04 安装 NVIDIA 显卡驱动
* https://zhuanlan.zhihu.com/p/59618999

### Ubuntu18.04 安装 Anaconda3
* https://blog.csdn.net/qq_15192373/article/details/81091098

### 安装ffmpeg
```
sudo apt install ffmpeg
```

### 升级安装包
```
sudo apt upgrade
```

## Anaconda配置环境
1. 创建python版本3.7.10的环境
```
conda create --name VideoPose3D python=3.7.10
```
2. 安装好之后，使用以下命令查看所有已安装的环境
```
conda info --envs
```
3. 激活环境
```
conda activate VideoPose3D
```
4. 退出环境
```
conda deactivate
```

5. 删除环境
```
conda remove -n VideoPose3D --all
```

## 安装pip依赖库

### 安装pytorch
* https://pytorch.org/
```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 安装opencv-python
```
pip install opencv-python==4.5.2.52
```

### 安装matplotlib
```
# 版本 > 3.2.2会出现兼容问题
pip install matplotlib==3.2.2
```

### 安装scipy
```
pip install scipy==1.6.3
```

### 安装torchsummary
```
pip install torch-summary==1.4.5
```

### 安装h5py
```
pip install h5py==3.2.1
```

### 安装tqdm
```
pip install tqdm==4.60.0
```

### 安装yacs
```
pip install yacs==0.1.8
```

### 安装numba
```
pip install numba==0.53.1
```

### 安装scikit-image
```
pip install scikit-image==0.18.1
```

### 安装filterpy
```
pip install filterpy==1.4.5
```

### 查看所有pip安装的信息
```
pip list
```

## 遇到的bugs

### NotImplementedError: It is not currently possible to manually set the aspect on 3D axes

* 解决方案以下二选一
```
conda install matplotlib=2.2.3
# ax.set_aspect('equal')
```

```
https://github.com/matplotlib/matplotlib/issues/15382
```