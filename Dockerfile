# 基础 PyTorch 官方镜像
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# 安装缺失的库
# 我们之前遇到了 libGL 和 libgthread 的问题，所以在这里安装它们
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 卷映射请自己在docker-compose里面设置