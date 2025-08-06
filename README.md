### README.md

#### TrackNetV1 - 基于深度学习的网球追踪系统

本项目旨在复现 TrackNetv1 模型，一个利用深度学习技术对网球运动进行精准追踪的系统。通过分析连续的视频帧，模型能够预测网球的精确位置，并可生成带有轨迹的可视化视频。

> 本项目基于[这个项目](https://github.com/yastrebksv/TrackNet)继续改进，贡献了一个可以直接调通的docker镜像以及requirements.txt。

#### 1\. 环境配置

我们使用 Docker 来确保一个稳定、可复现的开发环境。请按照以下步骤进行配置：

1.  **准备工作**：确保您的系统已安装 Git 和 Docker。
2.  **克隆仓库**：
    ```bash
    git clone https://github.com/codelancera-offical/TrackNetV1
    cd TrackNetV1
    ```
      * `Dockerfile` 和 `docker-compose.yml` 文件应位于此目录。
3.  **构建并启动容器**：
    ```bash
    docker-compose up --build -d
    ```
      * 此命令会根据 `Dockerfile` 构建一个包含所有必要依赖（PyTorch, OpenCV, ffmpeg 等）的镜像，并启动一个名为 `tracknetv1_container` 的后台容器。
4.  **进入容器**：
    ```bash
    docker exec -it tracknetv1_container bash
    ```
5.  **安装 Python 依赖**：
    在容器内的 `/app` 目录中，运行以下命令安装 Python 依赖：
    ```bash
    pip install -r requirements.txt
    ```

#### 2\. 数据集准备

本项目的数据集需要进行预处理。

1.  **下载原始数据集**：
      * 在这里 [https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut] 下载原始数据集。
2.  **放置数据集**：
      * 将下载后的 `Dataset` 文件夹放置在您主机上的项目根目录（`.../TrackNetV1`）下。
3.  **运行处理脚本**：
    在容器终端中，运行 `gt_gen.py` 脚本生成模型所需的训练和验证数据：
    ```bash
    python gt_gen.py --path_input /app/Dataset --path_output /app/datasets/trackNet
    ```
      * 处理完成后，`/app/datasets/trackNet` 目录下将生成 `labels_train.csv` 和 `labels_val.csv` 文件。

#### 3\. 模型推理

- [预训练模型权重文件](https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing)
- 推理视频放在/videos/example目录下即可

##### 开启线性插值

此模式下，程序会修复轨迹中的断点，提供更平滑的追踪效果。

```bash
python infer_on_video.py --model_path /app/models/pretrained_model_best.pt --video_path /app/videos/example/example000.mp4 --video_out_path /app/videos/output/example000.mp4 --extrapolation
```

##### 不开启线性插值

此模式下，程序将只显示模型实际检测到的点。

```bash
python infer_on_video.py --model_path /app/models/pretrained_model_best.pt --video_path /app/videos/example/example000.mp4 --video_out_path /app/videos/output/example000_nopolation.mp4
```

#### 4\. 模型性能

以下是模型在验证集上的性能指标：

  * **混淆矩阵**：
    | | 预测有球 (Positive) | 预测无球 (Negative) |
    | :--- | :--- | :--- |
    | **真实有球 (True)** | 5314 | 159 |
    | **真实无球 (False)**| 185 | 236 |

  * **性能分数**：

      * **Precision (精确率)**：0.966
      * **Recall (召回率)**：0.971
      * **F1-Score (F1 分数)**：0.969
