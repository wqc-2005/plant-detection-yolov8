# Plant Detection with YOLOv8

基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 的植物类别检测项目，支持自定义数据集训练、模型推理与结果可视化。

## 功能特性

- 基于 Ultralytics YOLOv8 的植物目标检测
- 支持自定义植物数据集训练
- 提供单张图片 / 视频推理脚本
- 训练过程可视化（loss、mAP 曲线等）
- 推理结果输出（标注图片 + 检测坐标）

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- ultralytics

## 安装

```bash
# 克隆仓库
git clone https://github.com/wmsdsb/plant-detection-yolov8.git
cd plant-detection-yolov8

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

## 数据集准备

将数据集组织为 YOLO 格式：

```
datasets/
└── plant/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

每张图片对应一个同名的 `.txt` 标签文件，格式为：

```
class_id  center_x  center_y  width  height
```

在 `data/plant.yaml` 中配置数据集路径和类别：

```yaml
path: datasets/plant
train: images/train
val: images/val

names:
  0: 类别1
  1: 类别2
  # 按需添加
```

## 使用方法

### 训练

```bash
python train.py --data data/plant.yaml --epochs 100 --batch 16
```

关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集配置文件路径 | `data/plant.yaml` |
| `--epochs` | 训练轮数 | 100 |
| `--batch` | 批次大小 | 16 |
| `--img` | 输入图片尺寸 | 640 |
| `--model` | 预训练模型 | `yolov8n.pt` |

### 推理

```bash
# 图片推理
python predict.py --source test_image.jpg

# 视频推理
python predict.py --source test_video.mp4

# 摄像头实时推理
python predict.py --source 0
```

## 项目结构

```
plant-detection-yolov8/
├── data/
│   └── plant.yaml          # 数据集配置
├── models/                 # 导出的模型文件
├── scripts/
│   ├── train.py            # 训练脚本
│   └── predict.py          # 推理脚本
├── utils/                  # 工具函数
├── requirements.txt        # 依赖列表
└── README.md
```

## 效果展示

> 训练和推理结果将在这里展示。

## 许可证

[MIT](LICENSE)
