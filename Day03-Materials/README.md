# Day03 CNN 实验合集

按学习通《神经网络介绍》《卷积神经网络》课件配套的 7 份案例指导书完成，环境为
`Day03-Materials/.venv`（Python 3.12.10 + TensorFlow 2.21.0 CPU + Keras 3.15）。

## 快速开始

```bash
cd Day03-Materials
.venv/Scripts/python.exe experiments/01-face-olivetti/train_face.py
```

每个实验独立目录，结果（图、模型、result.txt）输出到 `Day03-Materials/outputs/<实验名>/`。

## 实验列表与运行方式

| 实验 | 指导书 | 数据来源 | 运行命令 |
|---|---|---|---|
| 1 人脸识别（Olivetti） | 1.人脸检测与识别_指导书.pdf | 自动（已内置离线缓存） | `experiments/01-face-olivetti/train_face.py` |
| 2 表情识别（fer2013） | 2.基于卷积神经网络的表情识别案例.pdf | 课程数据 / `--synthetic` | `experiments/02-fer2013-expression/train_fer2013.py [--synthetic]` |
| 3 鸟类识别（AlexNet） | 3.卷积神经网络AlexNet实现鸟类识别.pdf | 课程数据 / `--synthetic` | `experiments/03-alexnet-birds/train_alexnet_birds.py [--synthetic]` |
| 4 MNIST 手写数字 | 4.mnist手写数字识别.pdf | 自动（本地已缓存） | `experiments/04-mnist/train_mnist.py` |
| 5 CIFAR-10 彩色分类 | 5.卷积神经网络实现彩色图像分类.pdf | 自动（本地已缓存） | `experiments/05-cifar10/train_cifar10.py` |
| 6 车牌识别 | 6.车牌识别.pdf | 课程数据 / `--synthetic` | `experiments/06-plate/train_plate.py [--synthetic]` |
| 7 乳腺癌识别 | 7.乳腺癌识别.pdf | 课程数据 / `--synthetic` | `experiments/07-breast-cancer/train_bca.py [--synthetic]` |

## 数据放置说明（真实课程数据）

课程私有数据集（指导书配套）目前未随 PDF 下发。拿到后按下表放入
`Day03-Materials/datasets/`，然后**不带 `--synthetic`** 重跑即得真实结果：

| 实验 | 放置路径 | 结构 |
|---|---|---|
| 2 | `datasets/fer2013/` | `train.csv`、`test.csv`、`val.csv`（emotion, pixels 两列） |
| 3 | `datasets/bird_photos/` | 4 个类别子目录（Bananaquit 等），共 565 张 224x224 彩图 |
| 6 | `datasets/licence_plate/` | 13675 张 jpg，文件名 `序号_车牌号.jpg`（如 `000000000_川W9BR26.jpg`） |
| 7 | `datasets/bca_data/` | 子目录 `0/`（正常）、`1/`（乳腺癌），50x50 切片图 |

## 离线数据缓存（已就绪，可删但重跑会重新下载）

- `scikit_learn_data/olivetti_py3.pkz` —— 由剑桥 AT&T 人脸库 zip 转换
  （构建脚本：`experiments/01-face-olivetti/build_olivetti_cache.py`）
- `keras_data/datasets/` —— MNIST npz、CIFAR-10 batches（CIFAR 由 hf-mirror parquet
  转换，脚本：`experiments/05-cifar10/build_cifar_cache.py`）

## 结果汇总

见各 `outputs/<实验>/result.txt` 与 `SUMMARY.md`。

## 文档索引

- `SUMMARY.md` —— 全部实验结果与和指导书的差异说明
- `分工计划表.md` —— 四人小组分工与协作约定（对齐实际开发流程）
- `experiments/` —— 7 个实验脚本 + 2 个数据缓存构建脚本
- `outputs/` —— 各实验指标（result.txt）、图表与小体积模型；大体积 .keras 模型与数据集缓存不入库（见 .gitignore）
