# -*- coding: utf-8 -*-
"""
各实验共享的路径配置：
- 数据统一放在 Day03-Materials/datasets/<子目录>（git 忽略，体积大）
- sklearn 数据缓存放在 Day03-Materials/scikit_learn_data
- Keras 数据缓存（MNIST/CIFAR）通过环境变量指向同一位置
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Day03-Materials/

DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)
SKLEARN_DATA_DIR = ROOT / "scikit_learn_data"
SKLEARN_DATA_DIR.mkdir(exist_ok=True)

# 让 keras 的 ~\.keras 缓存改到项目内，离线复用已下好的数据
os.environ.setdefault("KERAS_HOME", str(ROOT / "keras_data"))
(ROOT / "keras_data" / "datasets").mkdir(parents=True, exist_ok=True)

# MNIST/CIFAR 手动预下载的 npz/tar 放 datasets 后，直接拷/链接到 keras 缓存目录
KERAS_DATASETS = Path(os.environ["KERAS_HOME"]) / "datasets"

# 将本文件所在目录加入 sys.path 的调用方无需处理；仅保证中文打印不报错
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
