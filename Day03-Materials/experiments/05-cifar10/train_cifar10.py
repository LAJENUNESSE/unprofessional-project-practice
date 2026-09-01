# -*- coding: utf-8 -*-
"""
实验5：CIFAR-10 彩色图像分类
按《5.卷积神经网络实现彩色图像分类.pdf》实现：
  一、前期准备：datasets.cifar10.load_data -> 归一化 -> 可视化（10 类）
  二、构建 CNN：Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(64)
              -> Flatten -> Dense(64) -> Dense(10)
  三、编译：adam + SparseCategoricalCrossentropy(from_logits=True)
  四、训练：epochs=10，validation_data=测试集
  五、预测与评估：单张预测 + accuracy 曲线 + evaluate
运行：.venv/Scripts/python.exe experiments/05-cifar10/train_cifar10.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/
import path_config  # noqa: F401
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

OUT_DIR = path_config.ROOT / "outputs" / "05-cifar10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import pickle


def load_cifar10_data():
    """优先从本地已转换缓存加载（build_cifar_cache.py 生成，格式与 load_data 等价），
    避免本机网络直连官方源下载 170MB。缓存缺失时回退 datasets.cifar10.load_data()。"""
    cache_dir = (path_config.KERAS_DATASETS / "cifar-10-batches-py-target"
                 / "cifar-10-batches-py")
    if (cache_dir / "data_batch_1").exists():
        x_train, y_train = [], []
        for i in range(1, 6):
            with open(cache_dir / f"data_batch_{i}", "rb") as f:
                d = pickle.load(f, encoding="bytes")
            x_train.append(d[b"data"])
            y_train.extend(d[b"labels"])
        with open(cache_dir / "test_batch", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x_test, y_test = d[b"data"], list(d[b"labels"])
        x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return (x_train, np.array(y_train).reshape(-1, 1)), (x_test, np.array(y_test).reshape(-1, 1))
    return datasets.cifar10.load_data()


# ---------- 一、导入数据 ----------
(train_images, train_labels), (test_images, test_labels) = load_cifar10_data()
print("原始形状:", train_images.shape, test_images.shape, train_labels.shape, test_labels.shape)

# 归一化
train_images, test_images = train_images / 255.0, test_images / 255.0

# ---------- 可视化 ----------
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
plt.figure(figsize=(10, 5))
for i in range(20):
    plt.subplot(4, 10, i + 1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]], fontsize=7)
plt.suptitle("CIFAR-10 samples")
plt.tight_layout()
plt.savefig(OUT_DIR / "samples.png", dpi=110)
plt.close()
print(f"样本图已保存: {OUT_DIR / 'samples.png'}")

# ---------- 二、构建 CNN 网络 ----------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),  # 卷积层1
    layers.MaxPooling2D((2, 2)),                                            # 池化层1
    layers.Conv2D(64, (3, 3), activation="relu"),                           # 卷积层2
    layers.MaxPooling2D((2, 2)),                                            # 池化层2
    layers.Conv2D(64, (3, 3), activation="relu"),                           # 卷积层3
    layers.Flatten(),                     # 连接卷积层与全连接层
    layers.Dense(64, activation="relu"),  # 全连接层
    layers.Dense(10),                     # 输出层
])
model.summary()

# ---------- 三、编译 ----------
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# ---------- 四、模型训练 ----------
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# ---------- 五、预测 ----------
pre = model.predict(test_images, verbose=0)
idx = 1  # 指导书展示测试集索引1 -> ship
plt.imshow(test_images[idx])
plt.title(f"predict: {class_names[np.argmax(pre[idx])]}  true: {class_names[test_labels[idx][0]]}")
plt.savefig(OUT_DIR / "predict_sample.png", dpi=110)
plt.close()
print(f"索引{idx} 预测: {class_names[np.argmax(pre[idx])]} (真实: {class_names[test_labels[idx][0]]})")

# ---------- 评估 ----------
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"测试集 loss={test_loss:.4f}  accuracy={test_acc:.4f}")

# ---------- 模型评估曲线 ----------
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim([0.3, 1]); plt.legend(loc="lower right")
plt.title("Experiment 5: CIFAR-10 accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_curve.png", dpi=110)
plt.close()

model.save(OUT_DIR / "cifar10_cnn_model.keras")
with open(OUT_DIR / "result.txt", "w", encoding="utf-8") as f:
    f.write("实验5 CIFAR-10 彩色图像分类\n")
    f.write("训练: 50000 张 32x32 彩色图  测试: 10000 张  epochs=10\n")
    f.write(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n")
print(f"结果与模型已保存到 {OUT_DIR}")
