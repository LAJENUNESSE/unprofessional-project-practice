# -*- coding: utf-8 -*-
"""
实验4：MNIST 手写数字识别
按《4.mnist手写数字识别.pdf》实现：
  一、前期准备：导入数据（datasets.mnist.load_data）-> 归一化 -> 可视化 -> reshape 成 4D
  二、构建 CNN：Conv2D(32,3x3) -> MaxPool 2x2 -> Conv2D(64,3x3) -> MaxPool 2x2
              -> Flatten -> Dense(64) -> Dense(10)
  三、编译：adam + SparseCategoricalCrossentropy(from_logits=True)
  四、训练：epochs=10，validation_data=测试集
  五、预测：对测试集预测并展示第2张（索引1）的预测结果
运行：.venv/Scripts/python.exe experiments/04-mnist/train_mnist.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/
import path_config  # noqa: F401

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

OUT_DIR = path_config.ROOT / "outputs" / "04-mnist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 一、导入数据 ----------
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print("原始形状:", train_images.shape, test_images.shape, train_labels.shape, test_labels.shape)

# 归一化：像素值标准化至 0~1
train_images, test_images = train_images / 255.0, test_images / 255.0

# ---------- 可视化前 20 张 ----------
plt.figure(figsize=(10, 5))
for i in range(20):
    plt.subplot(4, 10, i + 1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i], fontsize=8)
plt.suptitle("MNIST samples")
plt.tight_layout()
plt.savefig(OUT_DIR / "samples.png", dpi=110)
plt.close()
print(f"样本图已保存: {OUT_DIR / 'samples.png'}")

# ---------- 调整图片格式为 4D ----------
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# ---------- 二、构建 CNN 网络 ----------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),  # 卷积层1，卷积核3*3
    layers.MaxPooling2D((2, 2)),                                            # 池化层1，2*2采样
    layers.Conv2D(64, (3, 3), activation="relu"),                           # 卷积层2，卷积核3*3
    layers.MaxPooling2D((2, 2)),                                            # 池化层2，2*2采样
    layers.Flatten(),        # 连接卷积层与全连接层
    layers.Dense(64, activation="relu"),  # 全连接层，特征进一步提取
    layers.Dense(10),                     # 输出层，输出预期结果
])
model.summary()

# ---------- 三、编译模型 ----------
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# ---------- 四、模型训练 ----------
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# ---------- 五、预测 ----------
pre = model.predict(test_images, verbose=0)
idx = 1  # 指导书展示的是测试集索引1的图片
plt.imshow(test_images[idx].reshape(28, 28), cmap=plt.cm.binary)
plt.title(f"predict: {np.argmax(pre[idx])}  true: {test_labels[idx]}")
plt.savefig(OUT_DIR / "predict_sample.png", dpi=110)
plt.close()
print(f"索引{idx} 预测结果: {np.argmax(pre[idx])} (真实值 {test_labels[idx]})")

# ---------- 评估 ----------
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"测试集 loss={test_loss:.4f}  accuracy={test_acc:.4f}")

# ---------- 训练曲线 ----------
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(loc="lower right")
plt.title("Experiment 4: MNIST accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_curve.png", dpi=110)
plt.close()

model.save(OUT_DIR / "mnist_cnn_model.keras")
with open(OUT_DIR / "result.txt", "w", encoding="utf-8") as f:
    f.write("实验4 MNIST 手写数字识别\n")
    f.write(f"训练: 60000 张  测试: 10000 张  epochs=10\n")
    f.write(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n")
print(f"结果与模型已保存到 {OUT_DIR}")
