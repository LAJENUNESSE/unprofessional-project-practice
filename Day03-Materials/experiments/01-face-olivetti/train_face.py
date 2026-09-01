# -*- coding: utf-8 -*-
"""
实验1：基于CNN的人脸识别（Olivetti Faces）
按《1.人脸检测与识别_指导书.pdf》实现：
  步骤1 导入相关包
  步骤2 获取数据集 fetch_olivetti_faces（400张 64x64 灰度人脸，40人各10张）
  步骤3 数据可视化（抽样展示）
  步骤4 数据处理（reshape 为 4D 张量 + 70/30 划分训练/测试）
  步骤5 构建 CNN（Conv2D 128 3x3 -> Conv2D 64 3x3 -> Flatten -> Dense 40 softmax）
  步骤6 训练（epochs=10）
  步骤7 测试（predict 并与真实标签比对）
运行：.venv/Scripts/python.exe experiments/01-face-olivetti/train_face.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/
import path_config  # noqa: F401  统一处理数据/输出目录

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "01-face-olivetti"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 步骤1~2：获取数据集 ----------
# 数据缓存到项目内 scikit_learn_data（若已有 olivetti.pkz 则离线加载，不联网）
data_home = path_config.SKLEARN_DATA_DIR
faces = datasets.fetch_olivetti_faces(data_home=str(data_home))
print("faces.images.shape =", faces.images.shape)  # 预期 (400, 64, 64)

# ---------- 步骤3：数据可视化（每人类1张，共40张缩略图）----------
fig, axes = plt.subplots(4, 10, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i * 10], cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(faces.target[i * 10], fontsize=8)
fig.suptitle("Olivetti faces: 40 subjects (one sample each)")
fig.tight_layout()
fig.savefig(OUT_DIR / "step3_faces_grid.png", dpi=110)
plt.close(fig)
print(f"可视化已保存: {OUT_DIR / 'step3_faces_grid.png'}")

# ---------- 步骤4：数据处理 ----------
X = faces.images  # 人脸数据，已归一化到 [0,1]
y = faces.target  # 人脸标签 0~39
X = X.reshape(400, 64, 64, 1)  # [N, 长, 宽] -> [N, 长, 宽, 通道]，灰度图通道数为1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("train:", X_train.shape, "test:", X_test.shape)

# ---------- 步骤5：构建CNN人脸识别模型 ----------
model = keras.Sequential()
# 第一层卷积，128 个 3x3 卷积核，relu 激活
model.add(keras.layers.Conv2D(128, kernel_size=3, activation="relu", input_shape=(64, 64, 1)))
# 第二层卷积，64 个 3x3 卷积核
model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu"))
# Flatten 把多维数组压平成一维，方便后面 Dense 使用
model.add(keras.layers.Flatten())
# 全连接输出层：40 类人脸，softmax 输出概率
model.add(keras.layers.Dense(40, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------- 步骤6：训练 ----------
history = model.fit(X_train, y_train, epochs=10)

# ---------- 步骤7：测试 ----------
y_prob = model.predict(X_test, verbose=0)
y_predict = np.argmax(y_prob, axis=1)
test_acc = float(np.mean(y_predict == y_test))
print("首个测试样本: 真实标签 =", y_test[0], ", 预测标签 =", y_predict[0])
print(f"测试集准确率 = {test_acc:.4f}  ({int((y_predict == y_test).sum())}/{len(y_test)})")

# 训练曲线
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history.history["accuracy"], label="train acc")
ax.plot(history.history["loss"], label="train loss")
ax.set_xlabel("epoch"); ax.legend(); ax.set_title("Experiment 1: training curves")
fig.tight_layout()
fig.savefig(OUT_DIR / "step6_training_curves.png", dpi=110)
plt.close(fig)

model.save(OUT_DIR / "face_cnn_model.keras")
with open(OUT_DIR / "result.txt", "w", encoding="utf-8") as f:
    f.write(f"实验1 人脸识别（Olivetti）\n训练样本: {X_train.shape[0]}  测试样本: {X_test.shape[0]}\n")
    f.write(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n")
print(f"结果与模型已保存到 {OUT_DIR}")
