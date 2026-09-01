# -*- coding: utf-8 -*-
"""
实验7：乳腺癌识别（IDC 载玻片图像二分类）
按《7.乳腺癌识别.pdf》实现：
  一、前期准备：data_dir=dataset/bca_data（子目录 0=正常,1=乳腺癌），统计图片数
  二、数据预处理：image_dataset_from_directory 8/2 划分（50x50, batch=16）
                -> 预处理函数 image/255.0 -> cache/shuffle/prefetch
                -> 可视化（class_names=["乳腺癌细胞","正常细胞"]）
  三、构建模型：Conv2D(16,3x3,same)x2 -> Pool -> Dropout(0.5)
              -> Conv2D(16)+Pool -> Conv2D(16)+Pool -> Flatten -> Dense(2, softmax)
  四、编译 adam + sparse_categorical_crossentropy；回调 EarlyStopping/ReduceLROnPlateau
  五、训练 -> 六、accuracy/loss 曲线 -> 混淆矩阵 + classification_report
用法：
  真实数据：把 bca_data 放到 datasets/bca_data/（子目录 0/、1/，50x50 PNG）
      .venv/Scripts/python.exe experiments/07-breast-cancer/train_bca.py
  合成验证：--synthetic 生成两类可区分的合成载玻片纹理图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/
import path_config  # noqa: F401

import numpy as np
np.random.seed(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
import tensorflow as tf
tf.random.set_seed(1)

ROOT = path_config.ROOT
DATA_DIR = ROOT / "datasets" / "bca_data"
OUT_DIR = ROOT / "outputs" / "07-breast-cancer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC = "--synthetic" in sys.argv
if SYNTHETIC and not DATA_DIR.exists():
    """生成合成载玻片图：0=正常细胞（浅粉圆形规则细胞），1=乳腺癌细胞（深紫大核异形）"""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(1)
    for cls, n in [("0", 600), ("1", 600)]:
        d = DATA_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            if cls == "0":
                base = np.full((50, 50, 3), (235, 205, 210), dtype=float)
                cell_color, n_cell, r_max = (215, 160, 170), 10, 5
            else:
                base = np.full((50, 50, 3), (200, 170, 200), dtype=float)
                cell_color, n_cell, r_max = (90, 40, 100), 18, 9
            img_arr = base + rng.normal(0, 10, (50, 50, 3))
            img = Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))
            dr = ImageDraw.Draw(img)
            for _ in range(n_cell):
                x, y = rng.integers(5, 45, 2)
                r = rng.integers(2, r_max)
                dr.ellipse([x - r, y - r, x + r, y + r], fill=cell_color)
            img.save(d / f"{cls}_{i:04d}.png")
    print("已生成合成乳腺癌数据 ->", DATA_DIR)

data_dir = Path(DATA_DIR)
image_count = len(list(data_dir.glob("*/*")))
print("图片总数为：", image_count)

# ---------- 二、数据预处理 ----------
batch_size = 16
img_height = 50
img_width = 50

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=12,
    image_size=(img_height, img_width), batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=12,
    image_size=(img_height, img_width), batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape, labels_batch.shape)
    break


def train_preprocessing(image, label):
    return image / 255.0, label


train_ds = train_ds.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE) \
    .cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE) \
    .cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 可视化
plt.figure(figsize=(10, 5))
cn = ["正常细胞", "乳腺癌细胞"]
for images, labels in train_ds.take(1):
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.xlabel(cn[int(labels[i])], fontsize=9)
        plt.xticks([]); plt.yticks([])
plt.suptitle("Breast cancer histopathology (synthetic)" if SYNTHETIC else "Breast cancer histopathology")
plt.tight_layout()
plt.savefig(OUT_DIR / "samples.png", dpi=110)
plt.close()

# ---------- 三、构建模型 ----------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same",
                           activation="relu", input_shape=[img_width, img_height, 3]),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation="softmax"),
])
model.summary()

# ---------- 四、编译与训练 ----------
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

NO_EPOCHS = 30
earlystopper = EarlyStopping(monitor="val_loss", patience=8, verbose=1)
annealer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6)

train_model = model.fit(train_ds, epochs=NO_EPOCHS, verbose=1,
                        validation_data=val_ds, callbacks=[earlystopper, annealer])

# ---------- 五、曲线 ----------
acc = train_model.history["accuracy"]
val_acc = train_model.history["val_accuracy"]
loss_h = train_model.history["loss"]
val_loss = train_model.history["val_loss"]
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right"); plt.title("Training and Validation Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_h, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right"); plt.title("Training and Validation Loss")
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=110)
plt.close()

# ---------- 六、混淆矩阵与报告 ----------
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

val_pre, val_label = [], []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    for p, l in zip(probs, labels):
        val_pre.append(class_names[int(np.argmax(p))])
        val_label.append(class_names[int(l)])


def plot_cm(labels, predictions):
    conf_numpy = confusion_matrix(labels, predictions)
    conf_df = pd.DataFrame(conf_numpy, index=class_names, columns=class_names)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_df, annot=True, fmt="d", cmap="BuGn")
    plt.xlabel("预测值"); plt.ylabel("真实值"); plt.title("混淆矩阵")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=110)
    plt.close()


plot_cm(val_label, val_pre)

report = __import__("sklearn").metrics.classification_report(val_label, val_pre)
print(report)

model.save(OUT_DIR / "bca_model.keras")
with open(OUT_DIR / "result.txt", "w", encoding="utf-8") as f:
    mode = "合成数据验证" if SYNTHETIC else "真实 bca_data 数据"
    f.write(f"实验7 乳腺癌识别（{mode}）\n")
    f.write(f"图片总数: {image_count}  类别: {class_names}\n")
    f.write(report)
print("结果已保存到", OUT_DIR)
