# -*- coding: utf-8 -*-
"""
实验6：车牌识别（CNN + 多标签分类）
按《6.车牌识别.pdf》实现：
  一、前期准备：数据集目录 licence_plate，文件名形如 "序号_车牌号.jpg"（车牌号含省份简称）
      从文件名提取车牌字符串 -> text2vec 编码为 [label_name_len, char_set_len] 的 one-hot
  二、数据预处理：decode_jpeg 灰度 -> resize (50,200) -> /255 -> shuffle -> batch
  三、构建模型：Conv2D(32,3x3)+Pool -> Conv2D(64,3x3)+Pool -> Flatten
              -> Dense(1000) x2 -> Dropout(0.3)
              -> Dense(label_name_len*char_set_len) -> Reshape([7, char_set_len]) -> Softmax
  四、训练：Adam(1e-4)，categorical_crossentropy，epochs=50
  五、评估曲线；七、保存/加载模型；八、预测并 vec2text 还原车牌文本
用法：
  真实数据：把车牌图片放到 datasets/licence_plate/（文件名 序号_车牌号.jpg）
      .venv/Scripts/python.exe experiments/06-plate/train_plate.py
  合成验证：--synthetic 用 PIL 生成合成车牌图片
"""
import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/
import path_config  # noqa: F401

import numpy as np
np.random.seed(1)
import random
random.seed(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
import tensorflow as tf
tf.random.set_seed(1)

ROOT = path_config.ROOT
DATA_DIR = ROOT / "datasets" / "licence_plate"
OUT_DIR = ROOT / "outputs" / "06-plate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC = "--synthetic" in sys.argv
if SYNTHETIC and not DATA_DIR.exists():
    """生成合成车牌图片（蓝底白字 / 绿牌渐变底黑字，文件名与课程数据同名规则）"""
    from PIL import Image, ImageDraw, ImageFont
    provinces = ["京", "沪", "粤", "川", "鄂"]
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    digits = "0123456789"
    # Windows 自带黑体支持中文
    font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    n = 600
    for i in range(n):
        plate = random.choice(provinces) + random.choice(letters) + \
            "".join(random.choice(letters + digits) for _ in range(5))
        img = Image.new("RGB", (226, 72), (20, 60, 180) if i % 3 else (30, 120, 60))
        dr = ImageDraw.Draw(img)
        # 逐字符绘制（省份简称 + 字母数字）
        x = 8
        for ch in plate:
            dr.text((x, 16), ch, font=font, fill=(255, 255, 255))
            x += 30 if ch in provinces else 28
        # 加一点噪声条纹模拟真实车牌
        for _ in range(30):
            xx, yy = random.randint(0, 225), random.randint(0, 71)
            dr.point((xx, yy), fill=(200, 200, 200))
        img.save(DATA_DIR / f"{i:09d}_{plate}.jpg", quality=90)
    print("已生成合成车牌数据 ->", DATA_DIR)

# ---------- 一、前期准备：读取文件名，提取标签 ----------
pictures_paths = [str(p) for p in DATA_DIR.glob("*.jpg")]
random.shuffle(pictures_paths)
all_label_names = [Path(p).name.split("_")[-1].split(".")[0] for p in pictures_paths]
print(pictures_paths[:3])
image_count = len(pictures_paths)
print("图片总数为：", image_count)

# 字符集：所有出现过的字符（省份简称 + 字母 + 数字）
characters = sorted(set("".join(all_label_names)))
print("字符集大小:", len(characters), "车牌长度:", set(len(x) for x in all_label_names))

# 将字符串数字化（指导书 text2vec）
label_name_len = len(all_label_names[0])
char_set_len = len(characters)
char_set = {c: i for i, c in enumerate(characters)}


def text2vec(text):
    """车牌字符串 -> [len, char_set_len] one-hot"""
    vector = np.zeros((label_name_len, char_set_len))
    for i, c in enumerate(text):
        vector[i, char_set[c]] = 1
    return vector


# ---------- 二、数据预处理 ----------
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [50, 200])
    return image / 255.0


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image), label


labels_onehot = np.array([text2vec(t) for t in all_label_names], dtype="float32")
ds = tf.data.Dataset.from_tensor_slices((pictures_paths, labels_onehot))
# 打乱后按 9:1 划分
n_train = int(image_count * 0.9)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
train_ds = (ds.take(n_train).shuffle(5000)
            .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds = (ds.skip(n_train)
          .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# ---------- 三、构建模型 ----------
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(50, 200, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),  # 卷积层1
    layers.MaxPooling2D((2, 2)),                   # 池化层1
    layers.Conv2D(64, (3, 3), activation="relu"),  # 卷积层2
    layers.MaxPooling2D((2, 2)),                   # 池化层2
    layers.Flatten(),                              # 连接卷积与全连接
    layers.Dense(1000, activation="relu"),
    layers.Dense(1000, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(label_name_len * char_set_len),
    layers.Reshape([label_name_len, char_set_len]),
    layers.Softmax(),
])
model.summary()

# ---------- 四、训练 ----------
optimizer = tf.keras.optimizers.Adam(1e-4)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
epochs = 50 if not SYNTHETIC else 30
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# ---------- 五、评估曲线 ----------
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss_h = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right"); plt.title("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_h, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right"); plt.title("Loss")
plt.suptitle("Licence Plate Recognition")
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=110)
plt.close()

# ---------- 七、保存&加载模型 ----------
model.save(OUT_DIR / "plate_model.keras")
new_model = tf.keras.models.load_model(OUT_DIR / "plate_model.keras")


def vec2text(vec):
    """还原标签（向量 -> 字符串）"""
    text = []
    for i, c in enumerate(vec):
        text.append(characters[c])
    return "".join(text)


# ---------- 八、预测 ----------
n_show = 8
plt.figure(figsize=(14, 5))
for images, labels in val_ds.take(1):
    preds = new_model.predict(images, verbose=0)
    for i in range(min(n_show, images.shape[0])):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")
        pred_text = vec2text(np.argmax(preds[i], axis=1))
        true_text = vec2text(np.argmax(labels[i], axis=1))
        plt.title(f"pred: {pred_text}\ntrue: {true_text}", fontsize=9)
        plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "predict_grid.png", dpi=110)
plt.close()

# 整体验证集字符级准确率
correct = total = 0
for images, labels in val_ds:
    preds = new_model.predict(images, verbose=0)
    for i in range(images.shape[0]):
        p = np.argmax(preds[i], axis=1)
        t = np.argmax(labels[i].numpy(), axis=1)
        correct += int((p == t).sum())
        total += len(t)
char_acc = correct / total
print(f"验证集字符级准确率: {char_acc:.4f}")

with open(OUT_DIR / "result.txt", "w", encoding="utf-8") as f:
    mode = "合成数据验证" if SYNTHETIC else "真实 licence_plate 数据"
    f.write(f"实验6 车牌识别（{mode}）\n")
    f.write(f"图片总数: {image_count}  字符集: {char_set_len}  epochs={epochs}\n")
    f.write(f"验证集字符级准确率: {char_acc:.4f}\n")
print("结果已保存到", OUT_DIR)
