# -*- coding: utf-8 -*-
"""
实验3：AlexNet 实现鸟类识别
按《3.卷积神经网络AlexNet实现鸟类识别.pdf》实现：
  一、前期准备：data_dir 指向鸟类图片目录（按类别分子目录），统计图片数
  二、数据预处理：image_dataset_from_directory 8/2 划分 train/val（227x227, batch=8）
                -> class_names -> 可视化 -> cache/shuffle/prefetch
  三、构建 AlexNet(8层)：5 个卷积块（96/256/384/384/256 + BN + 池化）
        -> Flatten -> Dense(4096)x2 + Dropout(0.5) -> Dense(nb_classes, softmax)
  四、训练 epochs=20
  五、模型评估：accuracy/loss 曲线
  七、保存&加载模型
  八、预测：对验证集 8 张图预测并展示
用法：
  真实数据：把 bird_photos 放到 datasets/bird_photos/（子目录为 4 类鸟名）
      .venv/Scripts/python.exe experiments/03-alexnet-birds/train_alexnet_birds.py
  合成验证：--synthetic 生成 4 类合成鸟类纹理图跑通全流程
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
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten,
                                     Dropout, BatchNormalization, Activation)

ROOT = path_config.ROOT
DATA_DIR = ROOT / "datasets" / "bird_photos"
OUT_DIR = ROOT / "outputs" / "03-alexnet-birds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC = "--synthetic" in sys.argv
if SYNTHETIC and not DATA_DIR.exists():
    """生成 4 类合成"鸟"图（不同颜色纹理的 224x224 彩图，每类 ~140 张）"""
    rng = np.random.default_rng(1)
    classes = ["Bananaquit", "Black Skimmer", "Black Throated Bushtiti", "Cockatoo"]
    base_colors = [(230, 200, 60), (40, 60, 120), (20, 20, 20), (240, 240, 240)]
    for c, cls in enumerate(classes):
        d = DATA_DIR / cls
        d.mkdir(parents=True, exist_ok=True)
        from PIL import Image, ImageDraw
        for i in range(140):
            arr = rng.normal(0, 30, (224, 224, 3)) + np.array(base_colors[c])
            # 每类不同几何图案，模拟可区分特征
            img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
            dr = ImageDraw.Draw(img)
            for k in range(c + 2):
                dr.ellipse([20 + k * 30, 30 + k * 25, 90 + k * 30, 100 + k * 25],
                           outline=tuple(int(x * 0.3) for x in base_colors[c]), width=6)
            img.save(d / f"{cls}_{i:03d}.jpg", quality=85)
    print("已生成合成鸟类数据 ->", DATA_DIR)

data_dir = DATA_DIR
data_dir = Path(data_dir)
image_count = len(list(data_dir.glob("*/*")))
print("图片总数为：", image_count)

# ---------- 二、数据预处理 ----------
batch_size = 8
img_height = 227
img_width = 227

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# 可视化
plt.figure(figsize=(10, 5))
plt.suptitle("Bird Identification")
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "samples.png", dpi=110)
plt.close()

# 再次检查数据
for image_batch, labels_batch in train_ds:
    print(image_batch.shape, labels_batch.shape)
    break

# 配置数据集
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ---------- 三、构建 AlexNet（8层）网络模型 ----------
def AlexNet(nb_classes, input_shape):
    input_tensor = Input(shape=input_shape)
    # 1st block
    x = Conv2D(96, (11, 11), strides=4, name="block1_conv1")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=2, name="block1_pool")(x)
    # 2nd block
    x = Conv2D(256, (5, 5), padding="same", name="block2_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=2, name="block2_pool")(x)
    # 3rd block
    x = Conv2D(384, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    # 4th block
    x = Conv2D(384, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    # 5th block
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = MaxPooling2D((3, 3), strides=2, name="block5_pool")(x)
    # full connection
    x = Flatten()(x)
    x = Dense(4096, activation="relu", name="fc1")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu", name="fc2")(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(nb_classes, activation="softmax", name="predictions")(x)
    model = models.Model(input_tensor, output_tensor)
    return model


nb_classes = len(class_names)  # 指导书示例为 1000 类 ImageNet 头；此处按数据集实际 4 类
model = AlexNet(nb_classes, (img_width, img_height, 3))
model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ---------- 四、训练模型 ----------
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# ---------- 五、模型评估 ----------
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.suptitle("Bird Identification")
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=110)
plt.close()

# ---------- 七、保存&加载模型 ----------
model.save(OUT_DIR / "my_model.keras")
new_model = tf.keras.models.load_model(OUT_DIR / "my_model.keras")

# ---------- 八、预测 ----------
plt.figure(figsize=(10, 5))
plt.suptitle("Bird identification (predict)")
for images, labels in val_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        img_array = tf.expand_dims(images[i], 0)
        predictions = new_model.predict(img_array, verbose=0)
        plt.title(f"{class_names[np.argmax(predictions)]}\n(true: {class_names[labels[i]]})",
                  fontsize=8)
        plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "predict_grid.png", dpi=110)
plt.close()

test_loss, test_acc = model.evaluate(val_ds, verbose=0)
with open(OUT_DIR / "result.txt", "w", encoding="utf-8") as f:
    mode = "合成数据验证" if SYNTHETIC else "真实 bird_photos 数据"
    f.write(f"实验3 AlexNet 鸟类识别（{mode}）\n")
    f.write(f"图片总数: {image_count}  类别: {class_names}\n")
    f.write(f"验证集准确率: {test_acc:.4f}\n")
print(f"验证集 accuracy={test_acc:.4f}，结果已保存到 {OUT_DIR}")
