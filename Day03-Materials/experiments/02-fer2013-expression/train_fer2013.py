# -*- coding: utf-8 -*-
"""
实验2：基于卷积神经网络的表情识别（fer2013）
按《2.基于卷积神经网络的表情识别案例.pdf》实现：
  步骤一 加载数据（data/fer2013/train.csv, test.csv, val.csv，emotion+pixels 两列）
  步骤二 可视化前 4 张
  步骤三 搭建 CNN（3 组 双卷积+池化：32/64/128）-> Dense(64) -> Dense(7 softmax)
        训练 rmsprop + sparse_categorical_crossentropy，评估并保存 model/fer.h5
  步骤四/五 加载模型识别自定义图片（predict_show）
用法：
  真实数据：把 fer2013 的 train/test/val.csv 放到 datasets/fer2013/ 后
      .venv/Scripts/python.exe experiments/02-fer2013-expression/train_fer2013.py
  合成验证（无真实数据时验证代码全流程）：
      .venv/Scripts/python.exe experiments/02-fer2013-expression/train_fer2013.py --synthetic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/
import path_config  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

ROOT = path_config.ROOT
DATA_DIR = ROOT / "datasets" / "fer2013"
OUT_DIR = ROOT / "outputs" / "02-fer2013"
MODEL_PATH = OUT_DIR / "fer.h5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC = "--synthetic" in sys.argv


def split_data(data):
    """按指导书：拆分 pixels（空格分隔）与 emotion 标签，转为 4D 张量"""
    X, y = [], []
    pixels = data["pixels"]
    emotions = data["emotion"]
    for pixel, emotion in zip(pixels, emotions):
        pixel = np.array(pixel.split(" "), "float32")
        X.append(pixel)
        y.append(emotion)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, 48, 48, 1)
    return X, y


def make_synthetic_csv(n_train=2000, n_test=400, n_val=400, seed=7):
    """生成 fer2013 同结构合成 CSV（7 类，48x48 灰度，有类间亮度/纹理差异）"""
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in [("train", n_train), ("test", n_test), ("val", n_val)]:
        emotions = rng.integers(0, 7, n)
        pixels = []
        for e in emotions:
            base = 40 + e * 28  # 每类不同基准亮度
            img = rng.normal(base, 25, 48 * 48).clip(0, 255).astype(int)
            # 加一个与类别相关的简单模式（便于 CNN 学到）
            img = img.reshape(48, 48)
            img[:, : (e + 1) * 6] = (img[:, : (e + 1) * 6] * 0.5).astype(int)
            pixels.append(" ".join(map(str, img.ravel())))
        rows.append(pd.DataFrame({"emotion": emotions, "pixels": pixels}))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for df, name in zip(rows, ["train.csv", "test.csv", "val.csv"]):
        df.to_csv(DATA_DIR / name, index=False)
    print("已生成合成 fer2013 CSV ->", DATA_DIR)


if SYNTHETIC and not all((DATA_DIR / f).exists() for f in ["train.csv", "test.csv", "val.csv"]):
    make_synthetic_csv()

# ---------- 步骤一：加载和处理数据 ----------
train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")
val_data = pd.read_csv(DATA_DIR / "val.csv")

X_train, y_train = split_data(train_data)
X_test, y_test = split_data(test_data)
X_val, y_val = split_data(val_data)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)

# ---------- 步骤二：查看前 4 张 ----------
plt.figure(figsize=(10, 3))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.gray()
    plt.imshow(X_train[i].reshape([48, 48]))
    plt.title(int(y_train[i]))
    plt.axis("off")
plt.suptitle("fer2013 samples (title=emotion)")
plt.tight_layout()
plt.savefig(OUT_DIR / "samples.png", dpi=110)
plt.close()

# ---------- 步骤三：搭建 CNN，训练并保存 ----------
batch_size = 64  # 每批数据量
epochs = 2 if SYNTHETIC else 15  # 指导书示例 epochs=2；真实数据加大到 15 保底收敛

model = models.Sequential()  # 创建序贯模型
# 第一层卷积
model.add(layers.Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation="relu"))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPool2D())
# 第二层卷积
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPool2D())
# 第三层卷积
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())  # 压平
# 全连接层
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(7, activation="softmax"))  # 输出 7 类表情

model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
model.summary()
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val))

# 评估模型
score = model.evaluate(X_test, y_test, verbose=0)
print("Test acc:", score[1])

# 保存模型（Keras3 需 save_format 或 .h5 后缀 + h5 依赖；此处按指导书保存为 h5）
MODEL_PATH.parent.mkdir(exist_ok=True)
model.save(MODEL_PATH)
print("模型已保存:", MODEL_PATH)

# ---------- 步骤四：加载保存的模型并识别自定义图片 ----------
from tensorflow.keras.models import load_model

new_model = load_model(MODEL_PATH)
new_model.summary()

labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "natural"]


def predict_show(src, save_path=None):
    """显示图片并识别表情（指导书 predict_show）"""
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    img = plt.imread(src)
    plt.imshow(img)
    # 灰化、调整大小
    from PIL import Image
    im = Image.open(src).convert("L").resize((48, 48))
    x = np.asarray(im, dtype="float32")
    x = x.reshape(-1, 48, 48, 1)
    result = new_model.predict(x, verbose=0)
    emotion = labels[int(np.argmax(result))]
    plt.subplot(1, 2, 2)
    plt.gray()
    plt.imshow(x.reshape(48, 48))
    plt.title(emotion, fontsize=20)
    if save_path:
        plt.savefig(save_path, dpi=110)
    plt.close()
    return emotion


# 用测试集里的一张作为"自定义图片"演示预测流程
demo_src = OUT_DIR / "demo_face.png"
plt.imsave(demo_src, X_test[0].reshape(48, 48), cmap="gray")
emo = predict_show(str(demo_src), save_path=str(OUT_DIR / "predict_demo.png"))
print(f"自定义图片识别结果: {emo} (真实: {labels[int(y_test[0])]})")

# ---------- 训练曲线 ----------
plt.plot(history.history["acc"], label="train acc")
plt.plot(history.history["val_acc"], label="val acc")
plt.xlabel("epoch"); plt.legend()
plt.title("Experiment 2: fer2013 accuracy")
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_curve.png", dpi=110)
plt.close()

with open(OUT_DIR / "result.txt", "w", encoding="utf-8") as f:
    mode = "合成数据验证" if SYNTHETIC else "真实 fer2013 数据"
    f.write(f"实验2 表情识别（fer2013，{mode}）\n")
    f.write(f"训练: {len(y_train)}  测试: {len(y_test)}  验证: {len(y_val)}  epochs={epochs}\n")
    f.write(f"测试集准确率: {score[1]:.4f}\n")
print("结果已保存到", OUT_DIR)
