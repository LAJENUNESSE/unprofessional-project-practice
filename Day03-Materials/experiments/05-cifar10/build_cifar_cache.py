# -*- coding: utf-8 -*-
"""
把 hf-mirror 下载的 CIFAR-10 parquet 转成 kerasdatasets.cifar10.load_data() 能直接用的
~/.keras/datasets/cifar-10-batches-py-target/cifar-10-batches-py/ 目录
（data_batch_1..5、test_batch、batches.meta，pickle 格式与原版一致），
这样实验5脚本无需联网即可 load_data()。
用法：.venv/Scripts/python.exe experiments/05-cifar10/build_cifar_cache.py
"""
import pickle
import io
import os
import sys
import numpy as np
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("需要 pyarrow: pip install pyarrow pandas pillow")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import path_config  # noqa: F401

PARQUET_DIR = Path(os.environ.get("TEMP", r"C:\Users\liu69\AppData\Local\Temp")) / "day03-data"
KERAS_DATASETS = path_config.KERAS_DATASETS
OUT = KERAS_DATASETS / "cifar-10-batches-py-target" / "cifar-10-batches-py"
OUT.mkdir(parents=True, exist_ok=True)


def parquet_to_batches(parquet_path: Path, n_batches: int):
    """读取 HF parquet（img 为 PNG bytes, label 为 int），返回 [(data Nx3072 uint8, labels N,), ...]"""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    n = len(df)
    assert n == 10000 * n_batches or n_batches == 1, f"行数 {n} 与批数不符"
    per = n // n_batches
    batches = []
    # 逐批解码，控制内存
    labels_all = df["label"].to_numpy()
    for b in range(n_batches):
        chunk = df.iloc[b * per : (b + 1) * per]
        data = np.empty((per, 3072), dtype=np.uint8)
        from PIL import Image
        for i, raw in enumerate(chunk["img"]):
            if isinstance(raw, dict):
                raw = raw["bytes"]
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)  # (32,32,3)
            # 原版格式：R 1024 | G 1024 | B 1024，行优先
            data[i] = np.concatenate([arr[:, :, 0].ravel(), arr[:, :, 1].ravel(), arr[:, :, 2].ravel()])
        batches.append((data, labels_all[b * per : (b + 1) * per].astype(np.int64).tolist()))
        print(f"{parquet_path.name} batch {b+1}/{n_batches} done")
    return batches


LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

# 训练集 -> data_batch_1..5
train_batches = parquet_to_batches(PARQUET_DIR / "cifar-train.parquet", 5)
for i, (data, labels) in enumerate(train_batches, 1):
    with open(OUT / f"data_batch_{i}", "wb") as f:
        pickle.dump({b"data": data, b"labels": labels,
                     b"batch_label": f"training batch {i} of 5",
                     b"filenames": [f"img_{j}" for j in range(len(labels))]}, f)

# 测试集 -> test_batch
test_data, test_labels = parquet_to_batches(PARQUET_DIR / "cifar-test.parquet", 1)[0]
with open(OUT / "test_batch", "wb") as f:
    pickle.dump({b"data": test_data, b"labels": test_labels,
                 b"batch_label": "testing batch 1 of 1",
                 b"filenames": [f"img_{j}" for j in range(len(test_labels))]}, f)

with open(OUT / "batches.meta", "wb") as f:
    pickle.dump({b"label_names": [x.encode() for x in LABELS],
                 b"num_cases_per_batch": 10000, b"num_vis": 3072}, f)

print("CIFAR-10 缓存已生成:", OUT)
