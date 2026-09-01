# -*- coding: utf-8 -*-
"""
从剑桥大学 AT&T 人脸库（att_faces.zip，400 张 92x112 PGM，40 人 x 10 张）
构建 sklearn fetch_olivetti_faces 期望的缓存文件 olivetti.pkz：
  内容 = faces 数组 (400, 4096) uint8（每行一张 64x64 转置展平图，0-255）
fetch_olivetti_faces 找到 olivetti.pkz 后不再联网。
用法：.venv/Scripts/python.exe experiments/01-face-olivetti/build_olivetti_cache.py <att_faces.zip路径>
"""
import sys
import zipfile
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import path_config  # noqa: F401

zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\liu69\AppData\Local\Temp\day03-data\att_faces.zip")
out_pkz = path_config.SKLEARN_DATA_DIR / "olivetti.pkz"

faces = np.empty((400, 4096), dtype=np.uint8)
with zipfile.ZipFile(zip_path) as z:
    names = sorted(n for n in z.namelist() if n.lower().endswith(".pgm"))
    assert len(names) == 400, f"期望 400 张 PGM，实际 {len(names)}"
    for i, name in enumerate(names):
        img = Image.open(BytesIO(z.read(name))).convert("L")
        img64 = np.asarray(img.resize((64, 64), Image.BILINEAR), dtype=np.uint8)
        # sklearn 内部: faces.reshape(400,64,64).transpose(0,2,1)
        # 故此处存入 (64,64) 的转置展平，还原后即为正确朝向
        faces[i] = img64.T.reshape(-1)
        if i % 100 == 0:
            print(f"{i}/400")

joblib.dump(faces, out_pkz, compress=6)
print("已生成:", out_pkz, f"({out_pkz.stat().st_size/1024:.0f} KB)")

# 自检：模拟 sklearn 的加载流程
faces_chk = np.float32(joblib.load(out_pkz))
faces_chk -= faces_chk.min()
faces_chk /= faces_chk.max()
imgs = faces_chk.reshape((400, 64, 64)).transpose(0, 2, 1)
assert imgs.shape == (400, 64, 64)
print("自检通过，images.shape =", imgs.shape)
