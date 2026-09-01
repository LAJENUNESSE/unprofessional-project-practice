# Day03 实验结果汇总（2026-09-01）

环境：`Day03-Materials/.venv`，Python 3.12.10 + TensorFlow 2.21.0（CPU）+ Keras 3.15

| # | 实验 | 数据 | 结果 | 输出目录 |
|---|---|---|---|---|
| 1 | 人脸识别（Olivetti，40 类） | AT&T 官方 zip 转离线缓存 | 训练 acc 1.000，**测试 acc 0.8667**（104/120） | `outputs/01-face-olivetti` |
| 2 | 表情识别（fer2013，7 类） | ⚠ 合成数据验证 | 全流程跑通：训练/评估/保存 h5/加载/预测自定义图均 OK | `outputs/02-fer2013` |
| 3 | 鸟类识别（AlexNet，4 类） | ⚠ 合成数据验证（560 张 227x227） | 20 epochs，**val acc 0.7679**，预测网格图正常 | `outputs/03-alexnet-birds` |
| 4 | MNIST 手写数字（10 类） | 本地缓存 | 训练 acc 0.9975，**测试 acc 0.9910**（指导书参考 ~0.989） | `outputs/04-mnist` |
| 5 | CIFAR-10 彩色分类（10 类） | hf-mirror parquet → keras 缓存 | 训练 acc 0.7915，**测试 acc 0.7208**（指导书参考 ~0.7046） | `outputs/05-cifar10` |
| 6 | 车牌识别（CNN+多标签 one-hot） | ⚠ 合成数据验证（600 张合成车牌） | 30 epochs，**验证集字符级 acc 0.6786**，pred/true 还原正常 | `outputs/06-plate` |
| 7 | 乳腺癌识别（二分类） | ⚠ 合成数据验证（1200 张 50x50） | 30 epochs + EarlyStopping，**val acc 1.00**，混淆矩阵/报告完整 | `outputs/07-breast-cancer` |

## 说明

- 标 ⚠ 的 4 个实验（2/3/6/7）代码 100% 按指导书实现，因课程私有数据集未下发，
  用结构化合成数据验证了端到端流程（数据加载→训练→评估→保存→加载→预测）。
  课程数据到位后按 `README.md` 的路径放入 `datasets/`，去掉 `--synthetic` 重跑即得真实结果。
- 实验 1 的 olivetti 数据因 figshare 被网络拒绝（403），改用剑桥大学 AT&T 官方 zip
  转换为 sklearn 缓存格式（`build_olivetti_cache.py`），内容与官方 npz 等价。
- 实验 5 的 CIFAR-10 因官方源仅 9KB/s，从 hf-mirror 下载 parquet 并转换为
  keras batches 格式（`build_cifar_cache.py`），load_data() 直接离线加载。
- 与指导书的差异：
  - 实验 1：train_test_split 增加 `random_state=42` 保证可复现（指导书未固定）。
  - 实验 2：`epochs` 合成验证用 2（同指导书），真实数据建议 15；`image.load_img`
    的 `grayscale=` 参数在 Keras 3 中已移除，改为 PIL 灰度化。
  - 实验 3：AlexNet 输出类别数按数据集实际 4 类（指导书示例代码写 1000 为 ImageNet 头）。
  - 模型保存：指导书的 `.h5` 保存保留（实验2），其余统一用 Keras 3 原生 `.keras` 格式。
