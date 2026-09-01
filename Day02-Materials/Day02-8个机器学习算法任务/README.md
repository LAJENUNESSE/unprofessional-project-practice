# Day02 八个机器学习算法任务 — 完成说明文档

> 完成日期：2026-09-01
> 运行环境：Python 3.14.5（venv）+ scikit-learn 1.9.0 + pandas 3.0.5 + numpy 2.5.2 + matplotlib 3.11.1
> 状态：**10 个 notebook 全部填写、执行通过**，输出与图表均已保存在 ipynb 文件内，直接打开即可查看结果。

---

## 一、目录结构

```
Day02-8个机器学习算法任务/
├── 01 机器学习绪论/              # 课程讲义（01 机器学习绪论.pdf、02 scikit-learn.pdf）
├── 02 监督学习-K近邻/
│   ├── 任务1_使用K-最近邻算法构建鸢尾花分类模型 - 学生版.ipynb   ✅ 已完成
│   ├── 任务2_基于K-最近邻算法构建红酒分类模型 - 学生版.ipynb     ✅ 已完成
├── 03 回归/
│   ├── 任务3_使用线性回归构建波士顿房价预测模型 - 学生版.ipynb   ✅ 已完成（数据集已替换，见说明）
│   ├── 任务4_使用逻辑回归构建肿瘤预测模型 -学生版.ipynb          ✅ 已完成
├── 05 决策树/
│   ├── abalone.data                                              # UCI 鲍鱼数据集（已下载到本地）
│   ├── 任务5_基于决策树及集成算法的回归与分类案例 - 学生版.ipynb ✅ 已完成
├── 06 聚类/
│   ├── 任务6.1_基于多种算法实现鸢尾花聚类 - 学生版.ipynb         ✅ 已完成
│   ├── 任务6.2_环形数据集聚类_学生版.ipynb                       ✅ 已完成
├── 07 降维/
│   ├── 任务7_基于PCA与LDA的数据降维实践_学生版.ipynb             ✅ 已完成
└── 08 模型评估/
    ├── 任务8.1_使用交叉验证评估模型_学生版.ipynb                 ✅ 已完成
    ├── 任务8.2_绘制ROC曲线及P-R曲线_学生版.ipynb                 ✅ 已完成
```

每个章节文件夹内同时保留了原任务的 PDF 说明文件。

## 二、各任务完成情况与关键结果

| 任务 | 主题 | 关键结果 | 备注 |
|------|------|----------|------|
| 任务1 | 鸢尾花 KNN 分类 | 测试集准确率 **0.974**（K=3） | 含数据探索、DataFrame 构建、花萼/花瓣散点图、新样本预测（预测为 setosa） |
| 任务2 | 红酒 KNN 分类 | 测试集准确率 **0.73**（默认 K=5） | 特征未做标准化，属教学演示预期结果；含 10 个随机特征组合散点图 |
| 任务3 | 线性回归房价预测 | 线性回归 test R² **0.64**，Ridge test R² **0.63** | ⚠️ 数据集已替换，见第三节 |
| 任务4 | 逻辑回归肿瘤预测 | cancer test **0.951**；iris 调参后（C=100）test **0.974** | 含 predict / predict_proba 对比 |
| 任务5 | 决策树回归 + 集成分类 | 鲍鱼年龄回归 R² **0.145**；分类：无剪枝树 0.88 / 剪枝树(max_depth=4) 0.90 / 随机森林 **0.97** / AdaBoost 0.94 | 直观展示决策树过拟合（train 1.00）与集成算法提升 |
| 任务6.1 | 多算法鸢尾花聚类 | KMeans（花瓣长宽两特征）ARI **0.886** | Birch/KMeans 一维演示 + make_blobs 四簇 + 聚类结果与真实标签双图对比 |
| 任务6.2 | DBSCAN 环形数据 | `DBSCAN(eps=0.2, min_samples=5)` 成功聚出**内、中、外 3 类**（0 噪声点） | 先演示 KMeans/Birch/MeanShift 失败，再调参成功 |
| 任务7 | PCA 与 LDA 降维 | iris：4 维→2 维 KNN 准确率不变（0.974）；wine：LDA 降维后逻辑回归 **1.0** > 原始 4 特征的 0.978 | 结论：LDA 利用标签信息，更适合有监督降维 |
| 任务8.1 | 交叉验证 | 3 折均值 **0.973**；留一法 150 折均值 **0.967**；ShuffleSplit（固定样本数/百分比两种）均通过 | |
| 任务8.2 | ROC 与 P-R 曲线 | 三种核函数 SVC 的 ROC 曲线对比（AUC≈0.99~1.0）；逻辑回归 P-R 曲线 | 使用 decision_function 计算距离，roc_curve/auc/precision_recall_curve 绘图 |

> 所有图表（散点图、预测对比曲线、ROC、P-R）均已生成并嵌入对应 notebook。

## 三、重要说明

### 1. 任务3 数据集替换（load_boston → OpenML boston）

原任务使用 `load_boston`，该函数自 **scikit-learn 1.2 起已被移除**（伦理争议：数据中含"每10万美元收入人群占比"这一特征）。本任务按如下方式加载**同一份原版数据**（506 条 × 13 特征）：

```python
from sklearn.datasets import fetch_openml
boston = fetch_openml(name="boston", version=1, as_frame=False, parser='auto')
```

首次运行需联网从 OpenML 下载（已缓存至本地 `~/scikit_learn_data/`），之后可离线运行。回归结果与经典教材一致（线性回归 test R²≈0.64）。

### 2. 任务5 数据文件

`abalone.data` 已从 UCI 机器学习库下载到「05 决策树」目录（4176 条），notebook 使用 `pd.read_csv('abalone.data', header=None)` 本地加载，离线可跑。Sex 特征 M/F/I 已按要求映射为 0/1/2。

### 3. 中文显示

每个 notebook 的第一个 cell 添加了环境配置：

```python
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

保证 matplotlib 中文标签正常显示（Windows 系统自带该字体）。

### 4. 随机性说明

所有涉及随机的过程（train_test_split、模型初始化、make_blobs 等）均已固定 `random_state`，结果可复现。任务6.1 第一、二个 cell 中的 `np.random.randint` 未固定种子，每次运行机选数字会不同，但聚类结论稳定。

## 四、如何运行

```bash
# 1. 激活虚拟环境（Day02-Materials 目录下）
.venv/Scripts/activate        # Git Bash
# 或 .venv\Scripts\activate.bat   （CMD）

# 2. 启动 Jupyter Lab / Notebook
jupyter lab
# 或直接用 VS Code 打开 .ipynb，右上角选择 .venv 解释器
```

依赖已全部安装在 `Day02-Materials\.venv`（含 ipykernel）。若在别处新建环境，安装清单：

```bash
pip install numpy pandas matplotlib scikit-learn ipykernel nbformat nbclient
```

## 五、核心知识点小结

1. **KNN**：基于距离的分类，对特征尺度敏感（任务2 的 0.73 就是未标准化的体现，可用 StandardScaler + Pipeline 提升）。
2. **线性/逻辑回归**：`score()` 分别输出 R² 与准确率；`predict_proba` 输出类别概率。
3. **决策树**：无约束必然过拟合（train=1.0, test 低）；`max_depth` 剪枝可缓解；随机森林/AdaBoost 等集成方法显著提升泛化。
4. **聚类**：KMeans 需指定 K，只能发现凸形簇；DBSCAN 基于密度，能发现任意形状簇（环形数据），核心参数 `eps` 与 `min_samples`。
5. **降维**：PCA 无监督（最大方差方向），LDA 有监督（类间距离最大）；有标签场景 LDA 通常更优。
6. **模型评估**：K 折交叉验证 / 留一法 / ShuffleSplit 的适用场景；ROC 曲线（TPR-FPR）与 P-R 曲线（精确率-召回率），AUC 越接近 1 越好。
