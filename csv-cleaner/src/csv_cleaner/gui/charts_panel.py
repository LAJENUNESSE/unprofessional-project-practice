"""matplotlib 图表面板：直方图 / 箱线图 / 柱状图 / 相关性热力图。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 修复负号显示为方块

HIST = "直方图"
BOX = "箱线图"
BAR = "柱状图(频次TOP10)"
HEAT = "相关性热力图"
CHART_TYPES = [HIST, BOX, BAR, HEAT]


class ChartsPanel(ttk.Frame):
    """选择列与图表类型后绘制，matplotlib 画布嵌入 Tkinter。"""

    def __init__(self, master: tk.Misc, get_df) -> None:
        """get_df: 返回当前 DataFrame 的回调（保持与主窗口数据同步）。"""
        super().__init__(master)
        self._get_df = get_df

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=2)
        self.col_var = tk.StringVar()
        self.col_box = ttk.Combobox(bar, textvariable=self.col_var, width=16, state="readonly")
        ttk.Label(bar, text="选择列：").pack(side="left")
        self.col_box.pack(side="left", padx=(0, 12))
        self.type_var = tk.StringVar(value=HIST)
        ttk.Label(bar, text="图表类型：").pack(side="left")
        type_box = ttk.Combobox(bar, textvariable=self.type_var, width=18,
                                values=CHART_TYPES, state="readonly")
        type_box.pack(side="left", padx=(0, 12))
        ttk.Button(bar, text="绘制", command=self.draw).pack(side="left")

        self.msg_var = tk.StringVar(value="打开数据后选择列与图表类型。")
        ttk.Label(self, textvariable=self.msg_var, foreground="#666").pack(anchor="w")

        self.figure = Figure(figsize=(6.4, 4.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh_columns(self, df: pd.DataFrame | None) -> None:
        """数据变化后刷新列下拉框。"""
        if df is None or df.empty:
            self.col_box["values"] = []
            self.col_var.set("")
            return
        numeric = list(df.select_dtypes("number").columns)
        # 默认选中第一个数值列，直方图开箱即用
        self.col_box["values"] = [str(c) for c in df.columns]
        if numeric and self.col_var.get() not in self.col_box["values"]:
            self.col_var.set(str(numeric[0]))

    def draw(self) -> None:
        df = self._get_df()
        if df is None or df.empty:
            self.msg_var.set("请先打开 CSV 文件。")
            return
        chart = self.type_var.get()
        column = self.col_var.get()
        if chart != HEAT and not column:
            self.msg_var.set("请先选择列（热力图除外）。")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        try:
            if chart == HIST:
                self._hist(ax, df, column)
            elif chart == BOX:
                self._box(ax, df, column)
            elif chart == BAR:
                self._bar(ax, df, column)
            else:
                self._heat(ax, df)
        except Exception as exc:  # 图表绘制失败不应崩溃整个程序
            self.msg_var.set(f"绘制失败：{exc}")
            self.canvas.draw_idle()
            return
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _hist(self, ax, df: pd.DataFrame, column: str) -> None:
        s = pd.to_numeric(df[column], errors="coerce").dropna()
        if s.empty:
            raise ValueError(f"列「{column}」没有可绘图的数值")
        s.plot.hist(ax=ax, bins=20, color="#4C72B0", edgecolor="white")
        ax.set_title(f"「{column}」分布直方图")
        ax.set_xlabel(column)
        ax.set_ylabel("频数")

    def _box(self, ax, df: pd.DataFrame, column: str) -> None:
        s = pd.to_numeric(df[column], errors="coerce").dropna()
        if s.empty:
            raise ValueError(f"列「{column}」没有可绘图的数值")
        ax.boxplot(s, vert=False)
        ax.set_title(f"「{column}」箱线图（圆点为异常值）")
        ax.set_yticks([])

    def _bar(self, ax, df: pd.DataFrame, column: str) -> None:
        top = df[column].value_counts().head(10)
        if top.empty:
            raise ValueError(f"列「{column}」没有数据")
        ax.bar([str(i) for i in top.index], top.values, color="#55A868")
        ax.set_title(f"「{column}」频次 TOP{len(top)}")
        ax.set_ylabel("频数")
        ax.tick_params(axis="x", rotation=30)

    def _heat(self, ax, df: pd.DataFrame) -> None:
        corr = df.select_dtypes("number").corr()
        if corr.shape[1] < 2:
            raise ValueError("数值列不足 2 列，无法绘制相关性热力图")
        img = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        cols = [str(c) for c in corr.columns]
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
        ax.set_yticks(range(len(cols)), cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("数值列相关性热力图")
        self.figure.colorbar(img, ax=ax, shrink=0.8)
