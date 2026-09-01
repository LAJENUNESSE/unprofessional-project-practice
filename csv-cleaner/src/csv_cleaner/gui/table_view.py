"""ttk.Treeview 数据表格视图：带滚动条，支持分页式预览与全量浏览。"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pandas as pd

PREVIEW_ROWS = 200  # Treeview 一次最多渲染行数，防止大表卡界面


class DataFrameTable(ttk.Frame):
    """用 Treeview 展示 DataFrame，超长时截断并提示。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._notice = tk.StringVar(value="尚未打开数据")
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(bar, textvariable=self._notice, foreground="#666").pack(side="left")

        self.tree = ttk.Treeview(self, show="headings")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        hsb.grid(row=2, column=0, sticky="ew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

    def set_dataframe(self, df: pd.DataFrame, title: str = "数据预览") -> None:
        """整表刷新。"""
        self.tree.delete(*self.tree.get_children())
        if df is None or df.empty:
            self._notice.set("暂无数据")
            return
        cols = [str(c) for c in df.columns]
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=110, anchor="w", stretch=False)
        shown = df.head(PREVIEW_ROWS)
        for _, row in shown.iterrows():
            values = ["" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v))
                      for v in row]
            self.tree.insert("", "end", values=values)
        extra = f"（仅显示前 {PREVIEW_ROWS} 行）" if len(df) > PREVIEW_ROWS else ""
        self._notice.set(f"{title}：{len(df)} 行 × {len(df.columns)} 列 {extra}")
