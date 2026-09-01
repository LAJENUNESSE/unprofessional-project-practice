"""Tkinter 主窗口：布局、事件绑定与状态管理。"""
from __future__ import annotations

import pathlib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .core.cleaner import Cleaner
from .core.inspector import inspect
from .core.loader import read_csv_any, sniff_encoding
from .core.report import export_report
from .gui.charts_panel import ChartsPanel
from .gui.table_view import DataFrameTable


class App:
    """主应用：持有 Cleaner 状态，左侧操作、右侧四个页签。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("CSV 数据清洗报表工具")
        root.geometry("1100x680")
        root.minsize(900, 560)

        self.cleaner: Cleaner | None = None
        self.source_path: str = ""
        self._column_combos: list[ttk.Combobox] = []

        self._build_menu()
        self._build_layout()
        self._set_status("请打开一个 CSV 文件开始。")

    # ---------- 界面构建 ----------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开 CSV…", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="导出清洗后 CSV…", command=self.export_csv)
        file_menu.add_command(label="导出分析报告 (Markdown)…", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.destroy)
        menubar.add_cascade(label="文件", menu=file_menu)
        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self.open_file())

    def _build_layout(self) -> None:
        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=6, pady=6)

        main.add(self._build_ops_panel(), weight=0)
        main.add(self._build_tabs(), weight=1)
        self.status = ttk.Label(self.root, relief="sunken", anchor="w", padding=(6, 2))
        self.status.pack(fill="x", side="bottom")

    def _build_ops_panel(self) -> ttk.Frame:
        panel = ttk.Frame(self.root, padding=4)

        def section(title: str) -> ttk.LabelFrame:
            frame = ttk.LabelFrame(panel, text=title, padding=6)
            frame.pack(fill="x", pady=4)
            return frame

        # 数据体检
        ttk.Button(section("① 数据体检"), text="重新体检",
                   command=self.run_inspection).pack(fill="x")

        # 行级操作
        row_ops = section("② 行级清洗")
        ttk.Button(row_ops, text="去除完全重复行", command=lambda: self._apply("drop_duplicates")).pack(fill="x", pady=1)
        ttk.Button(row_ops, text="文本列去首尾空格", command=lambda: self._apply("strip_whitespace")).pack(fill="x", pady=1)
        ttk.Button(row_ops, text="删除含缺失值的行", command=lambda: self._apply("drop_missing_rows")).pack(fill="x", pady=1)

        # 缺失值填充
        fill = section("③ 缺失值填充")
        self.col_combo = self._make_column_combo(fill)
        self.fill_strategy = tk.StringVar(value="mean")
        strategy_box = ttk.Combobox(fill, textvariable=self.fill_strategy, state="readonly",
                                    values=["mean(均值)", "median(中位数)", "mode(众数)", "constant(固定值)"])
        strategy_box.pack(fill="x", pady=1)
        strategy_box.current(0)
        self.fill_value = tk.StringVar()
        ttk.Entry(fill, textvariable=self.fill_value).pack(fill="x", pady=1)
        ttk.Button(fill, text="填充所选列", command=self._do_fill).pack(fill="x", pady=1)

        # 类型转换
        conv = section("④ 类型转换")
        self.conv_col = self._make_column_combo(conv)
        self.conv_kind = tk.StringVar(value="数值")
        ttk.Combobox(conv, textvariable=self.conv_kind, state="readonly",
                     values=["数值", "日期"]).pack(fill="x", pady=1)
        ttk.Button(conv, text="转换所选列", command=self._do_convert).pack(fill="x", pady=1)

        # 异常值
        out = section("⑤ 异常值处理 (IQR)")
        self.out_col = self._make_column_combo(out)
        self.out_action = tk.StringVar(value="clip(缩进边界)")
        ttk.Combobox(out, textvariable=self.out_action, state="readonly",
                     values=["clip(缩进边界)", "remove(删除行)"]).pack(fill="x", pady=1)
        ttk.Button(out, text="处理所选列", command=self._do_outliers).pack(fill="x", pady=1)

        # 列管理
        colmgmt = section("⑥ 列管理")
        self.rm_col = self._make_column_combo(colmgmt)
        ttk.Button(colmgmt, text="删除所选列", command=self._do_drop_column).pack(fill="x", pady=1)
        return panel

    def _make_column_combo(self, master: ttk.LabelFrame) -> ttk.Combobox:
        """生成一个由主窗口统一刷新的列选择下拉框。"""
        combo = ttk.Combobox(master, state="readonly")
        combo.pack(fill="x", pady=1)
        self._column_combos.append(combo)
        return combo

    def _build_tabs(self) -> ttk.Frame:
        holder = ttk.Frame(self.root)
        self.tabs = ttk.Notebook(holder)
        self.tabs.pack(fill="both", expand=True)

        self.table = DataFrameTable(self.tabs)
        self.tabs.add(self.table, text="数据预览")

        self.inspect_text = self._make_text_tab("体检结果")
        self.log_text = self._make_text_tab("清洗日志")

        self.charts = ChartsPanel(self.tabs, get_df=self._current_df)
        self.tabs.add(self.charts, text="图表")
        return holder

    def _make_text_tab(self, title: str) -> tk.Text:
        text = tk.Text(self.tabs, wrap="none", state="disabled",
                       font=("Consolas", 10), bg="#FAFAFA")
        self.tabs.add(text, text=title)
        return text

    def _set_text(self, widget: tk.Text, content: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled")

    # ---------- 状态与刷新 ----------

    def _current_df(self):
        return None if self.cleaner is None else self.cleaner.df

    def _require_data(self) -> Cleaner | None:
        if self.cleaner is None:
            messagebox.showinfo("提示", "请先打开一个 CSV 文件。")
            return None
        return self.cleaner

    def _refresh_all(self) -> None:
        """任何数据变更后统一刷新：预览、下拉框、体检页、日志页、状态栏。"""
        c = self.cleaner
        if c is None:
            return
        df = c.df
        self.table.set_dataframe(df)
        columns = [str(x) for x in df.columns]
        for combo in self._column_combos:
            combo["values"] = columns
            if columns and combo.get() not in columns:
                combo.set(columns[0])
        self.charts.refresh_columns(df)

        report = inspect(df)
        self._set_text(self.inspect_text, "\n".join(report.to_lines()))

        if c.log:
            log_lines = [f"{a.step:>2}. [{a.action}] {a.detail}" for a in c.log]
        else:
            log_lines = ["（尚未执行清洗操作）"]
        self._set_text(self.log_text, "\n".join(log_lines))

        self._set_status(f"文件：{pathlib.Path(self.source_path).name}　"
                         f"当前 {len(df)} 行 × {len(df.columns)} 列　"
                         f"已执行清洗步骤 {len(c.log)} 个")

    def _set_status(self, text: str) -> None:
        self.status.config(text=text)

    # ---------- 文件 ----------

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="打开 CSV 文件",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            df = read_csv_any(path)
        except Exception as exc:
            messagebox.showerror("打开失败", f"无法读取文件：\n{exc}")
            return
        if df.empty:
            messagebox.showwarning("空文件", "该文件没有可读的数据。")
            return
        self.source_path = path
        self.cleaner = Cleaner(df)
        self._refresh_all()
        self.tabs.select(self.table)
        self._set_status(f"已打开：{path}（编码 {sniff_encoding(path)}），"
                         f"{len(df)} 行 × {len(df.columns)} 列，建议先「重新体检」")

    def export_csv(self) -> None:
        c = self._require_data()
        if c is None:
            return
        path = filedialog.asksaveasfilename(
            title="导出清洗后 CSV", defaultextension=".csv",
            initialfile="cleaned.csv", filetypes=[("CSV 文件", "*.csv")],
        )
        if not path:
            return
        c.df.to_csv(path, index=False, encoding="utf-8-sig")  # BOM 保证 Excel 打开不乱码
        messagebox.showinfo("导出成功", f"已保存到：\n{path}")

    def export_report(self) -> None:
        c = self._require_data()
        if c is None:
            return
        path = filedialog.asksaveasfilename(
            title="导出分析报告", defaultextension=".md",
            initialfile="report.md", filetypes=[("Markdown", "*.md")],
        )
        if not path:
            return
        inspection = inspect(c.df)
        export_report(c.df, inspection, c, path, source_name=self.source_path)
        messagebox.showinfo("导出成功", f"分析报告已保存到：\n{path}")

    # ---------- 清洗操作 ----------

    def _apply(self, method: str) -> None:
        """无参数的一键操作统一入口。"""
        c = self._require_data()
        if c is None:
            return
        try:
            getattr(c, method)()
        except Exception as exc:
            messagebox.showerror("操作失败", str(exc))
            return
        self._refresh_all()

    def _do_fill(self) -> None:
        c = self._require_data()
        if c is None or not self.col_combo.get():
            return
        raw = self.fill_strategy.get()
        strategy = raw.split("(")[0]
        value: str | float | None = None
        if strategy == "constant":
            text = self.fill_value.get().strip()
            if not text:
                messagebox.showwarning("缺少填充值", "固定值填充需要在输入框填写内容。")
                return
            try:
                value = float(text)
            except ValueError:
                value = text
        try:
            c.fill_missing(self.col_combo.get(), strategy, fill_value=value)
        except Exception as exc:
            messagebox.showerror("填充失败", str(exc))
            return
        self._refresh_all()

    def _do_convert(self) -> None:
        c = self._require_data()
        if c is None or not self.conv_col.get():
            return
        try:
            if self.conv_kind.get() == "数值":
                c.convert_numeric(self.conv_col.get())
            else:
                c.convert_datetime(self.conv_col.get())
        except Exception as exc:
            messagebox.showerror("转换失败", str(exc))
            return
        self._refresh_all()

    def _do_outliers(self) -> None:
        c = self._require_data()
        if c is None or not self.out_col.get():
            return
        action = "clip" if self.out_action.get().startswith("clip") else "remove"
        try:
            c.handle_outliers(self.out_col.get(), action=action)
        except Exception as exc:
            messagebox.showerror("操作失败", str(exc))
            return
        self._refresh_all()

    def _do_drop_column(self) -> None:
        c = self._require_data()
        if c is None or not self.rm_col.get():
            return
        c.drop_columns([self.rm_col.get()])
        self._refresh_all()

    def run_inspection(self) -> None:
        c = self._require_data()
        if c is None:
            return
        self._refresh_all()
        self.tabs.select(self.inspect_text)


def main() -> None:
    root = tk.Tk()
    try:
        from tkinter import font
        default = font.nametofont("TkDefaultFont")
        default.configure(size=10)
    except tk.TclError:  # 极端环境下字体子系统不可用也不影响启动
        pass
    App(root)
    root.mainloop()
