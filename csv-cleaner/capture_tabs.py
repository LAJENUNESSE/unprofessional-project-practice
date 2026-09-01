"""演示素材采集：驱动真实 GUI 截取体检/图表/日志页签。"""
import sys
import time
import pathlib
import tkinter as tk

sys.path.insert(0, "src")
from PIL import ImageGrab

from csv_cleaner.app import App
from csv_cleaner.core.loader import read_csv_any
from csv_cleaner.core.cleaner import Cleaner

OUT = pathlib.Path("docs/screenshots")
OUT.mkdir(exist_ok=True)

root = tk.Tk()
root.attributes("-topmost", True)
app = App(root)
root.update()

# 自动准备"清洗到一半"的真实状态
app.source_path = "samples/messy_sales.csv"
app.cleaner = Cleaner(read_csv_any(app.source_path))
app.cleaner.drop_duplicates()
app.cleaner.strip_whitespace()
app.cleaner.convert_numeric("单价")
app.cleaner.fill_missing("客户年龄", "median")
app.cleaner.handle_outliers("数量", action="clip")
app._refresh_all()
root.update()


def snap(name: str) -> None:
    root.update_idletasks()
    time.sleep(0.4)
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(OUT / name)
    print("已截图", name)


def shoot_inspection() -> None:
    app.tabs.select(app.inspect_text)
    root.after(300, lambda: (snap("tab_inspection.png"), root.after(100, shoot_charts)))


def shoot_charts() -> None:
    app.charts.type_var.set("柱状图(频次TOP10)")
    app.charts.col_var.set("城市")
    app.charts.draw()
    app.tabs.select(app.charts)
    root.after(600, lambda: (snap("tab_charts.png"), root.after(100, shoot_log)))


def shoot_log() -> None:
    app.tabs.select(app.log_text)
    root.after(300, lambda: (snap("tab_log.png"), root.destroy()))


root.after(800, shoot_inspection)
root.mainloop()
print("全部素材采集完成 ->", OUT)
