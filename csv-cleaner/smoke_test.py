"""PyInstaller 冒烟测试：验证 3.14 环境下 tkinter/pandas/matplotlib 能被正确打包并运行。"""
import sys

import matplotlib
import pandas as pd
import tkinter

print("SMOKE_OK", sys.version_info[:3], "pandas", pd.__version__,
      "matplotlib", matplotlib.__version__, "tk", tkinter.TkVersion)
