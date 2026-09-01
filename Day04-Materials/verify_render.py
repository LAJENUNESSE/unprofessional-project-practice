# -*- coding: utf-8 -*-
"""离屏渲染验证：加载 test 目录标注并逐张截图，避免原生文件对话框。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "labelImg-master"))
sys.argv = ["labelImg.py"]

import labelImg
from PyQt5.QtCore import QTimer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
app, win = labelImg.get_main_app(sys.argv)
win.open_dir_dialog(dir_path=os.path.join(OUT_DIR, "test"), silent=True)
win.show()

names = ["bus", "zidane", "dog", "person"]


def grab(i):
    if i >= len(names):
        app.quit()
        return
    pix = win.grab()
    pix.save(os.path.join(OUT_DIR, f"verify_{names[i]}.png"))
    print(f"已截图 verify_{names[i]}.png, shapes={len(win.canvas.shapes)}")
    win.open_next_image()
    QTimer.singleShot(400, lambda: grab(i + 1))


QTimer.singleShot(600, lambda: grab(0))
sys.exit(app.exec_())
