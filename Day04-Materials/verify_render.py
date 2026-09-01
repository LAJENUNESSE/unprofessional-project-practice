# -*- coding: utf-8 -*-
"""离屏渲染验证：等启动异步任务完成后再打开 test 目录，逐张截图。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "labelImg-master"))
sys.argv = ["labelImg.py"]

import labelImg
from PyQt5.QtCore import QTimer

NAMES = ["bus", "dog", "person", "zidane"]
app, win = labelImg.get_main_app(sys.argv)
win.show()


def start():
    # 与 GUI"打开目录"等价：先设保存目录，import 后 load_file 会自动加载同目录 XML
    win.default_save_dir = os.path.join(HERE, "test")
    win.import_dir_images(os.path.join(HERE, "test"))
    print("scanned:", win.last_open_dir, "| files:", len(win.m_img_list))
    QTimer.singleShot(400, lambda: grab(0))


def grab(i):
    if i >= len(NAMES):
        app.quit()
        return
    pix = win.grab()
    pix.save(os.path.join(HERE, f"verify_{NAMES[i]}.png"))
    cur = os.path.basename(win.file_path or "?")
    print(f"verify_{NAMES[i]}.png | file={cur} | shapes={len(win.canvas.shapes)}")
    win.open_next_image()
    QTimer.singleShot(400, lambda: grab(i + 1))


QTimer.singleShot(1500, start)
sys.exit(app.exec_())
