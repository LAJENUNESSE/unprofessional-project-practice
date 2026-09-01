# -*- coding: utf-8 -*-
"""生成 PascalVOC 格式标注 XML（与 LabelImg 保存格式一致）。

框坐标为人工看图标注结果，格式：
    (xmin, ymin, xmax, ymax, truncated, difficult)
"""
import os
from lxml import etree

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")

# 每张图的目标框（人工标定）
ANNOTATIONS = {
    "bus.jpg": [
        ("bus",    (32,  230, 810, 728), 1, 0),  # 车身右侧出画
        ("person", (48,  390, 215, 905), 0, 0),  # 左侧米色外套
        ("person", (218, 400, 352, 858), 0, 0),  # 中间黑色外套
        ("person", (655, 388, 810, 888), 1, 0),  # 右侧背对镜头
    ],
    "zidane.jpg": [
        ("person", (160, 205,  700, 720), 1, 0),  # 左：齐达内
        ("person", (752,  45, 1165, 720), 1, 0),  # 右：安切洛蒂
    ],
    "dog.jpg": [
        ("dog", (138, 228, 325, 558), 0, 0),  # 坐姿黑狗（含尾巴）
    ],
    "person.jpg": [
        ("person", (195,  95, 285, 378), 0, 0),  # 蓝夹克站立者
        ("dog",    ( 58, 262, 205, 375), 0, 0),  # 左侧嗅地面的狗
    ],
}

# 图片真实尺寸 (width, height, depth)
SIZES = {
    "bus.jpg":    (810, 1080, 3),
    "zidane.jpg": (1280,  720, 3),
    "dog.jpg":    (768,  576, 3),
    "person.jpg": (640,  424, 3),
}


def sub(parent, tag, text):
    e = etree.SubElement(parent, tag)
    e.text = str(text)
    return e


def build_xml(folder, filename, size, objects):
    root = etree.Element("annotation")
    sub(root, "folder", folder)
    sub(root, "filename", filename)
    sub(root, "path", os.path.join(TEST_DIR, filename))
    src = etree.SubElement(root, "source")
    sub(src, "database", "Unknown")
    sz = etree.SubElement(root, "size")
    sub(sz, "width", size[0])
    sub(sz, "height", size[1])
    sub(sz, "depth", size[2])
    sub(root, "segmented", 0)
    for name, (xmin, ymin, xmax, ymax), truncated, difficult in objects:
        obj = etree.SubElement(root, "object")
        sub(obj, "name", name)
        sub(obj, "pose", "Unspecified")
        sub(obj, "truncated", truncated)
        sub(obj, "difficult", difficult)
        box = etree.SubElement(obj, "bndbox")
        sub(box, "xmin", xmin)
        sub(box, "ymin", ymin)
        sub(box, "xmax", xmax)
        sub(box, "ymax", ymax)
    return root


def main():
    for filename, objects in ANNOTATIONS.items():
        xml_path = os.path.join(TEST_DIR, os.path.splitext(filename)[0] + ".xml")
        tree = etree.ElementTree(
            build_xml("test", filename, SIZES[filename], objects)
        )
        tree.write(
            xml_path,
            encoding="us-ascii",
            xml_declaration=True,
            pretty_print=True,
        )
        print(f"已生成 {xml_path}")


if __name__ == "__main__":
    main()
