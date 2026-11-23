# -*- coding: utf-8 -*-
"""工具函数: 配置加载、类别定义、通用辅助"""
import os
import yaml

# 类别定义 (与 data.yaml / config.yaml 保持一致)
CLASS_NAMES = {0: "OK", 1: "broken_glue", 2: "lack_glue", 3: "overflow"}
CLASS_ZH = {0: "正常胶路", 1: "断胶", 2: "缺胶", 3: "溢胶"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 0, 255), 2: (0, 165, 255), 3: (255, 0, 0)}  # BGR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    """加载全局配置 config.yaml"""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_train_data_yaml():
    """
    生成带绝对路径的数据配置 data_abs.yaml 并返回其路径。
    避免 ultralytics 对 data.yaml 中相对 path 的解析差异(不同版本/cwd),
    使训练可从任意工作目录启动。
    """
    src = os.path.join(PROJECT_ROOT, "data", "dataset", "data.yaml")
    dst = os.path.join(PROJECT_ROOT, "data", "dataset", "data_abs.yaml")
    with open(src, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = os.path.join(PROJECT_ROOT, "data", "dataset").replace("\\", "/")
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return dst


def now_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def draw_chinese_label(image_bgr, text, origin, bg_color, font_size=20):
    """
    用 PIL 在图像上绘制中文标签(cv2.putText 不支持中文会显示??)。
    image_bgr: BGR ndarray, 原地修改并返回
    origin: (x, y) 标签左上角
    bg_color: BGR 元组(与检测框颜色一致)
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    x, y = int(origin[0]), int(origin[1])
    if y - font_size - 4 < 0:
        y = font_size + 4
    # 载入中文字体(按优先级), 找不到则退回默认字体
    font = None
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
               "C:/Windows/Fonts/simsun.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil)
    ty = y - font_size - 4
    try:
        box = d.textbbox((x, ty), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
    except Exception:
        tw, th = len(text) * font_size, font_size
    # 背景框比文字略宽, 避免文字溢出
    d.rectangle([x - 2, ty - 2, x + tw + 8, ty + th + 4],
                fill=(bg_color[2], bg_color[1], bg_color[0]))
    d.text((x + 3, ty), text, fill=(255, 255, 255), font=font)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    image_bgr[:] = out[:]
    return image_bgr
