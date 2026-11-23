# -*- coding: utf-8 -*-
"""
============================================================================
脚本功能: 合成 2.5D 光源下的锂电池胶路图像数据集 (毕设演示用)
业务背景: 康视达 - 锂电封装打胶工序胶检项目
----------------------------------------------------------------------------
真实产线数据涉密, 本脚本通过物理成像近似模拟胶路图像, 解决"毕设无数据"痛点:
  1) 2.5D 条状光源 -> 背景周期性明暗条纹 + 胶条方向性高光
  2) 胶路圆凸截面 -> 中心亮、边缘暗的体积感剖面
  3) 4 类目标: OK / 断胶(broken_glue) / 缺胶(lack_glue) / 溢胶(overflow)
  4) 同时输出 YOLO 格式标注 (LabelImg 兼容), 可直接训练
============================================================================
运行: python scripts/generate_synthetic_data.py --per-class 80
============================================================================
"""
import os
import cv2
import argparse
import numpy as np

IMGSZ = 640
# 类别: 0=OK 1=断胶 2=缺胶 3=溢胶
CLASS_ID = {"OK": 0, "broken_glue": 1, "lack_glue": 2, "overflow": 3}


def make_background(rng, size=IMGSZ):
    """生成 2.5D 光源下的金属电池顶盖背景: 暗色基底 + 低频纹理 + 明暗条纹"""
    base = rng.integers(22, 48)
    bg = np.full((size, size, 3), base, dtype=np.float32)
    # 低频金属纹理
    noise = rng.normal(0, 7, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 18)
    bg += texture[..., None]
    # 2.5D 条状光源: 周期性明暗条纹
    phase = rng.uniform(0, 2 * np.pi)
    period = rng.uniform(55, 110)
    stripe = 11 * np.sin(np.arange(size) / period * 2 * np.pi + phase)
    bg += stripe[None, :, None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def path_center(rng, size=IMGSZ):
    """胶路中心线: 略带弧度的近水平曲线 y(x)"""
    y_base = rng.uniform(size * 0.38, size * 0.62)
    amp = rng.uniform(4, 26)
    freq = rng.uniform(120, 320)
    phase = rng.uniform(0, 2 * np.pi)
    x = np.arange(size)[None, :]
    yc = y_base + amp * np.sin(x / freq * 2 * np.pi + phase)
    return yc


def render_glue(img, wf, yc, light_dir, rng, width_base):
    """按宽度剖面 wf 渲染胶条 (向量化): 圆凸高光 + 2.5D方向性明暗"""
    h, w = img.shape[:2]
    y = np.arange(h)[:, None]
    # 有效区域: 宽度>0 且 位于胶条截面内
    half = (width_base * wf) / 2.0
    dy = (y - yc) / np.maximum(half, 1e-6)
    inside = (wf[None, :] > 0) & (np.abs(dy) <= 1)
    cross = np.sqrt(np.clip(1 - dy ** 2, 0, 1))       # 圆凸剖面
    # 2.5D 方向性: 光源侧更亮
    gx = np.linspace(0, 1, w)[None, :]
    light = 0.62 + 0.38 * (gx if light_dir > 0 else 1 - gx)
    val = 205 * cross * light + 24
    out = img.astype(np.float32)
    # 仅在 inside 区域替换像素 (val 广播到 3 个颜色通道)
    v = np.clip(out * 0.12 + val[..., None], 0, 255)
    out[inside] = v[inside]
    return out.astype(np.uint8)


def add_overflow_blob(img, rng, region, yc, width_base):
    """溢胶: 在胶路外侧叠加溢出的胶滴/胶瘤"""
    x0, x1 = region
    cx = int(rng.uniform(x0 + 6, x1 - 6))
    side = rng.choice([-1, 1])
    cy = int(yc[0, cx] + side * width_base * rng.uniform(0.7, 1.0))
    rx = int(rng.uniform(14, 34))
    ry = int(rng.uniform(10, 24))
    blob = np.zeros_like(img)
    cv2.ellipse(blob, (cx, cy), (rx, ry), 0, 0, 360, (215, 215, 215), -1)
    blob = cv2.GaussianBlur(blob, (0, 0), 4)
    out = img.astype(np.float32)
    m = blob[:, :, 0] > 12
    out[m] = np.clip(out[m] * 0.15 + blob[m] * 1.05, 0, 255)
    return out.astype(np.uint8), (cx - rx, cy - ry, cx + rx, cy + ry)


def finalize(img, rng):
    """后处理: 轻微模糊 + 高斯噪声 + 亮度波动, 逼近真实成像"""
    k = rng.choice([3, 3, 5])
    img = cv2.GaussianBlur(img, (k, k), 0)
    noise = rng.normal(0, 3.5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    beta = rng.uniform(-16, 16)
    if beta > 0:
        img = cv2.add(img, np.full_like(img, int(beta)))
    else:
        img = cv2.subtract(img, np.full_like(img, int(-beta)))
    return img


def yolo_line(cls_id, x0, y0, x1, y1, size=IMGSZ):
    """生成 YOLO 标注行: cls xc yc w h (全部归一化到 0~1)"""
    xc, yc = (x0 + x1) / 2 / size, (y0 + y1) / 2 / size
    bw, bh = (x1 - x0) / size, (y1 - y0) / size
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def generate_one(defect, idx, out_img, out_lbl):
    rng = np.random.default_rng(idx * 997 + CLASS_ID[defect] * 131 + 7)
    size = IMGSZ
    img = make_background(rng, size)
    yc = path_center(rng, size)
    width_base = rng.uniform(34, 50)
    light_dir = rng.choice([-1, 1])

    # 胶路起止 (留出左右边距)
    x0 = int(rng.uniform(28, 70))
    x1 = int(rng.uniform(size - 70, size - 28))
    wf = np.zeros(size, dtype=np.float32)
    wf[x0:x1] = 1.0

    bbox = None
    blob_box = None
    if defect == "OK":
        # 正常胶路: 全段均匀, 标注整条胶路
        pass
    elif defect == "broken_glue":
        # 断胶: 中段完全断开
        g0 = int(rng.uniform(x0 + (x1 - x0) * 0.30, x0 + (x1 - x0) * 0.48))
        g1 = min(x1 - 10, g0 + int(rng.uniform(48, 92)))
        wf[g0:g1] = 0.0
        yy = yc[0, g0:g1]
        bbox = (g0 - 6, int(yy.min() - width_base), g1 + 6,
                int(yy.max() + width_base))
    elif defect == "lack_glue":
        # 缺胶: 中段胶宽缩窄至 30%~45%
        g0 = int(rng.uniform(x0 + (x1 - x0) * 0.28, x0 + (x1 - x0) * 0.55))
        g1 = min(x1 - 10, g0 + int(rng.uniform(70, 130)))
        wf[g0:g1] = rng.uniform(0.30, 0.45)
        yy = yc[0, g0:g1]
        bbox = (g0 - 6, int(yy.min() - width_base), g1 + 6,
                int(yy.max() + width_base))
    elif defect == "overflow":
        # 溢胶: 中段胶宽放大 + 外侧胶瘤
        g0 = int(rng.uniform(x0 + (x1 - x0) * 0.30, x0 + (x1 - x0) * 0.55))
        g1 = min(x1 - 15, g0 + int(rng.uniform(60, 110)))
        wf[g0:g1] = rng.uniform(1.9, 2.5)
        img = render_glue(img, wf, yc, light_dir, rng, width_base)
        img, blob_box = add_overflow_blob(img, rng, (g0, g1), yc, width_base)
        yy = yc[0, g0:g1]
        bx0, bx1 = g0 - 8, g1 + 8
        by0 = min(int(yy.min() - width_base * 2.5), blob_box[1] if blob_box else 9999)
        by1 = max(int(yy.max() + width_base * 2.5), blob_box[3] if blob_box else 0)
        bbox = (bx0, by0, bx1, by1)

    if defect != "overflow":
        img = render_glue(img, wf, yc, light_dir, rng, width_base)

    img = finalize(img, rng)

    # 生成标注
    if defect == "OK":
        yy = yc[0, x0:x1]
        box = (x0, int(yy.min() - width_base * 0.7),
               x1, int(yy.max() + width_base * 0.7))
        line = yolo_line(CLASS_ID["OK"], *box, size=size)
    else:
        line = yolo_line(CLASS_ID[defect], *bbox, size=size)

    name = f"{defect}_{idx:04d}"
    cv2.imwrite(os.path.join(out_img, name + ".jpg"), img)
    with open(os.path.join(out_lbl, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(line + "\n")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=80, help="每类生成图像数量")
    ap.add_argument("--out", type=str, default="data/raw", help="输出根目录")
    args = ap.parse_args()

    root = os.path.abspath(args.out)
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    total = 0
    for defect in CLASS_ID:
        for i in range(args.per_class):
            generate_one(defect, i, img_dir, lbl_dir)
            total += 1
        print(f"[OK] {defect:12s} 生成 {args.per_class} 张")
    print(f"\n完成: 共 {total} 张图像 + YOLO 标注 -> {root}")


if __name__ == "__main__":
    main()
