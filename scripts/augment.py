# -*- coding: utf-8 -*-
"""
============================================================================
脚本功能: 数据增强 - 亮度扰动 / 高斯噪声 / 翻转 / HSV色域扰动
业务意义: 模拟 2.5D 光源光照波动、反光与产线相机噪声, 提升模型鲁棒性
说明: 增强在合成数据划分后额外扩充训练集, 同步变换 YOLO 标注框
============================================================================
运行: python scripts/augment.py --times 1
============================================================================
"""
import os
import cv2
import argparse
import numpy as np


def augment_brightness(img, rng):
    factor = rng.uniform(0.7, 1.3)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def augment_noise(img, rng):
    noise = rng.normal(0, rng.uniform(5, 14), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment_hsv(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.8, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def flip(img, labels, mode, size):
    """mode: 'h' 水平 / 'v' 垂直, 同步更新 YOLO 标注"""
    if mode == "h":
        out = cv2.flip(img, 1)
        new = []
        for l in labels:
            c, xc, yc, w, h = l
            new.append((c, 1 - xc, yc, w, h))
        return out, new
    else:
        out = cv2.flip(img, 0)
        new = []
        for l in labels:
            c, xc, yc, w, h = l
            new.append((c, xc, 1 - yc, w, h))
        return out, new


def parse_labels(path):
    labels = []
    if not os.path.exists(path):
        return labels
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                labels.append(tuple([int(p[0])] + [float(v) for v in p[1:5]]))
    return labels


def save_labels(path, labels):
    with open(path, "w", encoding="utf-8") as f:
        for l in labels:
            f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")


def augment_dataset(dataset_dir, times, seed):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(dataset_dir, "images", "train")
    lbl_dir = os.path.join(dataset_dir, "labels", "train")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
    count = 0
    for f in files:
        stem = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(img_dir, f))
        labels = parse_labels(os.path.join(lbl_dir, stem + ".txt"))
        h, w = img.shape[:2]
        for t in range(times):
            a = img.copy()
            lab = list(labels)
            # 随机组合增强
            if rng.random() < 0.6:
                a = augment_brightness(a, rng)
            if rng.random() < 0.5:
                a = augment_noise(a, rng)
            if rng.random() < 0.5:
                a = augment_hsv(a, rng)
            if rng.random() < 0.5:
                a, lab = flip(a, lab, rng.choice(["h", "v"]), (w, h))
            name = f"{stem}_aug{t}"
            cv2.imwrite(os.path.join(img_dir, name + ".jpg"), a)
            save_labels(os.path.join(lbl_dir, name + ".txt"), lab)
            count += 1
    print(f"[OK] 数据增强完成, 新增 {count} 张训练图像 -> {img_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--times", type=int, default=1, help="每张图增强倍数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    augment_dataset(args.dataset, args.times, args.seed)


if __name__ == "__main__":
    main()
