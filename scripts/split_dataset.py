# -*- coding: utf-8 -*-
"""
============================================================================
脚本功能: 划分数据集为 训练集/验证集/测试集 (7:2:1), 支持真实数据与合成数据
说明: 输入 data/raw/{images,labels}, 输出 data/dataset/{images,labels}/{train,val,test}
      划分基于文件名随机打乱, 保证同类别按比例分布
============================================================================
运行: python scripts/split_dataset.py --seed 42
============================================================================
"""
import os
import shutil
import argparse
import random
from collections import defaultdict


def split_dataset(raw_dir, out_dir, train_r, val_r, test_r, seed, move=False):
    raw_img = os.path.join(raw_dir, "images")
    raw_lbl = os.path.join(raw_dir, "labels")
    names = [f for f in os.listdir(raw_img) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not names:
        raise RuntimeError(f"未在 {raw_img} 找到图像, 请先运行 generate_synthetic_data.py 或放入真实数据")

    # 按类别前缀分组, 保证每类按比例划分
    groups = defaultdict(list)
    for n in names:
        cls = n.split("_")[0]
        groups[cls].append(n)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls, files in groups.items():
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_tr = max(1, int(n * train_r))
        n_va = max(1, int(n * val_r))
        splits["train"] += files[:n_tr]
        splits["val"] += files[n_tr:n_tr + n_va]
        splits["test"] += files[n_tr + n_va:]
        print(f"[INFO] {cls:12s} 共{n:3d}  train={n_tr} val={n_va} test={n-n_tr-n_va}")

    # 创建目录 (先清空旧划分, 避免重复运行时数据累加)
    for sp in splits:
        for sub in ("images", "labels"):
            dst = os.path.join(out_dir, sub, sp)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.makedirs(dst, exist_ok=True)

    fn = shutil.move if move else shutil.copy2
    for sp, files in splits.items():
        for img in files:
            stem = os.path.splitext(img)[0]
            lbl = stem + ".txt"
            src_img = os.path.join(raw_img, img)
            dst_img = os.path.join(out_dir, "images", sp, img)
            fn(src_img, dst_img)
            src_lbl = os.path.join(raw_lbl, lbl)
            if os.path.exists(src_lbl):
                fn(src_lbl, os.path.join(out_dir, "labels", sp, lbl))
        print(f"[OK] {sp}: {len(files)} 张")

    print(f"\n划分完成 -> {os.path.abspath(out_dir)}")
    print("提示: 数据配置见 data/dataset/data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--train-r", type=float, default=0.7)
    ap.add_argument("--val-r", type=float, default=0.2)
    ap.add_argument("--test-r", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="移动而非复制(默认复制, 保留原始数据)")
    args = ap.parse_args()
    split_dataset(args.raw, args.out, args.train_r, args.val_r,
                  args.test_r, args.seed, args.move)


if __name__ == "__main__":
    main()
