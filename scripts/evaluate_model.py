# -*- coding: utf-8 -*-
"""
============================================================================
脚本功能: 评估当前权重在测试集上的分类表现 (混淆矩阵 + 各类召回)
用途: 诊断"溢胶被误判为正常胶路"等缺陷, 定位是偶发还是系统性问题
运行: python scripts/evaluate_model.py --weight weights/best.pt --imgsz 320
============================================================================
"""
import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

CLASS_ID = {"OK": 0, "broken_glue": 1, "lack_glue": 2, "overflow": 3}
ID2NAME = {v: k for k, v in CLASS_ID.items()}


def parse_label(txt_path):
    """读取 YOLO 标注第一行, 返回类别 id"""
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        return None
    return int(line.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="weights/best.pt")
    ap.add_argument("--img-dir", default="data/dataset/images/test")
    ap.add_argument("--lbl-dir", default="data/dataset/labels/test")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weight)
    files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith((".jpg", ".png"))])

    # confusion[true][pred]
    conf = np.zeros((4, 4), dtype=int)
    miss = []
    per_image = []

    for fn in files:
        name = os.path.splitext(fn)[0]
        true_id = parse_label(os.path.join(args.lbl_dir, name + ".txt"))
        if true_id is None:
            continue
        img = cv2.imread(os.path.join(args.img_dir, fn))
        results = model.predict(source=img, imgsz=args.imgsz, conf=args.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            pred_id = -1  # 漏检
        else:
            # top1 = 置信度最高的框
            top = max(range(len(boxes)), key=lambda i: float(boxes.conf[i]))
            pred_id = int(boxes.cls[top])

        per_image.append((fn, true_id, pred_id))
        if pred_id == -1:
            miss.append(fn)
        else:
            conf[true_id][pred_id] += 1

    print("\n=== 混淆矩阵 (行=真实, 列=预测) ===")
    header = "真实\\预测 " + " ".join(f"{ID2NAME[i][:7]:>8s}" for i in range(4)) + "  漏检"
    print(header)
    for i in range(4):
        row = " ".join(f"{conf[i][j]:8d}" for j in range(4))
        n_miss = sum(1 for _, t, p in per_image if t == i and p == -1)
        print(f"{ID2NAME[i][:8]:>9s} {row}  {n_miss:4d}")

    print("\n=== 各类召回率 (检出为对应类的比例) ===")
    for i in range(4):
        total = sum(1 for _, t, _ in per_image if t == i)
        hit = conf[i][i]
        print(f"  {ID2NAME[i]:12s} {hit}/{total} = {hit/total*100 if total else 0:.0f}%")

    print("\n=== 溢胶(overflow)逐张诊断 ===")
    for fn, t, p in per_image:
        if t == 3:
            pname = ID2NAME.get(p, "漏检")
            flag = "OK" if p == 3 else "<<< 误判"
            print(f"  {fn:24s} 预测={pname:12s} {flag}")

    print(f"\n漏检总数: {len(miss)}")


if __name__ == "__main__":
    main()
