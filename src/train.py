# -*- coding: utf-8 -*-
"""
============================================================================
模块功能: YOLOv8 模型训练与超参调优
  - 标准训练: 加载 config.yaml 超参, 训练并保存最佳权重
  - 调参对比: 对比不同 conf / IoU 对检测效果的影响 (毕设实验章节)
业务说明: 康视达锂电胶检项目高频迭代, 需针对小目标缺陷做超参适配
============================================================================
命令行:
  python -m src.train                      # 标准训练
  python -m src.train --epochs 50 --batch 8 # 自定义
  python -m src.train --compare            # 调参对比实验(需已有权重)
============================================================================
"""
import os
import argparse
from .utils import load_config, PROJECT_ROOT, ensure_dir, get_train_data_yaml


def train(cfg=None, epochs=None, batch=None, imgsz=None, device=None, model=None):
    """标准 YOLOv8 训练流程"""
    from ultralytics import YOLO
    if cfg is None:
        cfg = load_config()
    t = cfg.get("train", {})

    epochs = epochs or t.get("epochs", 100)
    batch = batch or t.get("batch", 16)
    imgsz = imgsz or t.get("imgsz", 640)
    device = device if device is not None else t.get("device", 0)
    model_name = model or t.get("model", "yolov8n.pt")

    data_yaml = get_train_data_yaml()
    if not os.path.exists(data_yaml):
        raise FileNotFoundError("数据集配置不存在, 请先生成数据并划分: 见 scripts/")

    ensure_dir(os.path.join(PROJECT_ROOT, "weights"))
    print("=" * 60)
    print(f"YOLOv8 训练启动 | 模型={model_name} epochs={epochs} batch={batch} imgsz={imgsz} device={device}")
    print(f"数据集: {data_yaml}")
    print("=" * 60)

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        lr0=t.get("lr0", 0.01),
        iou=t.get("iou", 0.7),
        patience=t.get("patience", 20),
        seed=t.get("seed", 42),
        project=os.path.join(PROJECT_ROOT, "weights", "runs"),
        name="glue_yolov8",
        exist_ok=True,
        verbose=True,
    )
    # 复制最佳权重到 weights/best.pt
    import shutil
    best = os.path.join(PROJECT_ROOT, "weights", "runs", "glue_yolov8", "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy2(best, os.path.join(PROJECT_ROOT, "weights", "best.pt"))
        print("[OK] 最佳权重已复制到 weights/best.pt")
    return results


def compare_hyperparams(weight_path=None, conf_values=None, iou_values=None):
    """
    调参对比实验: 在测试集上评估不同置信度阈值与 IoU 阈值的效果。
    输出 mAP@0.5 表格, 支撑毕设"超参对比分析"章节。
    """
    from ultralytics import YOLO
    import pandas as pd
    from .inference import resolve_weight

    weight = resolve_weight(weight_path)
    model = YOLO(weight)
    data_yaml = get_train_data_yaml()
    conf_values = conf_values or [0.15, 0.25, 0.35, 0.45]
    iou_values = iou_values or [0.35, 0.45, 0.55, 0.65]

    rows = []
    for conf in conf_values:
        for iou in iou_values:
            metrics = model.val(data=data_yaml, conf=conf, iou=iou, verbose=False)
            rows.append({
                "conf": conf,
                "iou": iou,
                "mAP50": round(float(metrics.box.map50), 4),
                "mAP50-95": round(float(metrics.box.map), 4),
                "precision": round(float(metrics.box.mp), 4),
                "recall": round(float(metrics.box.mr), 4),
            })
            print(f"  conf={conf} iou={iou} -> mAP50={rows[-1]['mAP50']}")
    df = pd.DataFrame(rows)
    out = os.path.join(PROJECT_ROOT, "outputs", "reports", "hyperparam_compare.xlsx")
    ensure_dir(os.path.dirname(out))
    df.to_excel(out, index=False)
    print(f"[OK] 调参对比结果已导出: {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--compare", action="store_true", help="运行调参对比实验")
    ap.add_argument("--weight", type=str, default=None)
    args = ap.parse_args()
    if args.compare:
        compare_hyperparams(args.weight)
    else:
        train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
              device=args.device, model=args.model)


if __name__ == "__main__":
    main()
