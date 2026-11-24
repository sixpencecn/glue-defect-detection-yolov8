# -*- coding: utf-8 -*-
"""
============================================================================
模块功能: 推理检测 (核心检测入口)
  输入图片 -> YOLOv8 推理 -> 缺陷框/类别/置信度 -> 可视化结果
  - 支持单图 / 批量图片 / numpy 数组
  - 自动判定 OK/NG (只要检出任一缺陷类即判 NG)
  - 检测前可选预处理, 降低 2.5D 反光干扰
============================================================================
命令行:
  python -m src.inference --image path/to/img.jpg
  python -m src.inference --dir data/dataset/images/test
============================================================================
"""
import os
import cv2
import time
import argparse
import numpy as np
from .utils import (load_config, PROJECT_ROOT, CLASS_NAMES, CLASS_ZH,
                    CLASS_COLORS, draw_chinese_label)


def resolve_weight(weight_path=None):
    """确定权重路径: 优先指定路径 -> weights/best.pt -> 预训练模型(演示)"""
    if weight_path and os.path.exists(weight_path):
        return weight_path
    best = os.path.join(PROJECT_ROOT, "weights", "best.pt")
    if os.path.exists(best):
        return best
    return "yolov8n.pt"  # 未训练时回退到预训练权重(演示流程, 类为COCO, 仅跑通pipeline)


class GlueDetector:
    def __init__(self, weight_path=None, config=None, use_preprocess=True):
        from ultralytics import YOLO
        if config is None:
            config = load_config()
        self.config = config
        inf = config.get("inference", {})
        self.conf = float(inf.get("conf", 0.55))
        self.iou = float(inf.get("iou", 0.45))
        self.max_det = int(inf.get("max_det", 100))
        # 每类最多保留的框数; 0=不限制. 本数据集每图仅1处缺陷, 设为1可抑制同类重复框误标
        self.max_boxes_per_class = int(inf.get("max_boxes_per_class", 1))
        self.weight = resolve_weight(weight_path)
        self.model = YOLO(self.weight)
        self.use_preprocess = use_preprocess
        if use_preprocess:
            from .preprocessing import GluePreprocessor
            self.preprocessor = GluePreprocessor(config)

    def detect(self, image):
        """
        检测单张图片。
        参数 image: 路径(str) 或 BGR ndarray
        返回 dict(ok, defects, annotated, elapsed_ms, ...)
        """
        t0 = time.time()
        if isinstance(image, str):
            original = cv2.imread(image)
            if original is None:
                raise FileNotFoundError(f"无法读取图像: {image}")
            src_name = os.path.basename(image)
        else:
            original = image.copy()
            src_name = "camera_frame"

        # 预处理 (不改变送入模型的图尺寸, YOLO内部会resize)
        if self.use_preprocess:
            input_img = self.preprocessor.preprocess(original)
        else:
            input_img = original

        results = self.model.predict(
            source=input_img, conf=self.conf, iou=self.iou,
            max_det=self.max_det, verbose=False,
        )
        res = results[0]
        defects = []
        annotated = original.copy()
        if res.boxes is not None and len(res.boxes) > 0:
            # 收集全部候选框
            candidates = []
            for box in res.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                candidates.append({"cls_id": cls_id, "conf": conf, "bbox": [x1, y1, x2, y2]})

            # 每类限框: 仅保留置信度最高的前 max_boxes_per_class 个, 抑制同类重复误标
            if self.max_boxes_per_class > 0:
                by_cls = {}
                for c in candidates:
                    by_cls.setdefault(c["cls_id"], []).append(c)
                candidates = []
                for items in by_cls.values():
                    items.sort(key=lambda x: -x["conf"])
                    candidates.extend(items[:self.max_boxes_per_class])

            for c in candidates:
                cls_id = c["cls_id"]
                conf = c["conf"]
                x1, y1, x2, y2 = c["bbox"]
                name = CLASS_NAMES.get(cls_id, f"class{cls_id}")
                zh = CLASS_ZH.get(cls_id, name)
                color = CLASS_COLORS.get(cls_id, (0, 255, 0))
                defects.append({
                    "cls_id": cls_id, "class": name, "name_zh": zh,
                    "conf": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                })
                # 绘制检测框与中文标签(PIL渲染, 避免cv2中文乱码)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{zh} {conf:.2f}"
                draw_chinese_label(annotated, label, (x1, y1), color, font_size=20)

        # OK/NG 判定: 检出任一缺陷类(非OK)即为 NG
        # 说明: 仅检出OK框 -> OK; 检出缺陷框 -> NG; 未检出任何目标 -> NG(保守策略, 触发人工复检)
        ng_defects = [d for d in defects if d["cls_id"] != 0]
        ok = len(ng_defects) == 0 and len(defects) > 0
        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "source": src_name,
            "ok": ok,
            "status": "OK" if ok else "NG",
            "defects": defects,
            "defect_count": len(ng_defects),   # 仅统计缺陷类(不含OK), 与统计报表口径一致
            "annotated": annotated,
            "elapsed_ms": elapsed_ms,
            "conf": self.conf,
            "iou": self.iou,
            "weight": os.path.basename(str(self.weight)),
        }

    def detect_dir(self, img_dir, save_annotated=True, out_dir=None):
        """批量检测目录下所有图片"""
        files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        out_dir = out_dir or os.path.join(PROJECT_ROOT, "outputs", "defect_shots")
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for f in sorted(files):
            r = self.detect(os.path.join(img_dir, f))
            if save_annotated:
                cv2.imwrite(os.path.join(out_dir, "det_" + f), r["annotated"])
            results.append(r)
            print(f"[检测] {f:24s} -> {r['status']:3s} 缺陷数={r['defect_count']} 耗时={r['elapsed_ms']}ms")
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--weight", type=str, default=None)
    ap.add_argument("--no-preprocess", action="store_true")
    args = ap.parse_args()
    det = GlueDetector(args.weight, use_preprocess=not args.no_preprocess)
    if args.image:
        r = det.detect(args.image)
        print(f"\n图片: {r['source']}  判定: {r['status']}  耗时: {r['elapsed_ms']}ms")
        for d in r["defects"]:
            print(f"  - {d['name_zh']}({d['class']}) 置信度={d['conf']:.2f} 框={d['bbox']}")
    elif args.dir:
        det.detect_dir(args.dir)
    else:
        ap.error("请指定 --image 或 --dir")


if __name__ == "__main__":
    main()
