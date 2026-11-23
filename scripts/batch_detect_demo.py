# -*- coding: utf-8 -*-
"""
============================================================================
脚本功能: 批量推理 Demo - 对测试集检测并自动生成 Excel 报表 + 缺陷截图
对应需求: 模块5(推理检测) + 模块6(统计导出) 的命令行集成演示
运行: python scripts/batch_detect_demo.py
============================================================================
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import GlueDetector
from src.statistics import DetectionRecorder
from src.utils import PROJECT_ROOT


def main():
    test_dir = os.path.join(PROJECT_ROOT, "data", "dataset", "images", "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError("测试集不存在, 请先运行数据生成与划分脚本")

    print("加载模型...")
    detector = GlueDetector()
    print(f"使用权重: {detector.weight}")
    recorder = DetectionRecorder()

    print(f"\n开始批量检测: {test_dir}\n" + "-" * 60)
    # save_annotated=False: 避免把OK图也存入缺陷截图目录, 仅由 recorder 保存NG截图
    results = detector.detect_dir(test_dir, save_annotated=False)
    for r in results:
        recorder.record(r)

    print("-" * 60)
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"检测完成: 共 {len(results)} 张, OK={ok}, NG={len(results)-ok}")
    excel = recorder.export_excel()
    print(f"报表: {excel}")
    print(f"缺陷截图目录: {recorder.shot_dir}")


if __name__ == "__main__":
    main()
