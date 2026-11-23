# -*- coding: utf-8 -*-
"""
============================================================================
模块功能: 数据统计与导出
  - 自动记录每次检测: OK/NG、缺陷类型、置信度、检测时间、耗时
  - 历史记录持久化 (JSON)
  - 导出 Excel 报表 (多工作表: 明细 + 汇总)
  - 保存缺陷截图
业务意义: 满足产线质量追溯与日报需求, 对标康视达实际交付的报表功能
============================================================================
"""
import os
import json
import cv2
from datetime import datetime
from .utils import load_config, PROJECT_ROOT, ensure_dir, now_str, CLASS_ZH


class DetectionRecorder:
    def __init__(self, config=None):
        if config is None:
            config = load_config()
        rp = config.get("report", {})
        self.history_path = os.path.join(PROJECT_ROOT, rp.get("history_db", "outputs/history.json"))
        self.excel_path = os.path.join(PROJECT_ROOT, rp.get("excel_path", "outputs/reports/detect_report.xlsx"))
        self.shot_dir = os.path.join(PROJECT_ROOT, rp.get("defect_shot_dir", "outputs/defect_shots"))
        ensure_dir(os.path.dirname(self.history_path))
        ensure_dir(os.path.dirname(self.excel_path))
        ensure_dir(self.shot_dir)
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_history(self):
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def record(self, result, save_shot=True):
        """记录一次检测结果 (result 来自 GlueDetector.detect)"""
        seq = len(self.history) + 1
        defect_types = [d["name_zh"] for d in result["defects"] if d["cls_id"] != 0]
        item = {
            "seq": seq,
            "time": now_str(),
            "source": result["source"],
            "status": result["status"],
            "defect_types": "、".join(defect_types) if defect_types else "-",
            "defect_count": result.get("defect_count",
                               len([d for d in result["defects"] if d["cls_id"] != 0])),
            "max_conf": max([d["conf"] for d in result["defects"]], default=0.0),
            "elapsed_ms": result["elapsed_ms"],
            "conf_thres": result.get("conf"),
            "iou_thres": result.get("iou"),
            "weight": result.get("weight"),
        }
        # 保存缺陷截图
        if save_shot and result["status"] == "NG":
            shot_name = f"NG_{seq:04d}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(os.path.join(self.shot_dir, shot_name), result["annotated"])
            item["shot"] = shot_name
        self.history.append(item)
        self._save_history()
        return item

    def export_excel(self):
        """导出 Excel 报表: Sheet1 检测明细, Sheet2 统计汇总"""
        import pandas as pd
        if not self.history:
            raise RuntimeError("暂无检测记录, 无法导出报表")
        df = pd.DataFrame(self.history)
        df = df.rename(columns={
            "seq": "序号", "time": "检测时间", "source": "图片/来源",
            "status": "判定", "defect_types": "缺陷类型", "defect_count": "缺陷数",
            "max_conf": "最高置信度", "elapsed_ms": "耗时(ms)",
            "conf_thres": "置信度阈值", "iou_thres": "IoU阈值",
            "weight": "模型权重", "shot": "缺陷截图",
        })
        # 汇总
        total = len(self.history)
        ng = sum(1 for h in self.history if h["status"] == "NG")
        ok = total - ng
        from collections import Counter
        type_counter = Counter()
        for h in self.history:
            if h["defect_types"] and h["defect_types"] != "-":
                for t in h["defect_types"].split("、"):
                    type_counter[t] += 1
        summary = [{"指标": "检测总数", "数值": total},
                   {"指标": "OK数量", "数值": ok},
                   {"指标": "NG数量", "数值": ng},
                   {"指标": "不良率(%)", "数值": round(ng / total * 100, 2) if total else 0},
                   {"指标": "平均耗时(ms)", "数值": round(sum(h["elapsed_ms"] for h in self.history) / total, 1) if total else 0}]
        for t, c in type_counter.items():
            summary.append({"指标": f"缺陷-{t}", "数值": c})
        df_sum = pd.DataFrame(summary)

        ensure_dir(os.path.dirname(self.excel_path))
        with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="检测明细", index=False)
            df_sum.to_excel(writer, sheet_name="统计汇总", index=False)
        print(f"[OK] Excel 报表已导出: {self.excel_path}")
        return self.excel_path

    def clear_history(self):
        self.history = []
        self._save_history()


if __name__ == "__main__":
    r = DetectionRecorder()
    print(f"历史记录条数: {len(r.history)}")
    print(f"报表路径: {r.excel_path}")
