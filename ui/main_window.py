# -*- coding: utf-8 -*-
"""
============================================================================
模块功能: 简易 PyQt5 可视化界面
  功能: 批量上传图片 -> 一键检测 -> 原图/结果图对比显示 -> 历史记录 -> 导出报表
  对应需求: 核心功能模块 7 (简易PyQt可视化界面)
  说明: 支持一次性上传多张图片, 后台线程逐张检测并显示进度
        结果区左右并排: 左侧原图, 右侧该图的带框检测结果
============================================================================
运行: python app.py
============================================================================
"""
import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox,
    QGroupBox, QSplitter, QHeaderView, QCheckBox, QListWidget,
    QProgressBar, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import GlueDetector
from src.statistics import DetectionRecorder
from src.utils import load_config, PROJECT_ROOT


class DetectWorker(QThread):
    """后台批量检测线程, 避免界面卡顿; 每检测完一张即发出结果信号"""
    result_ready = pyqtSignal(dict, int, int)   # (检测结果, 当前序号, 总数)
    all_finished = pyqtSignal(int)              # 全部完成, 参数为总数
    error = pyqtSignal(str)

    def __init__(self, detector, image_paths):
        super().__init__()
        self.detector = detector
        self.image_paths = image_paths

    def run(self):
        try:
            total = len(self.image_paths)
            for idx, path in enumerate(self.image_paths, 1):
                result = self.detector.detect(path)
                self.result_ready.emit(result, idx, total)
            self.all_finished.emit(total)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("锂电池胶路缺陷检测系统  |  YOLOv8 · 2.5D光源胶检")
        self.resize(1280, 800)
        self.cfg = load_config()
        self.detector = None
        self.recorder = DetectionRecorder(self.cfg)
        self.image_paths = []          # 已上传的多张图片路径
        self.result_cache = {}         # 路径 -> 检测结果(含 annotated), 用于切换图片回看
        self.current_result = None
        self._batch_ok = 0             # 本批次 OK 计数
        self._batch_ng = 0             # 本批次 NG 计数
        self._init_ui()
        self._load_model()
        self._refresh_history()

    # ---------- UI 构建 ----------
    def _init_ui(self):
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 标题
        title = QLabel("锂电池胶路缺陷检测系统 — 基于 YOLOv8")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#1a4f8b; padding:8px;")
        main_layout.addWidget(title)

        # 顶部控制区
        ctrl = QHBoxLayout()
        self.btn_open = QPushButton("上传图片(可多选)")
        self.btn_open.clicked.connect(self.open_images)
        self.btn_detect = QPushButton("一键检测")
        self.btn_detect.clicked.connect(self.run_detect)
        self.btn_export = QPushButton("导出Excel报表")
        self.btn_export.clicked.connect(self.export_report)
        self.btn_clear = QPushButton("清空历史")
        self.btn_clear.clicked.connect(self.clear_history)
        for b in (self.btn_open, self.btn_detect, self.btn_export, self.btn_clear):
            b.setMinimumHeight(34)
            ctrl.addWidget(b)
        self.cb_preprocess = QCheckBox("启用预处理(抑制2.5D反光)")
        self.cb_preprocess.setChecked(True)
        ctrl.addWidget(self.cb_preprocess)
        main_layout.addLayout(ctrl)

        # 主体: 左侧图像对比 + 右侧历史
        splitter = QSplitter(Qt.Horizontal)
        left = QGroupBox("图像与检测结果")
        left_layout = QVBoxLayout()

        # 图像对比区: 左原图 / 右结果图
        img_compare = QHBoxLayout()
        orig_box = QGroupBox("原图")
        orig_layout = QVBoxLayout()
        self.original_label = QLabel("等待上传图片")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(320, 400)
        self.original_label.setStyleSheet("border:1px solid #ccc; background:#f7f7f7;")
        orig_layout.addWidget(self.original_label)
        orig_box.setLayout(orig_layout)
        img_compare.addWidget(orig_box)

        res_box = QGroupBox("检测结果图(带框)")
        res_layout = QVBoxLayout()
        self.result_image_label = QLabel("尚未检测\n点击[一键检测]开始")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumSize(320, 400)
        self.result_image_label.setStyleSheet("border:1px solid #ccc; background:#f7f7f7;")
        res_layout.addWidget(self.result_image_label)
        res_box.setLayout(res_layout)
        img_compare.addWidget(res_box)
        left_layout.addLayout(img_compare)

        self.result_label = QLabel("检测状态: --")
        self.result_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        left_layout.addWidget(self.result_label)
        self.detail_label = QLabel("缺陷信息将在此显示")
        self.detail_label.setWordWrap(True)
        left_layout.addWidget(self.detail_label)

        # 已上传图片列表
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(88)
        self.image_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_list.itemClicked.connect(self.on_image_selected)
        left_layout.addWidget(QLabel("已上传图片(点击可切换查看原图与结果图):"))
        left_layout.addWidget(self.image_list)

        # 批量检测进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        left.setLayout(left_layout)
        splitter.addWidget(left)

        right = QGroupBox("检测历史")
        right_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["序号", "时间", "来源", "判定", "缺陷类型", "耗时(ms)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        right_layout.addWidget(self.table)
        right.setLayout(right_layout)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        self.statusBar().showMessage("就绪。左原图 / 右结果图对比; 支持一次上传多张批量检测。")

    def _load_model(self):
        try:
            self.detector = GlueDetector(use_preprocess=self.cb_preprocess.isChecked())
            self.statusBar().showMessage(f"模型加载完成: {os.path.basename(str(self.detector.weight))}")
        except Exception as e:
            QMessageBox.warning(self, "模型加载失败",
                                f"无法加载模型: {e}\n请先安装依赖并训练模型, 详见 README。")

    # ---------- 功能槽 ----------
    def open_images(self):
        """一次性选择多张图片"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择胶路图片(可多选, 按住Ctrl/Shift多选)",
            os.path.join(PROJECT_ROOT, "data", "dataset", "images", "test"),
            "图片 (*.jpg *.png *.jpeg)")
        if not paths:
            return
        self.image_paths = list(paths)
        self.result_cache.clear()
        self.image_list.clear()
        for p in self.image_paths:
            self.image_list.addItem(os.path.basename(p))
        first = self.image_paths[0]
        self._set_pixmap(self.original_label, cv2.imread(first), "无法读取原图")
        self._set_pixmap(self.result_image_label, None, "尚未检测\n点击[一键检测]开始")
        self.result_label.setText("检测状态: 待检测")
        self.detail_label.setText(f"已上传 {len(self.image_paths)} 张图片, 点击[一键检测]开始批量检测")
        self.statusBar().showMessage(f"已加载 {len(self.image_paths)} 张图片")

    def on_image_selected(self, item):
        """点击列表中的图片: 左侧显示原图, 右侧显示该图已有的检测结果"""
        row = self.image_list.row(item)
        if not (0 <= row < len(self.image_paths)):
            return
        path = self.image_paths[row]
        self._set_pixmap(self.original_label, cv2.imread(path), "无法读取原图")
        result = self.result_cache.get(path)
        if result is not None:
            self._set_pixmap(self.result_image_label, result["annotated"])
            self._update_result_text(result)
        else:
            self._set_pixmap(self.result_image_label, None, "该图尚未检测\n点击[一键检测]开始")
            self.result_label.setText("检测状态: 待检测")
            self.detail_label.setText(f"已选中: {os.path.basename(path)}")

    def run_detect(self):
        if not self.image_paths:
            QMessageBox.information(self, "提示", "请先上传图片(可多选)")
            return
        if self.detector is None:
            self._load_model()
        if self.detector is None:
            return
        self.detector.use_preprocess = self.cb_preprocess.isChecked()
        if self.detector.use_preprocess and not hasattr(self.detector, "preprocessor"):
            from src.preprocessing import GluePreprocessor
            self.detector.preprocessor = GluePreprocessor(self.cfg)
        self.btn_detect.setEnabled(False)
        self._batch_ok = 0
        self._batch_ng = 0
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.progress.setMaximum(len(self.image_paths))
        self.statusBar().showMessage(f"批量检测中... 共 {len(self.image_paths)} 张")
        self.worker = DetectWorker(self.detector, self.image_paths)
        self.worker.result_ready.connect(self.on_detect_done)
        self.worker.all_finished.connect(self.on_batch_finished)
        self.worker.error.connect(self.on_detect_error)
        self.worker.start()

    def on_detect_done(self, result, current, total):
        """每检测完一张: 左侧原图 + 右侧结果图 + 记录历史 + 更新进度"""
        self.current_result = result
        if 1 <= current <= len(self.image_paths):
            path = self.image_paths[current - 1]
            self.result_cache[path] = result
            self._set_pixmap(self.original_label, cv2.imread(path), "无法读取原图")
        self._set_pixmap(self.result_image_label, result["annotated"])
        status = result["status"]
        if status == "OK":
            self._batch_ok += 1
        else:
            self._batch_ng += 1
        self._update_result_text(result, progress=f"第 {current}/{total} 张")
        self.progress.setValue(current)
        try:
            self.recorder.record(result)
            self._refresh_history()
        except Exception as e:
            self.statusBar().showMessage(f"记录失败: {e}")

    def _update_result_text(self, result, progress=""):
        """统一更新判定标签与缺陷明细"""
        status = result["status"]
        color = "green" if status == "OK" else "red"
        parts = [f'<span style="color:{color};">检测判定: {status}</span>']
        if progress:
            parts.append(progress)
        parts.append(f'耗时: {result["elapsed_ms"]}ms')
        parts.append(f'置信度阈值: {result["conf"]}')
        self.result_label.setText("  |  ".join(parts))
        if result["defects"]:
            lines = [f"• {d['name_zh']} ({d['class']}): 置信度 {d['conf']:.2f}, 框 {d['bbox']}"
                     for d in result["defects"]]
            self.detail_label.setText("检出目标:\n" + "\n".join(lines))
        else:
            self.detail_label.setText("未检出目标 (请检查模型是否已针对胶路数据训练)")

    def on_batch_finished(self, total):
        """全部检测完成: 恢复按钮并显示汇总"""
        self.btn_detect.setEnabled(True)
        self.statusBar().showMessage(
            f"批量检测完成: 共 {total} 张, OK {self._batch_ok} 张 / NG {self._batch_ng} 张, 已全部记录")

    def on_detect_error(self, msg):
        self.btn_detect.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "检测出错", msg)

    def export_report(self):
        try:
            path = self.recorder.export_excel()
            QMessageBox.information(self, "导出成功", f"报表已导出:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def clear_history(self):
        reply = QMessageBox.question(self, "确认", "确定清空所有历史记录?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.recorder.clear_history()
            self._refresh_history()
            self.statusBar().showMessage("历史已清空")

    def _set_pixmap(self, label, bgr, placeholder=""):
        """把 BGR 图像显示到指定 QLabel; 无图时显示占位文字"""
        if bgr is None:
            label.setPixmap(QPixmap())
            label.setText(placeholder)
            return
        label.setText("")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label.setPixmap(
            pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_history(self):
        rows = self.recorder.history
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r.get("seq", ""), r.get("time", ""), r.get("source", ""),
                    r.get("status", ""), r.get("defect_types", ""), r.get("elapsed_ms", "")]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if j == 3 and v == "NG":
                    item.setForeground(Qt.red)
                elif j == 3 and v == "OK":
                    item.setForeground(Qt.darkGreen)
                self.table.setItem(i, j, item)


def run_app():
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
