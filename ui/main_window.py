# -*- coding: utf-8 -*-
"""
============================================================================
模块功能: 简易 PyQt5 可视化界面
  功能: 上传/导入图片 -> 一键检测 -> 显示缺陷框与判定 -> 历史记录 -> 导出报表
  对应需求: 核心功能模块 7 (简易PyQt可视化界面)
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
    QGroupBox, QSplitter, QHeaderView, QComboBox, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import GlueDetector
from src.statistics import DetectionRecorder
from src.utils import load_config, PROJECT_ROOT


class DetectWorker(QThread):
    """后台检测线程, 避免界面卡顿"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, detector, image_path):
        super().__init__()
        self.detector = detector
        self.image_path = image_path

    def run(self):
        try:
            result = self.detector.detect(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("锂电池胶路缺陷检测系统  |  YOLOv8 · 2.5D光源胶检")
        self.resize(1200, 760)
        self.cfg = load_config()
        self.detector = None
        self.recorder = DetectionRecorder(self.cfg)
        self.current_path = None
        self.current_result = None
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
        self.btn_open = QPushButton("上传图片")
        self.btn_open.clicked.connect(self.open_image)
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

        # 主体: 左侧图像显示 + 右侧历史
        splitter = QSplitter(Qt.Horizontal)
        left = QGroupBox("图像与检测结果")
        left_layout = QVBoxLayout()
        self.image_label = QLabel("请上传胶路图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border:1px solid #ccc; background:#f7f7f7;")
        left_layout.addWidget(self.image_label)
        self.result_label = QLabel("检测状态: --")
        self.result_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        left_layout.addWidget(self.result_label)
        self.detail_label = QLabel("缺陷信息将在此显示")
        self.detail_label.setWordWrap(True)
        left_layout.addWidget(self.detail_label)
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

        self.statusBar().showMessage("就绪。请先确认 weights/best.pt 已训练, 否则使用预训练权重演示流程。")

    def _load_model(self):
        try:
            self.detector = GlueDetector(use_preprocess=self.cb_preprocess.isChecked())
            self.statusBar().showMessage(f"模型加载完成: {os.path.basename(str(self.detector.weight))}")
        except Exception as e:
            QMessageBox.warning(self, "模型加载失败",
                                f"无法加载模型: {e}\n请先安装依赖并训练模型, 详见 README。")

    # ---------- 功能槽 ----------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择胶路图片", os.path.join(PROJECT_ROOT, "data"),
            "图片 (*.jpg *.png *.jpeg)")
        if not path:
            return
        self.current_path = path
        img = cv2.imread(path)
        self._show_image(img)
        self.result_label.setText("检测状态: 待检测")
        self.detail_label.setText(f"图片: {os.path.basename(path)}")
        self.statusBar().showMessage(f"已加载: {path}")

    def run_detect(self):
        if not self.current_path:
            QMessageBox.information(self, "提示", "请先上传图片")
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
        self.statusBar().showMessage("检测中...")
        self.worker = DetectWorker(self.detector, self.current_path)
        self.worker.finished.connect(self.on_detect_done)
        self.worker.error.connect(self.on_detect_error)
        self.worker.start()

    def on_detect_done(self, result):
        self.btn_detect.setEnabled(True)
        self.current_result = result
        self._show_image(result["annotated"])
        status = result["status"]
        color = "green" if status == "OK" else "red"
        self.result_label.setText(
            f'<span style="color:{color};">检测判定: {status}</span>  |  '
            f'耗时: {result["elapsed_ms"]}ms  |  置信度阈值: {result["conf"]}')
        if result["defects"]:
            lines = [f"• {d['name_zh']} ({d['class']}): 置信度 {d['conf']:.2f}, 框 {d['bbox']}"
                     for d in result["defects"]]
            self.detail_label.setText("检出目标:\n" + "\n".join(lines))
        else:
            self.detail_label.setText("未检出目标 (请检查模型是否已针对胶路数据训练)")
        try:
            self.recorder.record(result)
            self._refresh_history()
            self.statusBar().showMessage(f"检测完成, 判定 {status}, 已记录")
        except Exception as e:
            self.statusBar().showMessage(f"记录失败: {e}")

    def on_detect_error(self, msg):
        self.btn_detect.setEnabled(True)
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

    def _show_image(self, bgr):
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
