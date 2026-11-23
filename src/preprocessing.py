# -*- coding: utf-8 -*-
"""
============================================================================
模块功能: 图像预处理 (针对 2.5D 光源胶路反光/光照波动问题)
  1) ROI 裁剪: 只保留胶路区域, 去除无关背景
  2) 高斯滤波: 抑制细颗粒噪声
  3) CLAHE 亮度均衡: 限制对比度自适应直方图均衡, 抑制局部反光过曝、提升暗部胶路
业务意义: 传统阈值/轮廓法在反光下误检漏检严重, 预处理 + YOLO 显著提升鲁棒性
============================================================================
"""
import cv2
import numpy as np


class GluePreprocessor:
    def __init__(self, config=None):
        if config is None:
            from .utils import load_config
            config = load_config()
        img_cfg = config.get("image", {})
        roi = img_cfg.get("roi", {})
        self.roi = (roi.get("x1"), roi.get("y1"), roi.get("x2"), roi.get("y2"))
        self.gaussian_kernel = int(img_cfg.get("gaussian_kernel", 3))
        self.enable_clahe = bool(img_cfg.get("enable_clahe", True))
        self.clahe_clip = float(img_cfg.get("clahe_clip", 2.0))
        self.clahe_grid = int(img_cfg.get("clahe_grid", 8))
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )

    def crop_roi(self, image):
        """ROI 裁剪"""
        x1, y1, x2, y2 = self.roi
        h, w = image.shape[:2]
        x1 = 0 if x1 is None else x1
        y1 = 0 if y1 is None else y1
        x2 = w if x2 is None else min(x2, w)
        y2 = h if y2 is None else min(y2, h)
        if (x1, y1, x2, y2) == (0, 0, w, h):
            return image
        return image[y1:y2, x1:x2]

    def denoise(self, image):
        """高斯滤波"""
        if self.gaussian_kernel and self.gaussian_kernel > 1:
            k = self.gaussian_kernel if self.gaussian_kernel % 2 == 1 else self.gaussian_kernel + 1
            return cv2.GaussianBlur(image, (k, k), 0)
        return image

    def equalize(self, image):
        """CLAHE 亮度均衡: 在 LAB 空间对 L 通道均衡, 保留颜色信息"""
        if not self.enable_clahe:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    def preprocess(self, image, steps=("roi", "denoise", "equalize")):
        """完整预处理流水线。输入/输出均为 BGR ndarray。"""
        out = image.copy()
        if "roi" in steps:
            out = self.crop_roi(out)
        if "denoise" in steps:
            out = self.denoise(out)
        if "equalize" in steps:
            out = self.equalize(out)
        return out

    @staticmethod
    def from_file(path, config=None):
        """读取图像并预处理"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        return GluePreprocessor(config).preprocess(img), img


if __name__ == "__main__":
    # 快速自测: 对一张样例图做预处理并保存对比
    import os
    from .utils import PROJECT_ROOT
    p = GluePreprocessor()
    sample = os.path.join(PROJECT_ROOT, "data", "raw", "images", "OK_0000.jpg")
    if os.path.exists(sample):
        orig = cv2.imread(sample)
        proc = p.preprocess(orig)
        cv2.imwrite(os.path.join(PROJECT_ROOT, "docs", "preprocess_compare.png"),
                    np.hstack([orig, proc]))
        print("[OK] 预处理自测通过, 对比图已保存")
