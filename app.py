# -*- coding: utf-8 -*-
"""
============================================================================
锂电池胶路缺陷检测系统 - 图形界面入口
运行: python app.py
============================================================================
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    try:
        from ui.main_window import run_app
        run_app()
    except ImportError as e:
        print("=" * 60)
        print("依赖缺失, 请先执行:")
        print("  pip install -r requirements.txt")
        print(f"错误详情: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
