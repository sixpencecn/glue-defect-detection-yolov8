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


def _fix_qt_plugin_path():
    """
    修复 Windows 中文用户名路径下 Qt 平台插件加载失败的问题:
    "This application failed to start because no Qt platform plugin could be initialized"
    在导入 PyQt5 之前显式设置插件路径。
    """
    if sys.platform != "win32":
        return
    try:
        import site
        site_pkgs = site.getsitepackages()[-1]
        plugin_dir = os.path.join(site_pkgs, "PyQt5", "Qt5", "plugins")
        if os.path.isdir(os.path.join(plugin_dir, "platforms")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
    except Exception:
        pass  # 自动探测失败时保持默认行为


_fix_qt_plugin_path()


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
