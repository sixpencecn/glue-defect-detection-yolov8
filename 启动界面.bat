@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 锂电池胶路缺陷检测系统 - YOLOv8

echo ============================================
echo   锂电池胶路缺陷检测系统 - 基于 YOLOv8
echo ============================================
echo.

rem 优先使用系统 Python (D:\python\路径\python.exe), 已安装全部依赖
set "PY=D:\python\路径\python.exe"
if not exist "%PY%" set "PY=python"

echo 使用 Python: %PY%
echo 正在启动界面...
echo.

"%PY%" app.py

if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出, 请检查上方提示信息。
    pause
)
