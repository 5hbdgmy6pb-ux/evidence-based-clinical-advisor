@echo off
rem ============================================================
rem  本文件必须保存为 GBK(ANSI) 编码 + CRLF 换行。
rem  若用 UTF-8 保存，cmd.exe 会按 GBK 解码中文，
rem  多字节字符的尾字节会吞掉换行符，导致整个脚本解析失败。
rem ============================================================
cd /d "%~dp0"
title 个人健康文献助手

echo ========================================
echo    个人健康文献助手
echo ========================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [错误] 找不到虚拟环境 venv
    echo 请先按 README.md 的说明创建环境并安装依赖。
    echo.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"

echo 正在启动，浏览器会自动打开 http://localhost:8501
echo 界面会立即打开；首次提问时加载模型，约 15 秒。
echo 关闭本窗口即可停止服务。
echo.

python -m streamlit run app.py

echo.
echo 服务已停止。
pause
