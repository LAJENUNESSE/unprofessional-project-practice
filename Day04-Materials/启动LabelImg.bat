@echo off
rem 一键启动 LabelImg（双击运行）
cd /d "%~dp0labelImg-master"
"..\.venv\Scripts\python.exe" labelImg.py
