@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py --mode demo
) else (
    py -3.11 main.py --mode demo
)
pause
