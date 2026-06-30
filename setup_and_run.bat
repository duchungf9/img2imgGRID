@echo off
chcp 65001 >nul
title grid-app (Region Select + AI Inpainting)

echo =============================================
echo  grid-app - Setup and Run
echo  (Region Select + SAM + Inpainting)
echo =============================================
echo.

:: ─── Set HuggingFace cache to E: drive ───
set "HF_HOME=E:\huggingface_cache"
set "HUGGINGFACE_HUB_CACHE=E:\huggingface_cache\hub"
mkdir "%HF_HOME%\hub" 2>nul
echo [OK] HF cache: %HF_HOME%

:: ─── Check Python ───
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [LOI] Khong tim thay Python
    pause
    exit /b 1
)
echo [OK] Python:
python --version

:: ─── venv ───
if not exist "venv\" (
    echo [*] Tao venv...
    python -m venv venv
)
call venv\Scripts\activate.bat

:: ─── Install dependencies ───
echo [*] Cai dat thu vien...
pip install -r requirements.txt --quiet

:: ─── Run ───
echo.
echo =============================================
echo  Dang khoi dong server...
echo  Mo: http://localhost:5000
echo =============================================
echo.

set HF_HOME=%HF_HOME%
start http://localhost:5000
python app.py

pause
