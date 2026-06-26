@echo off
chcp 65001 >nul
title img2imgGRID

echo =============================================
echo  img2imgGRID - Setup & Run
echo =============================================
echo.

:: ─── Check Python ───
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [LOI] Khong tim thay Python. Hay cai Python tu python.org
    pause
    exit /b 1
)

echo [OK] Python:
python --version

:: ─── Create venv (if not exists) ───
if not exist "venv\" (
    echo [*] Tao virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

:: ─── Upgrade pip ───
echo [*] Nang cap pip...
python -m pip install --upgrade pip -q

:: ─── Install PyTorch CUDA ───
echo [*] Kiem tra PyTorch...
python -c "import torch; assert torch.cuda.is_available()" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [*] Cai dat PyTorch + CUDA (2.6.0+cu124) ~4GB...
    pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
    pip install torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
) else (
    echo [OK] PyTorch CUDA da co
)

:: ─── Install other dependencies ───
echo [*] Cai dat thu vien...
pip install -r requirements.txt --quiet

:: ─── Kiểm tra model see-through ───
if not exist "models\see_through\checkpoint-18000.pt" (
    echo.
    echo === See-Through Model ===
    echo Model body parts (1.2GB) chua co.
    echo Ban muon tai xuong de dung duoc See-Through tab?
    echo [1] Co, tai xuong (khuyen nghi neu co GPU)
    echo [2] Khong, de sau (van dung duoc SAM + Inpaint)
    choice /c 12 /n /m "Chon 1 hoac 2: "
    if errorlevel 2 (
        echo [*] Bo qua. Khi nao muon tai, chay: huggingface-cli download 24yearsold/l2d_sam_iter2 checkpoint-18000.pt
    ) else (
        echo [*] Dang tai model (1.2GB)...
        mkdir models\see_through 2>nul
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download('24yearsold/l2d_sam_iter2', 'checkpoint-18000.pt', local_dir='models/see_through', local_dir_use_symlinks=False)"
        echo [OK] Model da tai xong!
    )
    echo.
)

:: ─── Run ───
echo.
echo =============================================
echo  Dang khoi dong server...
echo  Mo trinh duyet: http://localhost:5000
echo =============================================
echo.

start http://localhost:5000
python app.py

pause
