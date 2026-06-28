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
venv\Scripts\python.exe -m pip install --upgrade pip -q

:: ─── Install PyTorch ───
echo [*] Kiem tra PyTorch...
venv\Scripts\python.exe -c "import torch" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [!] PyTorch chua co. Dang cai dat...

    :: Thu cai PyTorch + CUDA truoc
    echo [*] Thu cai PyTorch + CUDA (2.6.0+cu124)...
    venv\Scripts\pip.exe install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
    venv\Scripts\pip.exe install torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps

    :: Kiem tra xem CUDA co hoat dong khong
    venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()" 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [!] CUDA khong kha dung. Cai dat PyTorch CPU...
        venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-deps
        echo [OK] PyTorch CPU da cai xong
    ) else (
        echo [OK] PyTorch + CUDA da cai xong
    )
) else (
    :: PyTorch da co, kiem tra CUDA
    venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()" 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [OK] PyTorch co (CPU mode)
    ) else (
        echo [OK] PyTorch + CUDA da co
    )
)

:: ─── Install other dependencies ───
echo [*] Cai dat thu vien...
venv\Scripts\pip.exe install -r requirements.txt --quiet

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
        venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('24yearsold/l2d_sam_iter2', 'checkpoint-18000.pt', local_dir='models/see_through', local_dir_use_symlinks=False)"
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
venv\Scripts\python.exe app.py

pause
