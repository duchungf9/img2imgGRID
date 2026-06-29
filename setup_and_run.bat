@echo off
chcp 65001 >nul
title grid-app (Region Select + Flux AI)

echo =============================================
echo  grid-app - Setup and Run
echo  (Region Select + SAM + Inpainting + Flux AI)
echo =============================================
echo.

:: ─── Drive selection ───
set "CACHE_DRIVE=E:"
if not exist "%CACHE_DRIVE%\" (
    echo [!] O %CACHE_DRIVE% khong ton tai, dung C:
    set "CACHE_DRIVE=C:"
)
set "HF_HOME=%CACHE_DRIVE%\huggingface_cache"
mkdir "%HF_HOME%\hub" 2>nul
echo [OK] HF cache: %HF_HOME%

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

:: ─── Kiem tra model see-through ───
if not exist "models\see_through\checkpoint-18000.pt" (
    echo.
    echo === See-Through Model ===
    echo Model body parts (1.2GB) chua co.
    echo Ban muon tai xuong? [1=Co / 2=De sau]
    choice /c 12 /n /m "Chon: "
    if not errorlevel 2 (
        echo [*] Dang tai model...
        mkdir models\see_through 2>nul
        venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; hf_hub_download('24yearsold/l2d_sam_iter2', 'checkpoint-18000.pt', local_dir='models/see_through', local_dir_use_symlinks=False)"
        echo [OK] Model da tai xong!
    )
    echo.
)

:: ─── Flux AI Model (optional, ~22GB) ───
echo.
echo === Flux AI Inpainting Model ===
echo FLUX.1-Fill-dev (~22GB) - Chay local, sieu nhanh (2-3s)
echo Yeu cau: HuggingFace token + da accept license
echo License: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
echo.
echo Ban muon tai Flux model? [1=Co / 2=De sau]
choice /c 12 /n /m "Chon: "
if not errorlevel 2 (
    echo.
    echo Nhap HuggingFace token cua ban:
    echo (Lay tai: https://huggingface.co/settings/tokens)
    set /p "HF_TOKEN=Token: "
    if not "%HF_TOKEN%"=="" (
        echo [*] Dang tai FLUX.1-Fill-dev (~22GB, se lau)...
        set "HF_HOME=%HF_HOME%"
        set "HF_TOKEN=%HF_TOKEN%"
        venv\Scripts\python.exe -c ^
            "import os; os.environ['HF_TOKEN']='%HF_TOKEN%'; os.environ['HF_HOME']='%HF_HOME%';^
             from huggingface_hub import hf_hub_download;^
             files=['ae.safetensors','flux1-fill-dev.safetensors','model_index.json'];^
             for f in files:^
                 try:^
                     print(f'Downloading {f}...');^
                     hf_hub_download('black-forest-labs/FLUX.1-Fill-dev', f, cache_dir='%HF_HOME%/hub');^
                     print(f'  OK');^
                 except Exception as e:^
                     print(f'  Loi: {e}')"
        echo.
        echo [OK] Flux model da tai xong!
    ) else (
        echo [!] Bo qua (token rong)
    )
)
echo.

:: ─── Run ───
echo.
echo =============================================
echo  Dang khoi dong server...
echo  Mo trinh duyet: http://localhost:5000
echo =============================================
echo.

set HF_HOME=%HF_HOME%
start http://localhost:5000
venv\Scripts\python.exe app.py

pause
