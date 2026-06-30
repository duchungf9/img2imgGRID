@echo off
chcp 65001 >nul
title Download Flux Fill Model

echo =============================================
echo  Download FLUX.1-Fill-dev (continue)
echo =============================================
echo.
echo Dataset: E:\huggingface_cache
echo.
echo Can: HuggingFace token + da accept license tai:
echo https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
echo.

:: Check if token exists
if "%HF_TOKEN%"=="" (
    echo Nhap HuggingFace token cua ban:
    set /p "HF_TOKEN=Token: "
)

echo.
echo Starting download (tu dong resume)...
echo.

python -c ^
    "import os, requests as req, time;^
    TOKEN='%HF_TOKEN%';^
    BASE='https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main';^
    SNAP='E:/huggingface_cache/hub/models--black-forest-labs--FLUX.1-Fill-dev/snapshots/358293da0354175698b67ec8299acf928313a78a';^
    hdr={'Authorization': f'Bearer {TOKEN}'};^
    files=['transformer/diffusion_pytorch_model-00001-of-00003.safetensors','transformer/diffusion_pytorch_model-00002-of-00003.safetensors','transformer/diffusion_pytorch_model-00003-of-00003.safetensors','vae/diffusion_pytorch_model.safetensors'];^
    for fpath in files:^
        dest=os.path.join(SNAP,fpath);^
        os.makedirs(os.path.dirname(dest),exist_ok=True);^
        name=fpath.split('/')[-1];^
        resp=req.head(BASE+'/'+fpath,headers=hdr,allow_redirects=True);^
        total=int(resp.headers.get('content-length',0));^
        existing=os.path.getsize(dest) if os.path.exists(dest) else 0;^
        print(f'{name}: {existing/1e9:.1f}GB / {total/1e9:.1f}GB');^
        if existing>=total: print('  Done, skip'); continue;^
        dl_hdr=hdr.copy();^
        mode='ab' if existing>0 else 'wb';^
        if existing>0: dl_hdr['Range']=f'bytes={existing}-';^
        resp=req.get(BASE+'/'+fpath,headers=dl_hdr,stream=True,timeout=30);^
        t0=time.time(); dl=existing;^
        with open(dest,mode) as fp:^
            for chunk in resp.iter_content(chunk_size=16*1024*1024):^
                if chunk: fp.write(chunk); fp.flush(); dl+=len(chunk);^
                pct=dl/total*100;^
                print(f'\r  {pct:.0f}% ({dl/1e9:.1f}GB) {int((dl-existing)/(time.time()-t0+0.1)/1e6)}MB/s',end='',flush=True);^
        print(f'\n  OK ({dl/1e9:.1f}GB)');^
    print('All done!')"

if %ERRORLEVEL% neq 0 (
    echo.
    echo Loi xay ra. Kiem tra token va thu lai.
    pause
    exit /b 1
)

echo.
echo =============================================
echo  Download hoan tat!
echo  Chay app: python app.py
echo =============================================
pause
