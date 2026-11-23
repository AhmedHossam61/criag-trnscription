@echo off
REM Create an isolated venv for one ASR backend. Windows CMD. Uses uv.
REM
REM   setup.cmd qwencleo             recommended for Egyptian + code-switching
REM   setup.cmd cohere
REM   setup.cmd qwen3asr
REM   setup.cmd faster-whisper
REM
REM   set CUDA=cu128 && setup.cmd qwencleo       override the wheel index
REM   set PYTAG=3.12 && setup.cmd qwencleo       force a specific Python version
REM
REM One venv per backend on purpose: the backends need different torch and
REM transformers versions. Sharing one environment is how you end up
REM reinstalling torch at 3am.

setlocal EnableDelayedExpansion

set "BACKEND=%~1"
if "%CUDA%"=="" set "CUDA=cu126"
if "%PYTAG%"=="" set "PYTAG=3.12"

REM ---- refuse to run inside an active venv --------------------------------
if defined VIRTUAL_ENV (
    echo ERROR: a virtual environment is already active:
    echo          %VIRTUAL_ENV%
    echo.
    echo        Run  deactivate  first, then re-run this script.
    exit /b 1
)

REM ---- validate the backend name ------------------------------------------
if "%BACKEND%"=="" goto :usage
if /I "%BACKEND%"=="qwencleo"       goto :validated
if /I "%BACKEND%"=="qwen3asr"       goto :validated
if /I "%BACKEND%"=="cohere"         goto :validated
if /I "%BACKEND%"=="faster-whisper" goto :validated
echo unknown backend: %BACKEND%
goto :usage

:validated
set "VENV=.venv-%BACKEND%"
set "TORCH_INDEX=https://download.pytorch.org/whl/%CUDA%"

REM ---- uv must be installed -----------------------------------------------
where uv >nul 2>&1
if errorlevel 1 (
    echo ERROR: uv not found on PATH.
    echo.
    echo Install it with one of:
    echo   winget install --id=astral-sh.uv -e
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo   pip install uv
    exit /b 1
)
for /f "tokens=*" %%V in ('uv --version 2^>^&1') do echo ^>^> %%V

if exist "%VENV%\" (
    echo ERROR: %VENV% already exists.
    echo        Remove it first:   rmdir /s /q %VENV%
    exit /b 1
)

REM ---- ffmpeg is not a pip package ---------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: ffmpeg not found on PATH. The pipeline cannot decode audio.
    echo          Install it with:  winget install Gyan.FFmpeg
    echo          Then open a NEW cmd window so PATH refreshes.
    echo.
)

REM ---- create the venv ----------------------------------------------------
REM uv downloads a managed CPython if PYTAG's version is not installed.
echo ^>^> creating %VENV% (Python %PYTAG%)
uv venv "%VENV%" --python %PYTAG%
if errorlevel 1 (
    echo ERROR: uv venv failed.
    exit /b 1
)

REM Every install below targets this venv explicitly, so no activation needed.
set "UV_PIP=uv pip install --python %VENV%\Scripts\python.exe"

if /I "%BACKEND%"=="qwencleo"       goto :inst_qwencleo
if /I "%BACKEND%"=="qwen3asr"       goto :inst_qwen3asr
if /I "%BACKEND%"=="cohere"         goto :inst_cohere
if /I "%BACKEND%"=="faster-whisper" goto :inst_whisper

:inst_qwencleo
REM Order matters: torch first so the package cannot re-resolve it, then
REM QwenCleo with --no-deps, then its remaining runtime deps.
REM torch is deliberately NOT pinned. QwenCleo's README suggests 2.5.1, but
REM that pin has no wheels for Python 3.13 and none on the retired cu121
REM index. --no-deps means the pin is not enforced anyway.
echo ^>^> installing torch (%CUDA%)
%UV_PIP% torch torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 goto :failed_torch
call :verify_cuda
if errorlevel 1 goto :failed_cuda
echo ^>^> installing qwencleo-asr (--no-deps)
%UV_PIP% qwencleo-asr --no-deps
if errorlevel 1 goto :failed
echo ^>^> installing remaining deps
%UV_PIP% -r requirements-qwencleo.txt
if errorlevel 1 goto :failed
goto :verify

:inst_qwen3asr
echo ^>^> installing torch (%CUDA%)
%UV_PIP% torch torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 goto :failed_torch
call :verify_cuda
if errorlevel 1 goto :failed_cuda
%UV_PIP% -r requirements-qwen3asr.txt
if errorlevel 1 goto :failed
goto :verify

:inst_cohere
echo ^>^> installing torch (%CUDA%)
%UV_PIP% torch torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 goto :failed_torch
call :verify_cuda
if errorlevel 1 goto :failed_cuda
%UV_PIP% -r requirements-cohere.txt
if errorlevel 1 goto :failed
echo.
echo ^>^> NOTE: these weights are gated.
echo    1. accept the terms at
echo       https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026
echo    2. run:  hf auth login
goto :verify

:inst_whisper
REM CTranslate2 based -- no torch needed at all.
%UV_PIP% -r requirements-whisper.txt
if errorlevel 1 goto :failed
goto :verify

:verify
echo.
echo ^>^> verifying
"%VENV%\Scripts\python.exe" craig_transcribe.py --check-env --backend %BACKEND%
echo.
echo Done. Activate with:  %VENV%\Scripts\activate.bat
echo Then:                 python craig_transcribe.py recording.zip -o out --backend %BACKEND%
exit /b 0

:verify_cuda
REM Hard-fail if uv installed a CPU-only wheel or the driver can't reach the GPU.
"%VENV%\Scripts\python.exe" -c "import torch, sys; ok = torch.cuda.is_available(); print(f'  torch {torch.__version__} | CUDA build: {torch.version.cuda} | cuda.is_available: {ok}'); sys.exit(0 if ok else 1)"
exit /b %errorlevel%

:failed_cuda
echo.
echo ERROR: torch installed but torch.cuda.is_available() is False.
echo        The transcriber would fall back to CPU and be unusably slow.
echo.
echo Fix, in order of likelihood:
echo   1. NVIDIA driver missing or too old. Run:  nvidia-smi
echo      A driver supporting CUDA 12.x is required. Update via GeForce
echo      Experience or https://www.nvidia.com/Download/index.aspx
echo.
echo   2. uv resolved a CPU-only wheel (happens if the CUDA index was
echo      unreachable). Rebuild the venv with a fresh index:
echo          rmdir /s /q %VENV%
echo          set CUDA=cu128 ^&^& setup.cmd %BACKEND%
echo.
echo   3. Confirm the installed build is a CUDA wheel:
echo          %VENV%\Scripts\python.exe -c "import torch; print(torch.version.cuda)"
echo      Should print a version like 12.6, NOT None.
exit /b 1

:failed_torch
echo.
echo torch install failed. Most likely causes, in order:
echo.
echo   1. The CUDA index has no wheels for Python %PYTAG%.
echo      Python 3.13 needs torch 2.6 or newer. Try 3.12:
echo          rmdir /s /q %VENV%
echo          set PYTAG=3.12 ^&^& setup.cmd %BACKEND%
echo.
echo   2. That CUDA index no longer exists. cu121 was retired; current
echo      indexes are cu126, cu128, cu129, cu130. Try:
echo          rmdir /s /q %VENV%
echo          set CUDA=cu128 ^&^& setup.cmd %BACKEND%
echo.
echo   3. Your driver is genuinely too old. Check with nvidia-smi; a driver
echo      supporting CUDA 12.x is needed for any cu12x build.
exit /b 1

:failed
echo.
echo Install failed. See the uv output above.
exit /b 1

:usage
echo usage: setup.cmd {qwencleo^|qwen3asr^|cohere^|faster-whisper}
echo.
echo   set CUDA=cu128 ^&^& setup.cmd qwencleo      pick the wheel index
echo   set PYTAG=3.12 ^&^& setup.cmd qwencleo      pick the Python version
exit /b 1
