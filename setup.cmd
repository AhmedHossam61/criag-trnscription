@echo off
REM Create an isolated venv for one ASR backend.  Windows CMD.
REM
REM   setup.cmd qwencleo             recommended for Egyptian + code-switching
REM   setup.cmd cohere
REM   setup.cmd qwen3asr
REM   setup.cmd faster-whisper
REM
REM   set CUDA=cu128 && setup.cmd qwencleo       override the wheel index
REM   set PYTAG=-3.11 && setup.cmd qwencleo      force a specific interpreter
REM
REM One venv per backend on purpose: the backends need different torch and
REM transformers versions. Sharing one environment is how you end up
REM reinstalling torch at 3am.

setlocal EnableDelayedExpansion

set "BACKEND=%~1"
if "%CUDA%"=="" set "CUDA=cu126"

REM ---- refuse to run inside an active venv --------------------------------
REM Creating a venv while it is activated fails: python.exe is locked by the
REM running interpreter, so venvlauncher.exe cannot be copied over it. Worse,
REM `python -m venv` still exits 0, so the failure looks like success.
if defined VIRTUAL_ENV (
    echo ERROR: a virtual environment is already active:
    echo          %VIRTUAL_ENV%
    echo.
    echo        Run  deactivate  first, then re-run this script.
    exit /b 1
)

REM ---- validate the name BEFORE creating anything -------------------------
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

if exist "%VENV%\" (
    echo ERROR: %VENV% already exists.
    echo        Remove it first:   rmdir /s /q %VENV%
    exit /b 1
)

REM ---- pick an interpreter ------------------------------------------------
REM Prefer 3.12: QwenCleo's own README uses it, and ML wheels for the newest
REM Python often lag by months. Python 3.13 in particular has no torch build
REM before 2.6, which is why a 2.5.1 pin cannot resolve there.
set "PY="
if defined PYTAG goto :try_pytag

for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 set "PY=py -%%V"
    )
)
if not defined PY goto :fallback_python
goto :got_python

:try_pytag
py %PYTAG% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: no interpreter for PYTAG=%PYTAG%.  Installed versions:
    py -0
    exit /b 1
)
set "PY=py %PYTAG%"
goto :got_python

:fallback_python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: no Python found on PATH.
    exit /b 1
)
set "PY=python"
echo.
echo WARNING: no Python 3.10-3.12 found; falling back to the default python.
echo          If wheels fail to resolve, install 3.12 and retry:
echo              winget install Python.Python.3.12
echo.

:got_python
for /f "tokens=2" %%V in ('%PY% --version 2^>^&1') do set "PYVER=%%V"
echo ^>^> interpreter: %PY%  ^(Python %PYVER%^)

REM ---- ffmpeg is not a pip package ---------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: ffmpeg not found on PATH. The pipeline cannot decode audio.
    echo          Install it with:  winget install Gyan.FFmpeg
    echo          Then open a NEW cmd window so PATH refreshes.
    echo.
)

echo ^>^> creating %VENV%
%PY% -m venv "%VENV%"

REM `python -m venv` can report a copy failure and still exit 0, so verify the
REM interpreter actually landed rather than trusting the exit code.
if not exist "%VENV%\Scripts\python.exe" (
    echo.
    echo ERROR: venv creation failed -- %VENV%\Scripts\python.exe is missing.
    echo        Usually this means the target was locked by a running process.
    echo        Close any shell using it, then:  rmdir /s /q %VENV%
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet

if /I "%BACKEND%"=="qwencleo"       goto :inst_qwencleo
if /I "%BACKEND%"=="qwen3asr"       goto :inst_qwen3asr
if /I "%BACKEND%"=="cohere"         goto :inst_cohere
if /I "%BACKEND%"=="faster-whisper" goto :inst_whisper

:inst_qwencleo
REM Order matters: torch first so the package cannot re-resolve it, then
REM QwenCleo with --no-deps, then its remaining runtime deps.
REM torch is deliberately NOT pinned. QwenCleo's README suggests 2.5.1, but
REM that version has no wheels for Python 3.13 and none at all on the
REM retired cu121 index. --no-deps means the pin is not enforced anyway.
echo ^>^> installing torch ^(%CUDA%^)
pip install torch torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 goto :failed_torch
echo ^>^> installing qwencleo-asr ^(--no-deps^)
pip install qwencleo-asr --no-deps
if errorlevel 1 goto :failed
echo ^>^> installing remaining deps
pip install -r requirements-qwencleo.txt
if errorlevel 1 goto :failed
goto :verify

:inst_qwen3asr
echo ^>^> installing torch ^(%CUDA%^)
pip install torch torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 goto :failed_torch
pip install -r requirements-qwen3asr.txt
if errorlevel 1 goto :failed
goto :verify

:inst_cohere
echo ^>^> installing torch ^(%CUDA%^)
pip install torch torchaudio --index-url "%TORCH_INDEX%"
if errorlevel 1 goto :failed_torch
pip install -r requirements-cohere.txt
if errorlevel 1 goto :failed
echo.
echo ^>^> NOTE: these weights are gated.
echo    1. accept the terms at
echo       https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026
echo    2. run:  hf auth login
goto :verify

:inst_whisper
REM CTranslate2 based -- no torch needed at all.
pip install -r requirements-whisper.txt
if errorlevel 1 goto :failed
goto :verify

:verify
echo.
echo ^>^> verifying
python craig_transcribe.py --check-env --backend %BACKEND%
echo.
echo Done. Activate with:  %VENV%\Scripts\activate.bat
echo Then:                 python craig_transcribe.py recording.zip -o out --backend %BACKEND%
exit /b 0

:failed_torch
echo.
echo torch install failed. Most likely causes, in order:
echo.
echo   1. The CUDA index has no wheels for Python %PYVER%.
echo      Python 3.13 needs torch 2.6 or newer. Install 3.12 and retry:
echo          winget install Python.Python.3.12
echo          rmdir /s /q %VENV%
echo          set PYTAG=-3.12 ^&^& setup.cmd %BACKEND%
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
echo Install failed. See the pip output above.
exit /b 1

:usage
echo usage: setup.cmd {qwencleo^|qwen3asr^|cohere^|faster-whisper}
echo.
echo   set CUDA=cu128 ^&^& setup.cmd qwencleo     pick the wheel index
echo   set PYTAG=-3.12 ^&^& setup.cmd qwencleo    pick the interpreter
exit /b 1
