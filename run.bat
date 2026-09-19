@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Corpus - Local-First Research Engine

if not exist ".venv\Scripts\python.exe" (
  echo Not set up yet - running setup.bat first.
  call setup.bat
  exit /b
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8765"
set "MODEL=qwen3:1.7b"

:: ---- Ollama must be serving on 127.0.0.1:11434; start it if it is not.
call :ollama_ensure || exit /b 1

:: ---- After a reboot Ollama's GPU probe can time out and it silently runs the model on the CPU
::      (5x slower). Load the model, check where it landed, and restart Ollama once if it is CPU-only
::      on a machine that has an NVIDIA GPU.
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :serve
echo Checking that %MODEL% is on the GPU...
.venv\Scripts\python.exe generate.py --gpu-check %MODEL%
if not errorlevel 2 goto :serve
echo Ollama put the model on the CPU. Restarting Ollama so it can find the GPU (takes 1-3 minutes)...
taskkill /f /im ollama.exe >nul 2>&1
taskkill /f /im "ollama app.exe" >nul 2>&1
taskkill /f /im llama-server.exe >nul 2>&1
ping -n 4 127.0.0.1 >nul
call :ollama_ensure || exit /b 1
.venv\Scripts\python.exe generate.py --gpu-check %MODEL%
if errorlevel 2 echo Still on the CPU - answers will be slow. Check "ollama serve" output for "GPU discovery".

:serve
echo Corpus is at http://127.0.0.1:%PORT%   (Ctrl+C or close this window to stop)
.venv\Scripts\python.exe server.py %PORT% --open
exit /b

:: ---- subroutine: make sure an Ollama server answers on the loopback port
:ollama_ensure
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>&1 && exit /b 0
where ollama >nul 2>&1 || (echo Ollama is not installed. Run setup.bat. & pause & exit /b 1)
echo Starting Ollama (its GPU check can take 1-3 minutes after a reboot)...
start "Ollama" /min ollama serve
set /a WAIT=0
:ollama_wait
ping -n 4 127.0.0.1 >nul
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>&1 && exit /b 0
set /a WAIT+=3
if %WAIT% LSS 240 goto :ollama_wait
echo Ollama did not answer on 127.0.0.1:11434 within 4 minutes.
pause
exit /b 1
