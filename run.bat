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

:: Ollama must be serving on 127.0.0.1:11434; start it if it is not.
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
  where ollama >nul 2>&1 || (echo Ollama is not installed. Run setup.bat. & pause & exit /b 1)
  echo Starting Ollama...
  start "Ollama" /min ollama serve
)

echo Corpus -> http://127.0.0.1:%PORT%   (Ctrl+C or close this window to stop)
.venv\Scripts\python.exe server.py %PORT% --open
