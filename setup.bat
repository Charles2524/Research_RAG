@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Corpus setup

echo.
echo  ==============================================================
echo   Corpus - Local-First Research Engine : one-time setup
echo  ==============================================================
echo.
echo   This installs everything the app needs on this PC:
echo     1. Python packages into .venv          (~1 GB, PyTorch)
echo     2. Ollama and the qwen3:1.7b model     (1.9 GB)
echo     3. The embedding + reranking models    (~220 MB)
echo   Mostly download time: 10-25 minutes on a normal connection.
echo   Safe to run again: finished steps are skipped.
echo.

:: ---------------------------------------------------------------- 1. Python 3.12
echo [1/8] Python 3.12
set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if defined PY goto :py_ok
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1 && set "PY=python"
if defined PY goto :py_ok
echo.
echo   Python 3.12 was not found. Install it from https://www.python.org/downloads/
echo   and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
start "" https://www.python.org/downloads/
goto :fail
:py_ok
echo       using: %PY%

:: ---------------------------------------------------------------- 2. venv + packages
echo [2/8] Python packages
if exist ".venv\Scripts\python.exe" goto :venv_ok
echo       creating .venv
%PY% -m venv .venv || goto :fail
:venv_ok
set "VPY=.venv\Scripts\python.exe"
%VPY% -c "import torch, sentence_transformers, streamlit, pymupdf4llm, sqlite_vec, yaml, dotenv" >nul 2>&1
if not errorlevel 1 (echo       already installed & goto :pkgs_ok)
echo       installing requirements.txt - this downloads ~1 GB and can take 5-15 minutes
%VPY% -m pip install --upgrade pip --quiet || goto :fail
%VPY% -m pip install -r requirements.txt || goto :fail
:pkgs_ok

:: ---------------------------------------------------------------- 3. VC++ runtime (PyMuPDF)
echo [3/8] Visual C++ runtime
%VPY% -c "import pymupdf" >nul 2>&1
if not errorlevel 1 goto :vc_ok
echo.
echo   The PDF parser needs the Microsoft Visual C++ 2015-2022 x64 runtime.
echo   Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe then press a key here.
start "" https://aka.ms/vs/17/release/vc_redist.x64.exe
pause >nul
%VPY% -c "import pymupdf" >nul 2>&1
if errorlevel 1 (echo   still failing: install the runtime and run setup.bat again & goto :fail)
:vc_ok
echo       ok

:: ---------------------------------------------------------------- 4. .env
echo [4/8] Settings file (.env)
if not exist ".env" copy /y ".env.example" ".env" >nul
%VPY% -c "import re,sys;t=open('.env',encoding='utf-8').read();m=re.search(r'^CONTACT_EMAIL=(.*)$',t,re.M);sys.exit(0 if m and m.group(1).strip() and m.group(1).strip()!='you@example.com' else 1)" >nul 2>&1
if not errorlevel 1 goto :env_ok
echo       The scholarly APIs ^(OpenAlex, Crossref, ...^) require a contact email in their requests.
set "EMAIL="
set /p "EMAIL=      Your email address: "
if "%EMAIL%"=="" (echo   an email address is required & goto :fail)
%VPY% -c "import re,sys;p='.env';t=open(p,encoding='utf-8').read();e=sys.argv[1].strip();t=re.sub(r'^CONTACT_EMAIL=.*$','CONTACT_EMAIL='+e,t,flags=re.M) if re.search(r'^CONTACT_EMAIL=',t,re.M) else t+'\nCONTACT_EMAIL='+e+'\n';open(p,'w',encoding='utf-8').write(t)" "%EMAIL%" || goto :fail
:env_ok
echo       ok

:: ---------------------------------------------------------------- 5. Ollama
echo [5/8] Ollama (runs the language model locally)
set /a TRIES=0
:ollama_check
where ollama >nul 2>&1
if not errorlevel 1 goto :ollama_found
set /a TRIES+=1
if %TRIES% GTR 3 (echo   Ollama still not found on PATH. Install it and run setup.bat again. & goto :fail)
echo.
echo   Ollama is not installed. Download and run the installer from https://ollama.com/download
echo   then press a key here to continue.
start "" https://ollama.com/download
pause >nul
goto :ollama_check
:ollama_found
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>&1
if not errorlevel 1 goto :ollama_up
echo       starting the Ollama server
start "Ollama" /min ollama serve
set /a WAIT=0
:ollama_wait
timeout /t 3 /nobreak >nul
curl -s -m 3 http://127.0.0.1:11434/api/tags >nul 2>&1 && goto :ollama_up
set /a WAIT+=3
if %WAIT% LSS 120 goto :ollama_wait
echo   Ollama did not answer on 127.0.0.1:11434 within 2 minutes. Start it with "ollama serve" and run setup.bat again.
goto :fail
:ollama_up
echo       ok

:: ---------------------------------------------------------------- 6. models
echo [6/8] Language models
echo       ollama pull qwen3:1.7b  (1.9 GB, skipped if already present)
ollama pull qwen3:1.7b || goto :fail
set "PULL4B=n"
set /p "PULL4B=      Also pull qwen3:4b? Slower 'reasons first' model, 2.5 GB, optional [y/N]: "
if /i "%PULL4B%"=="y" (ollama pull qwen3:4b || goto :fail)

:: ---------------------------------------------------------------- 7. embedding + reranking models
echo [7/8] Embedding and reranking models (~220 MB from Hugging Face, once)
%VPY% -m fetch --models || goto :fail

:: ---------------------------------------------------------------- 8. verify + shortcut
echo [8/8] Verifying the installation
%VPY% -m pytest tests\test_phase0.py -q -p no:cacheprovider >nul 2>&1
if errorlevel 1 (echo   The phase 0 checks failed. Run: .venv\Scripts\python -m pytest tests\test_phase0.py & goto :fail)
%VPY% -c "import config,generate;generate.ensure_model(config.load_config(),'qwen3:1.7b')" || goto :fail
echo       ok

:: the real Desktop folder may be redirected (OneDrive), so let the shell resolve it
set "VBS=%TEMP%\corpus_shortcut.vbs"
> "%VBS%" echo Set s = CreateObject("WScript.Shell")
>>"%VBS%" echo p = s.SpecialFolders("Desktop") ^& "\Corpus.lnk"
>>"%VBS%" echo Set fso = CreateObject("Scripting.FileSystemObject")
>>"%VBS%" echo If fso.FileExists(p) Then WScript.Quit 3
>>"%VBS%" echo Set l = s.CreateShortcut(p)
>>"%VBS%" echo l.TargetPath = "%CD%\run.bat"
>>"%VBS%" echo l.WorkingDirectory = "%CD%"
>>"%VBS%" echo l.IconLocation = "%SystemRoot%\System32\shell32.dll,14"
>>"%VBS%" echo l.Description = "Corpus - Local-First Research Engine"
>>"%VBS%" echo l.Save
cscript //nologo "%VBS%" >nul 2>&1
if errorlevel 3 (echo       desktop shortcut already exists) else if errorlevel 1 (echo       could not create the desktop shortcut - use run.bat) else (echo       desktop shortcut created: Corpus)
del "%VBS%" >nul 2>&1

echo.
echo  ==============================================================
echo   Setup complete.
echo   Start the app with run.bat or the Corpus shortcut on the desktop.
echo   The library starts empty: click "Add Papers" in the app to
echo   search, download and index open-access papers on any topic.
echo  ==============================================================
echo.
set "RUNNOW=y"
set /p "RUNNOW=  Start it now? [Y/n]: "
if /i not "%RUNNOW%"=="n" call run.bat
exit /b 0

:fail
echo.
echo   Setup stopped. Fix the problem above and run setup.bat again; finished steps are skipped.
pause
exit /b 1
