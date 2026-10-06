@echo off
rem ===================================================================
rem  NeuroBlocks - double-click to start the block editor (Windows)
rem  First run: creates a private Python environment and installs the
rem  packages it needs (PyTorch is reused if you already have it).
rem ===================================================================
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto run

where python >nul 2>nul || goto nopython
echo Setting up NeuroBlocks for the first time - this can take a few minutes...
python -m venv .venv --system-site-packages || goto nopython
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -c "import torch" 2>nul || .venv\Scripts\python.exe -m pip install torch
.venv\Scripts\python.exe -m pip install -e . || goto failed

:run
echo Starting NeuroBlocks... (close this window to quit)
.venv\Scripts\python.exe -m neuroblocks gui %*
goto :eof

:nopython
echo.
echo Python 3.10 or newer is needed.
echo Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
echo then double-click start.bat again.
pause
goto :eof

:failed
echo.
echo Installing the packages failed - see the messages above.
pause
