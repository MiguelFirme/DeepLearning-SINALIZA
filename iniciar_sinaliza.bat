@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else (
    set "PYTHON_CMD=python"
)

echo Iniciando Backend (FastAPI na porta 8000)...
start "Sinaliza - Backend" cmd /k %PYTHON_CMD% -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

echo Iniciando Frontend (Vite na porta 5173)...
start "Sinaliza - Frontend" /D "%~dp0frontend" cmd /k npm run dev

echo.
echo Servicos iniciados em janelas separadas.
echo Feche as janelas "Sinaliza - Backend" e "Sinaliza - Frontend" para encerrar.
pause