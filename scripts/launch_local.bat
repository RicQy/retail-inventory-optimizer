@echo off
REM Launch script for running the FastAPI application locally on Windows

echo Setting environment variables...
set APP_ENV=test
set PYTHONPATH=%PYTHONPATH%;.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Please create one first:
    echo python -m venv .venv
    echo .venv\Scripts\activate
    echo pip install -r requirements.txt
    exit /b 1
)

echo Starting FastAPI application...
.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000 --reload
