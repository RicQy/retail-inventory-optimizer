#!/bin/bash
# Launch script for running the FastAPI application locally

# Set environment variables
export APP_ENV=test
export PYTHONPATH="${PYTHONPATH}:."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please create one first:"
    echo "python -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Check if we're on Windows or Unix
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "Starting FastAPI application on Windows..."
    .venv/Scripts/python.exe -m uvicorn api.main:app --port 8000 --reload
else
    # Unix/Linux/Mac
    echo "Starting FastAPI application on Unix/Linux/Mac..."
    source .venv/bin/activate
    python -m uvicorn api.main:app --port 8000 --reload
fi
