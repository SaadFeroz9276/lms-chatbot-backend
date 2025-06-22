@echo off
title LMS Chatbot Auto Launcher

echo ===========================================
echo        Setting up LMS Chatbot (Backend)
echo ===========================================

REM Activate Python virtual environment (or create if not present)
IF NOT EXIST venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Installing Python dependencies...
pip install --upgrade pip
pip install fastapi uvicorn langchain langchain-community langchain-core huggingface-hub pydantic

echo Starting RAG API using Uvicorn...
start "LMS Chatbot API" cmd /k "uvicorn rag_api:app --reload"

echo ===========================================
echo       Starting Ollama (Mistral LLM)
echo ===========================================

REM Check if Ollama is installed manually first.
echo Make sure Ollama is already running and mistral is pulled:
echo   ollama pull mistral
timeout /t 3

echo ===========================================
echo     Launching Frontend React App
echo ===========================================

cd frontend\chatbot-ui

IF NOT EXIST node_modules (
    echo Installing frontend dependencies...
    call npm install
)

echo Starting React frontend...
start "LMS Chatbot UI" cmd /k "npm start"

echo ===========================================
echo All services started. Access the chatbot at:
echo http://localhost:3000
echo ===========================================
pause
