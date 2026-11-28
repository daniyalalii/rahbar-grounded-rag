@echo off
:: RAHBAR dev launcher — runs API with real Ollama (deepseek-r1:1.5b)
:: Double-click this file, or run from cmd: start_dev.bat
:: Then open http://localhost:8000

set RETRIEVER=stub
set LLM_PROVIDER=ollama-host
set OLLAMA_URL=http://localhost:11434
set LLM_MODEL=deepseek-r1:1.5b
set EMBED_MODEL=nomic-embed-text
set SERVE_UI_DIR=%~dp0ui
set LOG_LEVEL=INFO

echo Starting RAHBAR with Ollama (deepseek-r1:1.5b)...
echo Open http://localhost:8000 in your browser
echo.

cd /d "%~dp0api"
"%~dp0.venv\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000 --reload
pause

