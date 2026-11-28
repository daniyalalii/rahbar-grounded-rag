# RAHBAR dev server — real Ollama mode
# Models used: nomic-embed-text (embedding, stub retriever) + deepseek-r1:1.5b (LLM)
# Usage: .\start_dev.ps1
# Then open: http://localhost:8000

$env:RETRIEVER      = "stub"           # stub retriever (no Chroma needed locally)
$env:LLM_PROVIDER   = "ollama-host"    # real Ollama on localhost
$env:OLLAMA_URL     = "http://localhost:11434"
$env:LLM_MODEL      = "deepseek-r1:1.5b"
$env:EMBED_MODEL    = "nomic-embed-text"
$env:SERVE_UI_DIR   = "$PSScriptRoot\ui"
$env:LOG_LEVEL      = "INFO"

Write-Host "Starting RAHBAR with Ollama (deepseek-r1:1.5b)..." -ForegroundColor Cyan
Write-Host "Open http://localhost:8000 in your browser" -ForegroundColor Green

Set-Location "$PSScriptRoot\api"
& "$PSScriptRoot\.venv\Scripts\uvicorn.exe" app.main:app --host 0.0.0.0 --port 8000 --reload
