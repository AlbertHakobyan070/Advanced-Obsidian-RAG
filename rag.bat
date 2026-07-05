@echo off
REM ===========================================================================
REM rag.bat - launcher for Personal RAG
REM   Activates the project venv, cd's to the project, then runs a command.
REM
REM   rag serve              -> start the warm HTTP endpoint on :8051 (for agents)
REM   rag console            -> start the management console on :8052 (browser admin)
REM   rag ocr                -> start the DeepSeek-OCR vision server on :8100
REM   rag web                -> start the Streamlit web app (browser UI)
REM   rag query "..."        -> one-shot CLI query (cold, human use)
REM   rag chat               -> interactive warm CLI loop
REM   rag <anything>         -> forwarded to: python main.py <anything>
REM
REM   Each server BLOCKS its window - run serve / console / ocr in SEPARATE
REM   terminals. On 16GB RAM, only start the ones you actually need right now.
REM ===========================================================================
setlocal

set "VENV=%LOCALAPPDATA%\rag\venv"
set "PROJECT=A:\DS_Vault\DS Main Vault\rag_project"

REM OCR ingest passes need this (non-default Tesseract install path); harmless otherwise.
set "TESSDATA_PREFIX=%LOCALAPPDATA%\Programs\Tesseract-OCR\tessdata"
REM Silence the benign cross-filesystem hardlink warning from uv.
set "UV_LINK_MODE=copy"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [rag.bat] ERROR: venv not found at "%VENV%"
    echo            The project venv may have moved or been reinstalled.
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"
cd /d "%PROJECT%"

if /i "%~1"=="serve" (
    REM Warm endpoint for agents/bots. Port 8051 - NOT 8000 (Jupyter).
    echo [rag.bat] Starting RAG endpoint on http://127.0.0.1:8051 ...
    echo [rag.bat] Leave this window open. Health: curl.exe http://127.0.0.1:8051/health
    python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051
    goto :end
)

if /i "%~1"=="console" (
    REM Management console for corpus admin (ingest/index/delete/jobs). Browser only.
    echo [rag.bat] Starting management console on http://127.0.0.1:8052 ...
    echo [rag.bat] Open that URL in a browser. Leave this window open.
    python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052
    goto :end
)

if /i "%~1"=="ocr" (
    REM DeepSeek-OCR vision server on :8100 (decoder on GPU1). Only needed for
    REM --ocr-engine vlm ingest passes. Every flag here is load-bearing - see
    REM config.yaml pdf.vlm_ocr. Stop it (Ctrl+C) when the OCR pass is done.
    echo [rag.bat] Starting DeepSeek-OCR server on http://127.0.0.1:8100 (GPU1) ...
    echo [rag.bat] Leave open only while OCR-ingesting; it holds ~3.5GB VRAM.
    "A:\Llamacpp\llama-b9860-bin-win-vulkan-x64\llama-server.exe" -m "A:\Llamacpp\models\deepseek-ocr\DeepSeek-OCR-Q8_0.gguf" --mmproj "A:\Llamacpp\models\deepseek-ocr\mmproj-DeepSeek-OCR-Q8_0.gguf" -ngl 999 -dev Vulkan1 -c 8192 -np 1 --host 127.0.0.1 --port 8100 --jinja --chat-template-file "A:\Llamacpp\deepseek-ocr-passthrough.jinja" --flash-attn off
    goto :end
)

if /i "%~1"=="web" (
    REM Streamlit browser UI (preset selector + top-k controls in the sidebar).
    echo [rag.bat] Starting Streamlit app - a browser tab will open ...
    streamlit run app.py
    goto :end
)

REM Everything else is forwarded straight to main.py (query, chat, eval, ingest-*, index).
python main.py %*

:end
endlocal
