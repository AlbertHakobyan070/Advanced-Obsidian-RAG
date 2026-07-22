@echo off
REM ===========================================================================
REM rag.bat - Windows launcher template for the RAG pipeline (Windows only).
REM
REM   This is a CONVENIENCE WRAPPER around the standard Python commands. The
REM   README and MANUAL.md show the exact `python -m uvicorn ...` and
REM   `python main.py ...` commands - those work on every platform with no
REM   wrapper at all. This .bat just reduces typing once you've set up the
REM   paths below.
REM
REM   QUICK START:
REM     1. Edit the VENV and PROJECT lines below to match your install.
REM     2. From a project root that contains this file, type `rag serve`,
REM        `rag console`, `rag query "..."`, etc. (the .bat picks up the
REM        working directory automatically if PROJECT is left blank).
REM
REM   COMMANDS (each BLOCKS its window; run each in a SEPARATE terminal):
REM     rag serve              = start the warm HTTP endpoint on :8051
REM     rag console            = start the management console on :8052 (browser)
REM     rag web                = start the Streamlit web app (browser UI)
REM     rag query "..."        = one-shot CLI query (cold, human use)
REM     rag chat               = interactive warm CLI loop
REM     rag [anything]         = forwarded to: python main.py [anything]
REM
REM   On 16 GB RAM, only start the ones you actually need right now.
REM ===========================================================================
setlocal

REM --- Edit these two lines to match your local install ---
set "VENV=%USERPROFILE%\.venvs\rag"
set "PROJECT="

REM OCR ingest passes need this if Tesseract is installed in a non-default
REM location; harmless if the env var is unset. Adjust for your install.
set "TESSDATA_PREFIX=%LOCALAPPDATA%\Programs\Tesseract-OCR\tessdata"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [rag.bat] ERROR: venv not found at "%VENV%"
    echo            Edit the VENV line at the top of this file, or activate
    echo            your venv manually before running the python commands.
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"
if not "%PROJECT%"=="" cd /d "%PROJECT%"

if /i "%~1"=="serve" (
    echo [rag.bat] Starting RAG endpoint on http://127.0.0.1:8051 ...
    echo [rag.bat] Leave this window open. Health: curl.exe http://127.0.0.1:8051/health
    python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051
    goto :end
)

if /i "%~1"=="console" (
    echo [rag.bat] Starting management console on http://127.0.0.1:8052 ...
    echo [rag.bat] Open that URL in a browser. Leave this window open.
    python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052
    goto :end
)

if /i "%~1"=="web" (
    echo [rag.bat] Starting Streamlit app - a browser tab will open ...
    streamlit run app.py
    goto :end
)

REM Everything else is forwarded straight to main.py (query, chat, eval, ingest-*, index).
python main.py %*

:end
endlocal
