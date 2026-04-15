#!/usr/bin/env bash
# setup.sh — One-shot environment setup for the Voice AI Agent.
# Usage: bash setup.sh

set -e
echo ""
echo "════════════════════════════════════════════"
echo "  🎙️  Voice AI Agent — Setup"
echo "════════════════════════════════════════════"
echo ""

# ── Python version check ──────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "✔  Python: $PY_VER ($PYTHON)"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "→  Creating virtual environment (.venv)…"
    $PYTHON -m venv .venv
fi
source .venv/bin/activate
echo "✔  Virtual env activated"

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "→  Installing Python dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "✔  Dependencies installed"

# ── .env ─────────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✔  Created .env from .env.example — EDIT IT before running the app."
else
    echo "✔  .env already exists"
fi

# ── Output dir ────────────────────────────────────────────────────────────────
mkdir -p output
echo "✔  output/ directory ready"

# ── Ollama check ──────────────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    echo "✔  Ollama found: $(ollama --version)"
    MODEL=$(grep OLLAMA_MODEL .env | cut -d= -f2 | tr -d '"')
    MODEL=${MODEL:-mistral}
    echo "→  Pulling Ollama model: $MODEL (this may take a few minutes)…"
    ollama pull "$MODEL" || echo "⚠  Could not pull $MODEL — pull it manually: ollama pull $MODEL"
else
    echo "⚠  Ollama not found. Install from https://ollama.com then run: ollama pull mistral"
    echo "   Alternatively, set LLM_BACKEND=groq in .env to use the Groq cloud API."
fi

echo ""
echo "════════════════════════════════════════════"
echo "  ✅  Setup complete!"
echo ""
echo "  Run the app:"
echo "    source .venv/bin/activate"
echo "    streamlit run app.py"
echo ""
echo "  Test without audio:"
echo "    python test_pipeline.py"
echo ""
echo "  Benchmark models:"
echo "    python benchmark.py --audio your_file.wav"
echo "════════════════════════════════════════════"
echo ""
