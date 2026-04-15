"""
config.py — configuration loader.
All settings come from environment variables (loaded from .env).

Supported LLM backends:  gemini | openai | openrouter | groq | ollama
Supported STT backends:  whisper | groq
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root & output directory
ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / os.getenv("OUTPUT_DIR", "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# STT
STT_BACKEND: str = os.getenv("STT_BACKEND", "whisper")   # "whisper" | "groq"
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium/large
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# LLM
# Options: "gemini" | "openai" | "groq" | "ollama"
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")

# Gemini (Google AI Studio)
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# OpenAI (or any OpenAI-compatible endpoint, e.g. OpenRouter free models)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Override base URL to use a free/compatible provider (e.g. OpenRouter)
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")  

# OpenRouter (specific config if needed, but uses OpenAI client)
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "microsoft/wizardlm-2-8x22b")  # example free model

# Groq (free tier — fast hosted LLMs)
GROQ_LLM_MODEL: str = os.getenv("GROQ_LLM_MODEL", "llama3-8b-8192")

# Ollama (local, no API key needed)
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "hf.co/Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF:Q4_0")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")