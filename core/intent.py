"""
core/intent.py — Intent classification via LLM.

Supported backends (set LLM_BACKEND in .env):
  gemini  — Google Gemini API (free tier via AI Studio)
  openai  — OpenAI API or any OpenAI-compatible endpoint (OpenRouter, etc.)
  groq    — Groq cloud API (free tier, very fast)
  ollama  — local Ollama server (no API key needed)

Intents
-------
  create_file   — create an empty file or folder
  write_code    — generate and save code to a file
  summarize     — summarise provided text
  chat          — general conversation / Q&A
  unknown       — could not determine intent
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import config


VALID_INTENTS = {"create_file", "write_code", "summarize", "chat", "unknown"}


@dataclass
class ParsedIntent:
    intents: List[str] = field(default_factory=lambda: ["unknown"])
    filename: Optional[str] = None
    language: Optional[str] = None
    description: str = ""
    text_to_summarize: Optional[str] = None
    raw_llm_output: str = ""

    @property
    def primary(self) -> str:
        return self.intents[0] if self.intents else "unknown"

    @property
    def is_compound(self) -> bool:
        return len(self.intents) > 1

# Shared System Prompt

_SYSTEM_PROMPT = """You are an intent-extraction engine for a voice-controlled local AI agent.
The user has spoken a command that has been transcribed to text.

Your job is to analyse the command and return ONLY a valid JSON object with these fields:

{
  "intents": ["<intent1>", "<intent2>"],
  "filename": "<name.ext or null>",
  "language": "<python|js|bash|...>",
  "description": "<one-line description of what the user wants>",
  "text_to_summarize": "<text to summarise, or null>"
}

Valid intents (use exactly these strings):
  create_file   — user wants to create a file or folder
  write_code    — user wants to generate code saved to a file
  summarize     — user wants text summarised
  chat          — general question, greeting, or anything else
  unknown       — truly unintelligible command

Rules:
- Multiple intents are allowed for compound commands.
- If the user says "create a python file with X", use ["write_code"].
- Only use "create_file" alone if no code generation is requested.
- Return ONLY the JSON object — no markdown fences, no extra text.
- DO NOT GENERATE CODE. Only extract intent.
- DO NOT RETURN CODE IN YOUR RESPONSE. Return ONLY the JSON."""


# Backend Implementation

def _call_gemini(user_text: str) -> str:
    """Google Gemini via google-generativeai SDK. Free tier: 15 RPM / 1M TPD."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    response = model.generate_content(user_text)
    return response.text


def _call_openai(user_text: str) -> str:
    """OpenAI or any OpenAI-compatible endpoint (OpenRouter free models, Together AI, etc.)."""
    from openai import OpenAI  # type: ignore

    kwargs: dict = {"api_key": config.OPENAI_API_KEY}
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL
    elif config.LLM_BACKEND == "openrouter":
        kwargs["base_url"] = "https://openrouter.ai/api/v1"

    client = OpenAI(**kwargs)
    model = config.OPENAI_MODEL if config.LLM_BACKEND != "openrouter" else config.OPENROUTER_MODEL
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return response.choices[0].message.content


def _call_groq(user_text: str) -> str:
    """Groq cloud API — free tier with generous limits (6000 RPD on llama3-8b)."""
    from groq import Groq  # type: ignore

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return response.choices[0].message.content


def _call_ollama(user_text: str) -> str:
    """Local Ollama server — completely free, no API key needed."""
    import ollama  # type: ignore

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    return response["message"]["content"]



# router

_BACKEND_MAP = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "openrouter": _call_openai,
    "groq":   _call_groq,
    "ollama": _call_ollama,
}


def _call_llm(user_text: str) -> str:
    backend = config.LLM_BACKEND.lower()
    fn = _BACKEND_MAP.get(backend)
    if fn is None:
        raise ValueError(
            f"Unknown LLM_BACKEND '{backend}'. "
            f"Valid options: {', '.join(_BACKEND_MAP)}"
        )
    return fn(user_text)


# JSON extractor

def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a string (strips markdown fences)."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in LLM output: {raw!r}")


def _fallback_intent_detector(raw: str, original_text: str) -> ParsedIntent:
    """
    Fallback intent detector for when LLM returns raw text instead of JSON.
    Detects intents based on keywords and patterns in the original transcription and response.
    """
    text_lower = original_text.lower()
    response_lower = raw.lower()
    
    # Extracting filename if mentioned  
    filename = None
    filename_match = re.search(r'(?:save|file|write|create)\s+(?:to|a|an|the)?\s+(?:file\s+)?["`]?(\w+\.[\w]+)["`]?', text_lower)
    if filename_match:
        filename = filename_match.group(1)
    
    # Extracting programming language
    language = None
    lang_patterns = {
        'python': r'\b(python|py)\b',
        'js': r'\b(javascript|js)\b',
        'bash': r'\b(bash|shell)\b',
        'java': r'\b(java)\b',
        'cpp': r'\b(c\+\+|cpp)\b',
        'go': r'\b(go|golang)\b',
        'rust': r'\b(rust)\b',
    }
    for lang, pattern in lang_patterns.items():
        if re.search(pattern, text_lower):
            language = lang
            break
    
    # Detecting intent
    intents = []
    
    # Checking for code generation patterns
    code_patterns = [
        r'\b(write|create|generate|implement|code|function|script)\b',
        r'(def |function |class |const |var |let )',
        r'(\#\!|import |from |export )',
    ]
    has_code_request = any(re.search(p, text_lower) or re.search(p, response_lower) for p in code_patterns)
    
    if has_code_request and (filename or language):
        intents.append("write_code")
    elif has_code_request:
        intents.append("write_code")
    
    # Checking for file creation
    if re.search(r'\b(create|make|new)\b.*\b(file|folder|directory)\b', text_lower) and not filename:
        intents.append("create_file")
    
    # Checking for summarizing
    if re.search(r'\b(summarize|summarise|summary|tldr)\b', text_lower):
        intents.append("summarize")
    
    # Checking for chatting
    if re.search(r'\b(hello|hi|hey|what|how|tell|explain|what are|why)\b', text_lower):
        intents.append("chat")
    
    # setting Default to chat if nothing else matched
    if not intents:
        intents = ["chat"]
    
    # Filtering to valid intents
    intents = [i for i in intents if i in VALID_INTENTS]
    if not intents:
        intents = ["chat"]
    
    return ParsedIntent(
        intents=intents,
        filename=filename,
        language=language,
        description=original_text,
        raw_llm_output=raw,
    )


def _parse_response(raw: str, original_text: str) -> ParsedIntent:
    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        # Fallback: trying to detect intent from raw text when JSON parsing fails
        return _fallback_intent_detector(raw, original_text)

    intents = data.get("intents", ["unknown"])
    intents = [i for i in intents if i in VALID_INTENTS] or ["unknown"]

    return ParsedIntent(
        intents=intents,
        filename=data.get("filename"),
        language=data.get("language"),
        description=data.get("description", original_text),
        text_to_summarize=data.get("text_to_summarize"),
        raw_llm_output=raw,
    )


def classify(transcribed_text: str) -> ParsedIntent:
    """
    Classifies the user's intent from transcribed speech.

    Parameters
    ----------
    transcribed_text : str
        Output from the STT step.

    Returns
    -------
    ParsedIntent
        Structured intent object.
    """
    if not transcribed_text or transcribed_text.startswith("[STT Error]"):
        return ParsedIntent(
            intents=["unknown"],
            description="No valid transcription to classify.",
        )

    try:
        raw = _call_llm(transcribed_text)
    except Exception as exc:  # noqa: BLE001
        return ParsedIntent(
            intents=["unknown"],
            description=f"[LLM Error] {exc}",
            raw_llm_output=str(exc),
        )

    return _parse_response(raw, transcribed_text)