"""
tools/executor.py — Tools execution layer.

Uses the same LLM backend configured in .env (gemini | openai | groq | ollama)
for code generation, summarisation, and chat.

Safety: ALL file I/O is rooted at config.OUTPUT_DIR.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from core.intent import ParsedIntent


# Result type

@dataclass
class ActionResult:
    success: bool
    action_taken: str
    output: str
    file_path: Optional[Path] = None
    error: Optional[str] = None


# Unified LLM caller (mirrors core/intent.py router)

def _llm(prompt: str) -> str:
    """Sending a plain prompt to whichever LLM backend is configured."""
    backend = config.LLM_BACKEND.lower()

    if backend == "gemini":
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=config.GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config=genai.GenerationConfig(temperature=0.2),
        )
        return model.generate_content(prompt).text.strip()

    if backend in ("openai", "openrouter"):
        from openai import OpenAI  # type: ignore
        kwargs: dict = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        elif backend == "openrouter":
            kwargs["base_url"] = "https://openrouter.ai/api/v1"
        client = OpenAI(**kwargs)
        model = config.OPENROUTER_MODEL if backend == "openrouter" else config.OPENAI_MODEL
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    if backend == "groq":
        from groq import Groq  # type: ignore
        client = Groq(api_key=config.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=config.GROQ_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    if backend == "ollama":
        import ollama  # type: ignore
        resp = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"].strip()

    raise ValueError(f"Unknown LLM_BACKEND '{backend}'.")


# Helpers

def _safe_path(filename: str) -> Path:
    safe = (config.OUTPUT_DIR / filename).resolve()
    if not str(safe).startswith(str(config.OUTPUT_DIR.resolve())):
        raise ValueError(f"Path traversal detected: {filename!r}")
    return safe


def _default_filename(language: Optional[str]) -> str:
    ext_map = {
        "python": ".py", "py": ".py",
        "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts",
        "bash": ".sh", "shell": ".sh",
        "html": ".html", "css": ".css",
        "java": ".java", "c": ".c",
        "cpp": ".cpp", "c++": ".cpp",
        "go": ".go", "rust": ".rs", "sql": ".sql",
    }
    ext = ext_map.get((language or "").lower(), ".py")
    return f"output{ext}"


def _generate_code(transcribed_text: str, language: Optional[str], description: str) -> str:
    lang = language or "python"
    prompt = (
        f"Generate clean, well-commented {lang} code that fulfils this request:\n"
        f"{description or transcribed_text}\n\n"
        "Return ONLY the code — no markdown fences, no explanation."
    )
    return _llm(prompt)


def _generate_summary(text: str) -> str:
    prompt = f"Summarise the following text concisely in 3-5 sentences:\n\n{text}"
    return _llm(prompt)


def _chat_response(transcribed_text: str) -> str:
    return _llm(transcribed_text)


# Tool functions

def tool_create_file(intent: ParsedIntent, _text: str) -> ActionResult:
    filename = intent.filename or "new_file.txt"
    try:
        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
            msg = f"Created empty file: `{path.name}`"
        else:
            msg = f"File already exists: `{path.name}` (no changes made)"
        return ActionResult(success=True, action_taken=f"Create file → {path.name}",
                            output=msg, file_path=path)
    except Exception as exc:
        return ActionResult(success=False, action_taken="Create file", output="", error=str(exc))


def tool_write_code(intent: ParsedIntent, transcribed_text: str) -> ActionResult:
    filename = intent.filename or _default_filename(intent.language)
    try:
        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        code = _generate_code(transcribed_text, intent.language, intent.description)
        path.write_text(code, encoding="utf-8")
        return ActionResult(success=True, action_taken=f"Generate & save code → {path.name}",
                            output=code, file_path=path)
    except Exception as exc:
        return ActionResult(success=False, action_taken="Write code", output="", error=str(exc))


def tool_summarize(intent: ParsedIntent, transcribed_text: str) -> ActionResult:
    source = intent.text_to_summarize or transcribed_text
    summary = _generate_summary(source)
    file_path = None
    if intent.filename:
        try:
            file_path = _safe_path(intent.filename)
            file_path.write_text(summary, encoding="utf-8")
        except Exception:
            pass
    action = "Summarise text" + (f" → saved to {file_path.name}" if file_path else "")
    return ActionResult(success=True, action_taken=action, output=summary, file_path=file_path)


def tool_chat(intent: ParsedIntent, transcribed_text: str) -> ActionResult:
    response = _chat_response(transcribed_text)
    return ActionResult(success=True, action_taken="General chat", output=response)


def tool_unknown(_intent: ParsedIntent, _text: str) -> ActionResult:
    return ActionResult(
        success=False,
        action_taken="No action (unknown intent)",
        output="I couldn't understand your command. Please try again with a clearer request.",
        error="Intent not recognised",
    )


# Dispatcher

_TOOL_MAP = {
    "create_file": tool_create_file,
    "write_code":  tool_write_code,
    "summarize":   tool_summarize,
    "chat":        tool_chat,
    "unknown":     tool_unknown,
}


def execute(intent: ParsedIntent, transcribed_text: str) -> list[ActionResult]:
    """Execute all tools for the detected intents (supports compound commands)."""
    return [_TOOL_MAP.get(name, tool_unknown)(intent, transcribed_text)
            for name in intent.intents]