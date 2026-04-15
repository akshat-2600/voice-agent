"""
core/pipeline.py — End-to-end pipeline orchestrator.

Wires together:
  STT → Intent classification → Tool execution → History logging

Yields incremental status updates as a generator so the UI can show
live progress (Streamlit status/spinner pattern).
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

from core import stt, intent, memory
from tools.executor import execute, ActionResult
from core.intent import ParsedIntent


def run(
    audio_bytes: bytes,
    file_ext: str,
    session_state,
    confirmed: bool = True,       # Human-in-the-loop flag
) -> Generator[dict, None, None]:
    """
    Runs the full pipeline.

    Yields dicts with keys: stage, data.

    Stages: transcribing -> classifying -> confirming -> executing -> done | error
    """
    # 1. Speech To TEXT
    yield {"stage": "transcribing", "data": None}
    transcription = stt.transcribe(audio_bytes, file_ext)
    yield {"stage": "transcribed", "data": transcription}

    if transcription.startswith("[STT Error]"):
        yield {"stage": "error", "data": transcription}
        return

    # 2. Intent classification 
    yield {"stage": "classifying", "data": None}
    parsed: ParsedIntent = intent.classify(transcription)
    yield {"stage": "classified", "data": parsed}

    # 3. Human-in-the-loop
    if not confirmed:
        # Caller must re-invoke with confirmed=True after user approves
        yield {"stage": "awaiting_confirmation", "data": parsed}
        return

    # 4. Tool execution
    yield {"stage": "executing", "data": None}
    results: list[ActionResult] = execute(parsed, transcription)
    yield {"stage": "executed", "data": results}

    # 5. Memory
    entry = memory.HistoryEntry.build(transcription, parsed, results)
    memory.append(session_state, entry)

    yield {"stage": "done", "data": {"transcription": transcription, "intent": parsed, "results": results}}