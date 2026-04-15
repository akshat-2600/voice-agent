"""
core/stt.py — Speech-to-Text engine.

Supports two backends:
  1. whisper  — runs OpenAI Whisper locally
  2. groq     — uses the Groq cloud API 
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Union

import config

# Whisper (local)

_whisper_model = None 

def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper  
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)
    return _whisper_model


def _transcribe_whisper(audio_bytes: bytes, file_ext: str = ".wav") -> str:
    model = _load_whisper()
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    result = model.transcribe(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return result["text"].strip()



# Groq (cloud API)

def _transcribe_groq(audio_bytes: bytes, file_ext: str = ".wav") -> str:
    from groq import Groq  

    client = Groq(api_key=config.GROQ_API_KEY)
    filename = f"audio{file_ext}"
    transcription = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        response_format="text",
    )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


# Public API

def transcribe(audio_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Convert raw audio bytes to text using the configured backend.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data (WAV or MP3).
    file_ext : str
        File extension hint ('.wav' or '.mp3').

    Returns
    -------
    str
        Transcribed text, or an error message prefixed with '[STT Error]'.
    """
    if not audio_bytes:
        return "[STT Error] No audio data received."

    try:
        if config.STT_BACKEND == "groq":
            return _transcribe_groq(audio_bytes, file_ext)
        else:
            return _transcribe_whisper(audio_bytes, file_ext)
    except Exception as exc:  # noqa: BLE001
        return f"[STT Error] {exc}"