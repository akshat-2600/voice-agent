"""
core/memory.py — In session memory / action history.

Stores a chronological log of every pipeline run within the current
Streamlit session. 
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class HistoryEntry:
    timestamp: str
    transcription: str
    intents: List[str]
    description: str
    actions: List[str]
    outputs: List[str]
    file_paths: List[Optional[str]]
    success: bool

    @classmethod
    def build(cls, transcription, intent, results) -> "HistoryEntry":
        from tools.executor import ActionResult  

        return cls(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            transcription=transcription,
            intents=intent.intents,
            description=intent.description,
            actions=[r.action_taken for r in results],
            outputs=[r.output for r in results],
            file_paths=[str(r.file_path) if r.file_path else None for r in results],
            success=all(r.success for r in results),
        )


def get_history(session_state) -> List[HistoryEntry]:
    if "_agent_history" not in session_state:
        session_state["_agent_history"] = []
    return session_state["_agent_history"]


def append(session_state, entry: HistoryEntry) -> None:
    get_history(session_state).append(entry)


def clear(session_state) -> None:
    session_state["_agent_history"] = []