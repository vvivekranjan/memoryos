from __future__ import annotations
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parents[1] / ".skills" / "SYSTEM_PROMPT.txt"

def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()