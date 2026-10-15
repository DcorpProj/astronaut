# settings.py
"""Global runtime settings for Astronaut AI (stored in .astronaut_settings.json)."""

import json
from pathlib import Path

SETTINGS_PATH = Path(".astronaut_settings.json")

DEFAULTS = {
    "system_prompt": "",
    "temperature": 0.4,
    "max_steps": 15,
    "keep_alive": "30m",
    "history_max_messages": 40,
    "ai_titles": True,
    "stream_chunks": True,
    "disabled_tools": [],
    # --- self-verification ---
    "verification_enabled": True,
    "max_verification_retries": 3,
    "min_verification_attempts": 2,
    "verify_trivial_skip": True,
    # --- thinking ---
    "thinking_mode": "auto",
    # --- planning ---
    "planning_enabled": False,
    # --- workspace ---
    "workspace": "",
}


def load():
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return {**DEFAULTS, **data}
        except Exception:
            pass
    return dict(DEFAULTS)


def save(data):
    merged = {**DEFAULTS, **(data or {})}
    SETTINGS_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return merged