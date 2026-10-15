# providers_config.py
"""Store provider settings (including API keys) in a local JSON file."""

import json
import os
from pathlib import Path

CONFIG_PATH = Path(".astronaut_providers.json")


def _load():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data):
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_providers():
    """Return dict of provider_name -> {api_key, base_url, model, models}."""
    return _load()


def get_active():
    data = _load()
    return data.get("_active", "ollama")


def set_active(name):
    data = _load()
    data["_active"] = name
    _save(data)


def save_provider(name, api_key=None, base_url=None, model=None, models=None):
    data = _load()
    entry = data.get(name, {})
    if api_key is not None:
        entry["api_key"] = api_key
    if base_url is not None:
        entry["base_url"] = base_url
    if model is not None:
        entry["model"] = model
    if models is not None:
        entry["models"] = models
    data[name] = entry
    _save(data)


def get_provider_settings(name):
    return _load().get(name, {})


def delete_provider(name):
    data = _load()
    data.pop(name, None)
    _save(data)