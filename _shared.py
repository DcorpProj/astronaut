# plugins/_shared.py
"""Shared helpers for plugins: workspace resolution, safe paths,
and per-model context budgets sourced from Ollama itself."""

import os
from pathlib import Path

from config import CONFIG


# ════════════════════════════════════════════════════════════
# WORKSPACE
# ════════════════════════════════════════════════════════════

def ws() -> Path:
    try:
        p = Path(CONFIG["workspace"]).expanduser().resolve()
    except Exception:
        p = Path(".").resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def safe_path(user_path: str) -> Path:
    root = ws()
    raw = (user_path or "").strip()
    if raw.startswith("./"):
        raw = raw[2:]
    raw = raw.lstrip("/\\")
    if not raw:
        return root
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / raw).resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"Access to {user_path} denied")
    return resolved


def ws_str() -> str:
    return str(ws())


# ════════════════════════════════════════════════════════════
# MODEL CONTEXT — sourced from Ollama, not a table
# ════════════════════════════════════════════════════════════

_DEFAULT_CONTEXT = 8192       # only used if Ollama can't be reached
_CHARS_PER_TOKEN = 3.5        # conservative for code-heavy content
_CTX_CACHE = {}               # model_name -> tokens


def _ollama_base() -> str:
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not base.startswith("http"):
        base = "http://" + base
    return base.rstrip("/")


def _read_active_model() -> str:
    """Model name of the currently active provider, read from disk."""
    try:
        from providers_config import get_active, get_provider_settings
        active = get_active()
        s = get_provider_settings(active)
        m = (s.get("model") or "").strip()
        if m:
            return m
    except Exception:
        pass
    try:
        import json
        p = Path(".astronaut_settings.json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            m = (data.get("model") or "").strip()
            if m:
                return m
    except Exception:
        pass
    return str(CONFIG.get("model") or "")


def get_context_window(model_name: str = None) -> int:
    """Return the context window of `model_name` (or the active model)
    in tokens, as reported by Ollama's /api/show.

    Falls back to 8192 only if Ollama is unreachable or has no answer.
    """
    name = (model_name or _read_active_model() or "").strip()
    if not name:
        return _DEFAULT_CONTEXT

    if name in _CTX_CACHE:
        return _CTX_CACHE[name]

    ctx = _query_ollama_context(name)
    if ctx:
        _CTX_CACHE[name] = ctx
        return ctx

    return _DEFAULT_CONTEXT


def _query_ollama_context(model_name: str) -> int:
    """Ask Ollama how big this model's context window is.

    Ollama's /api/show returns a `model_info` dict with keys like
    "qwen2.context_length", "llama.context_length", "general.context_length".
    Different model families use different prefixes, so we match on the
    suffix rather than a specific key.
    """
    try:
        import requests
        r = requests.post(
            f"{_ollama_base()}/api/show",
            json={"model": model_name},
            timeout=5,
        )
        if r.status_code != 200:
            return 0

        data = r.json() or {}
        info = data.get("model_info") or {}

        # Preferred: any "*context_length" key
        for k, v in info.items():
            if k.endswith(".context_length") and isinstance(v, int) and v > 0:
                return v

        # Some builds use "context_length" without a prefix
        v = info.get("context_length")
        if isinstance(v, int) and v > 0:
            return v

        # Older builds put it under "parameters"
        params = data.get("parameters") or {}
        if isinstance(params, dict):
            for k in ("num_ctx", "context_length"):
                v = params.get(k)
                if v is not None:
                    try:
                        v = int(v)
                        if v > 0:
                            return v
                    except Exception:
                        pass

        return 0
    except Exception:
        return 0


def context_budget(model_name: str = None,
                   fraction: float = 0.30,
                   max_chars: int = 200_000,
                   min_chars: int = 1024) -> int:
    """Character budget for a single tool result.

    `fraction` = share of the context window one tool result may take.
    Converts tokens → chars using 3.5 chars/token (conservative for code).
    """
    tokens = get_context_window(model_name)
    chars = int(tokens * fraction * _CHARS_PER_TOKEN)
    return max(min_chars, min(chars, max_chars))


def truncate_for_model(text: str,
                       model_name: str = None,
                       fraction: float = 0.30,
                       label: str = "truncated") -> str:
    """Truncate `text` to the active model's tool-result budget.

    Cuts at a newline near the limit. Appends a note so the model
    (and any human reading the log) knows the content was clipped.
    """
    if not text:
        return text
    budget = context_budget(model_name, fraction=fraction)
    if len(text) <= budget:
        return text

    head = text[:budget]
    cut = head.rfind("\n")
    if cut > budget * 0.85:
        head = head[:cut]

    return (head
            + f"\n... ({label}; showed {len(head)} of {len(text)} chars, "
            f"budget {budget})")


def invalidate_context_cache(model_name: str = None):
    """Clear the cached context length.

    Call this after pulling a model or changing its num_ctx parameter.
    With no argument, clears the entire cache.
    """
    if model_name is None:
        _CTX_CACHE.clear()
    else:
        _CTX_CACHE.pop(model_name.strip(), None)