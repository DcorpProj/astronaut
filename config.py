# config.py
import json
import os
import threading
from pathlib import Path

import requests


# ════════════════════════════════════════════════════════════
# CONFIG  (leaf module — nothing here imports from web/cli/etc.)
# ════════════════════════════════════════════════════════════

CONFIG = {
    "provider": os.getenv("ASTRONAUT_PROVIDER", "ollama"),
    "model": os.getenv("ASTRONAUT_MODEL", "qwen2.5:7b"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "workspace": os.getenv("ASTRONAUT_WORKSPACE", "./workspace"),
    "mode": "Standard",
    "max_steps": 15,
    "no_history": False,
    "keep_alive": "30m",
    "temperature": 0.4,
    "history_max_messages": 40,
}

BASE_RULES = (
    "You are Astronaut AI. You DO have file system access and MANY tools.\n"
    "You DO have: create_file, open_file, delete_file, list_directory, run_command, "
    "ddg_search, fetch_url, resolve-library-id, query-docs, and others.\n"
    "NEVER say 'I don't have access to tools' — you DO.\n"

    "!!! HOW TO CALL TOOLS !!!\n"
    "  - WRONG: writing Python code like `context7(query=...)` in your text.\n"
    "  - WRONG: writing `import X` or `requests.get(...)` in your text.\n"
    "  - WRONG: calling a tool by its SERVER NAME (context7, tavily, firecrawl) — "
    "those are servers, not tools.\n"
    "  - WRONG: writing 'I will call X' — that does nothing.\n"
    "  - RIGHT: return a REAL TOOL CALL. The framework executes it and returns the result.\n"
    "  - TOOL NAMES are the function names shown in the tool list:\n"
    "      resolve-library-id, query-docs, create_file, ddg_search, fetch_url, etc.\n"
    "  - To use Context7: FIRST call `resolve-library-id(libraryName=...)`, "
    "THEN `query-docs(libraryId=..., query=...)`.\n"
    "  - NEVER invent tool names. NEVER write Python code as a fake call.\n"

    "TOOL ARGUMENTS — STRICT:\n"
    "  - Use ONLY the arguments listed in the tool schema. Never invent new ones.\n"
    "  - fetch_url takes ONLY: url, max_chars. It does NOT save files. "
    "There is no 'destination', 'file', 'save_to', 'output', 'path' argument.\n"
    "  - To save text — call create_file with filename + content. Two separate steps.\n"
    "  - If you're unsure about a tool's arguments, call it with the minimum and read the error.\n"

    "WORKFLOW: 'FIND X AND SAVE TO FILE':\n"
    "  - Step 1: run the search (ddg_search). You now have the data.\n"
    "  - Step 2: IMMEDIATELY call create_file with that data as content. "
    "Copy the ACTUAL text — snippets, titles, URLs — into the content argument.\n"
    "  - Step 3: STOP. Do NOT fetch each URL. Do NOT re-search. The data you "
    "already have IS the result the user wants.\n"
    "  - If the user said 'save to file', they want a snapshot of the search "
    "results, not full-page scrapes of every link.\n"

    "WHEN A VERIFIER SAYS THE FILE IS MISSING:\n"
    "  - Do NOT create an empty file.\n"
    "  - Do NOT invent tool arguments.\n"
    "  - Look UP in the transcript — the data from your previous tool call is there. "
    "Copy it into create_file's content argument.\n"

    "EMPTY CONTENT IS FORBIDDEN:\n"
    "  - Never call create_file with content=\"\" or content=\".\".\n"
    "  - Never call append_to_file with empty content.\n"
    "  - If you truly have nothing to write, say so instead of calling the tool.\n"

    "LIBRARY QUESTIONS:\n"
    "  - For questions about any library/framework, use `resolve-library-id` + `query-docs`.\n"
    "  - Do NOT answer library API questions from memory — APIs change.\n"

    "PLANNING:\n"
    "  - For non-trivial tasks, write a short numbered plan ('Step 1: ...').\n"
    "  - End the plan with '---' on its own line, then execute it.\n"
    "  - Write the plan ONCE. Never repeat it.\n"

    "TOOL USE — STRICT RULES:\n"
    "  1. Text is not action. If you need to create a file, CALL create_file. "
    "Writing the file's content in your response does NOT create the file. "
    "NEVER say 'создан', 'saved', 'file created' unless create_file returned success "
    "IN THIS SAME RESPONSE.\n"
    "  2. Do the MINIMUM number of tool calls needed. Stop as soon as you have enough data.\n"
    "  3. NEVER call the same tool twice with the same (or near-identical) arguments.\n"
    "  4. If a tool fails once, try ONE alternative. If that also fails, STOP and "
    "report the failure honestly.\n"
    "  5. Only tools relevant to the user's request. If the user asks about the web, "
    "do not touch RAG/index/code tools.\n"

    "ANSWER FORMAT:\n"
    "  - Once you have enough information, give the final answer in natural language.\n"
    "  - DO NOT call more tools after the task is complete.\n"
    "  - DO NOT repeat tool calls, step numbers, or intermediate results in the final answer.\n"
    "  - Format final answers in Markdown.\n"

    "HONESTY:\n"
    "  - Never invent facts. If tools returned insufficient data, say so.\n"
    "  - Report failures plainly. Do not pretend you succeeded.\n"

    "CONFIRMATION:\n"
    "  - Only do what the user explicitly asked. Do not commit, push, or delete "
    "unless explicitly requested.\n"
)

PROMPTS = {
    "Standard": BASE_RULES + "\nAll tools are available.",
    "Code": BASE_RULES + "\nCode mode: focus on programming. Use create_file to save code, git_* to commit. Use execute_python for complex logic.",
    "Minimal": "You are Astronaut AI in Minimal mode. Only run_command, list_directory, open_file, create_file, file_info. Be concise, no plans.",
    "Creator": BASE_RULES + "\nCreator mode: all tools + experimental.",
}

PROVIDER_PRESETS = {
    "ollama": {
        "label": "Ollama (local)",
        "needs_key": False,
        "needs_base": False,
        "default_base": "http://localhost:11434",
        "default_models": ["qwen2.5:7b", "qwen3:8b", "granite4.1:8b", "llama3.1:8b", "mistral:7b"],
    },
    "openai": {
        "label": "OpenAI",
        "needs_key": True,
        "needs_base": False,
        "default_base": "https://api.openai.com/v1",
        "default_models": ["gpt-4o-mini", "gpt-4o"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "needs_key": True,
        "needs_base": False,
        "default_base": "https://openrouter.ai/api/v1",
        "default_models": ["anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-70b-instruct"],
    },
    "groq": {
        "label": "Groq",
        "needs_key": True,
        "needs_base": False,
        "default_base": "https://api.groq.com/openai/v1",
        "default_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "together": {
        "label": "Together AI",
        "needs_key": True,
        "needs_base": False,
        "default_base": "https://api.together.xyz/v1",
        "default_models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
    },
    "custom": {
        "label": "Custom (OpenAI-compatible)",
        "needs_key": True,
        "needs_base": True,
        "default_base": "",
        "default_models": [],
    },
}

LIBRARY_HINTS = {
    "pydantic", "fastapi", "flask", "django", "sqlalchemy", "sqlmodel",
    "alembic", "numpy", "pandas", "scipy", "sympy", "sklearn", "scikit-learn",
    "pytorch", "torch", "tensorflow", "keras", "jax",
    "react", "vue", "svelte", "angular", "next.js", "nextjs", "nuxt",
    "tailwind", "tailwindcss", "bootstrap", "material-ui", "mui",
    "express", "nestjs", "fastify", "sveltekit", "remix", "astro",
    "prisma", "drizzle", "mongoose", "sequelize", "typeorm",
    "requests", "httpx", "aiohttp", "urllib3",
    "pytest", "unittest", "nose", "hypothesis",
    "ruff", "black", "isort", "flake8", "mypy", "pyright",
    "poetry", "pipenv", "uv", "conda",
    "vite", "webpack", "rollup", "esbuild", "parcel",
    "lodash", "axios", "zod", "yup", "joi",
    "rust", "cargo", "tokio", "serde", "axum", "actix",
    "django-rest", "drf", "pydantic-ai", "langchain", "llamaindex",
    "ollama", "openai", "anthropic", "gradio", "streamlit",
}


# ════════════════════════════════════════════════════════════
# AGENTS.md — project-level instructions
# ════════════════════════════════════════════════════════════

_AGENTS_MAX_PER_FILE  = 40_000
_AGENTS_MAX_FILES     = 8
_AGENTS_MAX_TOTAL     = 120_000

_AGENTS_LEGACY_NAMES = (
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
    ".aider.conf.md",
)


def _agents_user_global_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home())) / "astronaut"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "astronaut"
    return base / "AGENTS.md"


def _agents_find_in_dir(d: Path):
    for name in _AGENTS_LEGACY_NAMES:
        p = d / name
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def discover_agents_files(workspace: str, cwd: str = None):
    ws = Path(workspace).expanduser().resolve()
    cwd = Path(cwd).expanduser().resolve() if cwd else ws

    found = []
    seen = set()

    def _add(p):
        if not p or p in seen:
            return
        try:
            if p.is_file():
                seen.add(p)
                found.append(p)
        except OSError:
            pass

    _add(_agents_user_global_path())

    chain = []
    cur = cwd
    stop = ws.parent
    while True:
        chain.append(cur)
        if cur == ws or cur == cur.parent:
            break
        if cur == stop:
            break
        cur = cur.parent

    for d in reversed(chain):
        _add(_agents_find_in_dir(d))

    _add(_agents_find_in_dir(ws))

    return found[:_AGENTS_MAX_FILES]


def load_agents_md(workspace: str, cwd: str = None):
    files = discover_agents_files(workspace, cwd)
    if not files:
        return "", []

    ws = Path(workspace).expanduser().resolve()
    parts = []
    meta = []
    total = 0

    for p in files:
        try:
            size = p.stat().st_size
            if size > _AGENTS_MAX_PER_FILE:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        truncated = False
        if total + len(text) > _AGENTS_MAX_TOTAL:
            text = text[: _AGENTS_MAX_TOTAL - total]
            truncated = True
        total += len(text)

        try:
            rel = str(p.relative_to(ws))
        except ValueError:
            rel = p.name

        meta.append({
            "path":      str(p),
            "rel":       rel,
            "size":      size,
            "truncated": truncated,
        })

        parts.append(
            f"--- AGENTS.md: {rel} ---\n"
            f"{text.rstrip()}\n"
            f"--- END AGENTS.md: {rel} ---"
        )

    if not parts:
        return "", []

    block = (
        "[PROJECT INSTRUCTIONS — AGENTS.md]\n"
        "The following files were found in the workspace and describe\n"
        "project conventions, constraints, and gotchas. Treat them as\n"
        "authoritative for this project. They OVERRIDE your general\n"
        "defaults, but never override the user's direct instructions.\n\n"
        + "\n\n".join(parts)
        + "\n[END PROJECT INSTRUCTIONS]"
    )
    return block, meta


def get_system_prompt(mode: str) -> str:
    return PROMPTS.get(mode, PROMPTS["Standard"])


# ════════════════════════════════════════════════════════════
# MODEL CAPABILITIES — dynamic probe via Ollama /api/show
# ════════════════════════════════════════════════════════════

_THINK_CACHE: dict[str, bool] = {}
_THINK_LOCK  = threading.Lock()


def _ollama_base_url() -> str:
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not base.startswith("http"):
        base = "http://" + base
    return base.rstrip("/")


def _query_ollama_thinking(model_name: str):
    """Return True / False / None (None = Ollama unreachable or unclear)."""
    try:
        r = requests.post(
            f"{_ollama_base_url()}/api/show",
            json={"model": model_name},
            timeout=5,
        )
        if r.status_code != 200:
            return None

        data = r.json() or {}

        # 1. Modern Ollama: `capabilities` array
        caps = data.get("capabilities") or []
        if isinstance(caps, list) and caps:
            return "thinking" in [str(c).lower() for c in caps]

        # 2. Some builds expose a `thinking` metadata block
        th = data.get("thinking")
        if isinstance(th, dict):
            values = th.get("values") or []
            for v in values:
                if str(v).lower() in ("true", "1", "low", "medium", "high"):
                    return True
            return False
        if th is True:
            return True
        if th is False:
            return False

        return None
    except Exception:
        return None


def model_supports_thinking(model_name: str) -> bool:
    """Return True if Ollama reports the model supports thinking.

    Falls back to a name-based heuristic only if Ollama is unreachable,
    so offline use degrades to 'qwen3* only' instead of crashing.
    """
    if not model_name:
        return False

    with _THINK_LOCK:
        if model_name in _THINK_CACHE:
            return _THINK_CACHE[model_name]

    supported = _query_ollama_thinking(model_name)
    if supported is None:
        supported = "qwen3" in model_name.lower()

    with _THINK_LOCK:
        _THINK_CACHE[model_name] = supported
    return supported


def invalidate_thinking_cache(model_name: str | None = None):
    """Clear the thinking-capability cache.

    Call with no argument to clear everything (e.g. after switching
    providers), or with a model name to re-probe just that model.
    """
    with _THINK_LOCK:
        if model_name is None:
            _THINK_CACHE.clear()
        else:
            _THINK_CACHE.pop(model_name, None)


# ════════════════════════════════════════════════════════════
# WORKSPACE PERSISTENCE
# ════════════════════════════════════════════════════════════
# If .astronaut_settings.json contains a non-empty "workspace", use it.
# Env var ASTRONAUT_WORKSPACE wins if set explicitly.

def _apply_saved_workspace():
    if os.getenv("ASTRONAUT_WORKSPACE"):
        return
    try:
        p = Path(".astronaut_settings.json")
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        ws = (data.get("workspace") or "").strip()
        if ws:
            CONFIG["workspace"] = ws
    except Exception:
        pass


_apply_saved_workspace()