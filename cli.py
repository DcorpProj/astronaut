# cli.py
"""Astronaut AI — engine: agent loop, verifier, commands, terminal REPL."""

import json
import os
import pathlib
import re
import sys
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from config import CONFIG, get_system_prompt, PROVIDER_PRESETS, LIBRARY_HINTS
from config import model_supports_thinking
from providers import get_provider
from history import (
    list_sessions, get_meta, update_meta, delete_session,
    create_session, append_message, load_messages, clear_messages,
    get_current_session, set_current_session,
)
import settings as settings_store
from providers_config import (
    get_active, set_active, get_provider_settings, save_provider,
)
from mcp_client import get_manager as get_mcp_manager

import repomap as _repomap
import planning as _planning


# ============================================================
# AGGRESSIVE LEAKED TOOL-CALL PARSER
# ============================================================

_TOOL_CALL_TAGGED = re.compile(
    r'<\s*tool[_\s]*call\s*>(.*?)<\s*/\s*tool[_\s]*call\s*>',
    re.DOTALL | re.IGNORECASE,
)

_FUNCTION_CALL_TAGGED = re.compile(
    r'<\s*function[_\s]*call\s*>(.*?)<\s*/\s*function[_\s]*call\s*>',
    re.DOTALL | re.IGNORECASE,
)

_TOOLS_TAGGED = re.compile(
    r'<\s*tools\s*>(.*?)<\s*/\s*tools\s*>',
    re.DOTALL | re.IGNORECASE,
)

_BARE_JSON_CALL = re.compile(
    r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,[^{}]*"arguments"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)

_QWEN3_FUNC_POS = re.compile(
    r'<\s*function\s*=\s*([^\s>]+)\s*>(.*?)<\s*/\s*function\s*>',
    re.DOTALL | re.IGNORECASE,
)
_QWEN3_FUNC_ATTR = re.compile(
    r'<\s*function\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)<\s*/\s*function\s*>',
    re.DOTALL | re.IGNORECASE,
)
_QWEN3_PARAM_POS = re.compile(
    r'<\s*parameter\s*=\s*([^\s>]+)\s*>(.*?)<\s*/\s*parameter\s*>',
    re.DOTALL | re.IGNORECASE,
)
_QWEN3_PARAM_ATTR = re.compile(
    r'<\s*parameter\s+name\s*=\s*["\']([^"\']+)["\']\s*>(.*?)<\s*/\s*parameter\s*>',
    re.DOTALL | re.IGNORECASE,
)


# ============================================================
# FILE ATTACHMENTS
# ============================================================

_FILE_ATTACH_RE = re.compile(r'!-\s*([^\n\r]+?)\s*-!')
_ATTACH_OPEN = "!-"
_ATTACH_CLOSE = "-!"
_ATTACH_MAX_BYTES = 200_000


def _resolve_attach_path(raw: str, workspace: Path) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = workspace / p
    return p.resolve()


def _expand_file_attachments(text: str, workspace: Path,
                             max_bytes: int = _ATTACH_MAX_BYTES):
    if not text or _ATTACH_OPEN not in text or _ATTACH_CLOSE not in text:
        return text, []
    attached = []

    def _repl(m):
        raw = m.group(1).strip()
        if not raw:
            return m.group(0)
        try:
            p = _resolve_attach_path(raw, workspace)
        except Exception as e:
            return f"[attachment error: bad path '{raw}': {e}]"
        try:
            if not p.exists():
                return f"[attachment error: not found: {raw}]"
            if not p.is_file():
                return f"[attachment error: not a file: {raw}]"
            size = p.stat().st_size
            if size > max_bytes:
                return (f"[attachment error: too large "
                        f"({size} bytes > {max_bytes}): {raw}]")
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return f"[attachment error: cannot read '{raw}': {e}]"
            attached.append(str(p))
            lang = (p.suffix.lstrip(".").lower() or "text")
            return (
                f"\n--- ATTACHED FILE: {p.name} ({size} bytes) ---\n"
                f"```{lang}\n{content}\n```\n"
                f"--- END ATTACHED FILE: {p.name} ---\n"
            )
        except Exception as e:
            return f"[attachment error: '{raw}': {e}]"

    expanded = _FILE_ATTACH_RE.sub(_repl, text)
    return expanded, attached


def _list_attachment_markers(text: str):
    if not text or _ATTACH_OPEN not in text or _ATTACH_CLOSE not in text:
        return []
    return [m.group(1).strip() for m in _FILE_ATTACH_RE.finditer(text)]


# ============================================================
# MISC HELPERS
# ============================================================

def _coerce_scalar(v: str):
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_qwen3_xml_block(block: str):
    if not block:
        return None
    m = _QWEN3_FUNC_ATTR.search(block)
    if not m:
        m = _QWEN3_FUNC_POS.search(block)
    if not m:
        return None
    name = m.group(1).strip()
    body = m.group(2)
    args = {}
    for pm in _QWEN3_PARAM_ATTR.finditer(body):
        args[pm.group(1).strip()] = _coerce_scalar(pm.group(2))
    if not args:
        for pm in _QWEN3_PARAM_POS.finditer(body):
            args[pm.group(1).strip()] = _coerce_scalar(pm.group(2))
    if not name:
        return None
    return {"name": name, "arguments": args}


def _parse_qwen25coder_tools_block(content: str) -> list:
    found = []
    for m in _TOOLS_TAGGED.finditer(content):
        blob = m.group(1).strip()
        if not blob:
            continue
        data = None
        try:
            data = json.loads(blob)
        except Exception:
            try:
                data = [json.loads(line) for line in blob.splitlines()
                        if line.strip()]
            except Exception:
                data = None
        if data is None:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("function")
            args = item.get("arguments") or item.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            if name and isinstance(name, str) and isinstance(args, dict):
                call = {"name": name, "arguments": args}
                if call not in found:
                    found.append(call)
    return found


def _extract_json_call(blob: str):
    blob = blob.strip()
    if not blob:
        return None
    blob = re.sub(r'</?\s*tool[_\s]*call\s*>', '', blob, flags=re.IGNORECASE)
    blob = re.sub(r'</?\s*function[_\s]*call\s*>', '', blob, flags=re.IGNORECASE)
    start = blob.find("{")
    if start < 0:
        return None
    end = blob.rfind("}")
    if end <= start:
        return None
    candidate = blob[start:end + 1]
    try:
        data = json.loads(candidate)
    except Exception:
        try:
            fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
            fixed = fixed.replace("'", '"')
            data = json.loads(fixed)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    name = data.get("name") or data.get("function") or data.get("tool")
    args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
    if isinstance(name, dict):
        inner = name
        name = inner.get("name")
        args = inner.get("arguments") or args
    if not name or not isinstance(name, str):
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return {"name": name, "arguments": args}


def _parse_leaked_tool_calls(content: str) -> list:
    if not content:
        return []
    lowered = content.lower()
    has_hint = (
        "<tool_call>" in lowered
        or "<function_call>" in lowered
        or "<tools>" in lowered
        or "<function=" in lowered
        or "<function " in lowered
        or "<parameter=" in lowered
        or "<parameter " in lowered
        or '"arguments"' in lowered
        or '"name"' in lowered
    )
    if not has_hint:
        return []
    found = []

    for call in _parse_qwen25coder_tools_block(content):
        if call not in found:
            found.append(call)

    for pat in (_TOOL_CALL_TAGGED, _FUNCTION_CALL_TAGGED):
        for m in pat.finditer(content):
            block = m.group(1)
            call = _extract_json_call(block)
            if not call:
                call = _parse_qwen3_xml_block(block)
            if call and call not in found:
                found.append(call)

    if not found and ("<function=" in lowered or "<function " in lowered):
        call = _parse_qwen3_xml_block(content)
        if call and call not in found:
            found.append(call)

    if not found:
        for m in _BARE_JSON_CALL.finditer(content):
            call = _extract_json_call(m.group(0))
            if call and call not in found:
                found.append(call)

    return found


def _strip_tool_call_syntax(content: str) -> str:
    if not content:
        return ""
    out = _TOOLS_TAGGED.sub("", content)
    out = _TOOL_CALL_TAGGED.sub("", out)
    out = _FUNCTION_CALL_TAGGED.sub("", out)
    if "<function=" in out.lower() or "<function " in out.lower():
        out = re.sub(
            r'<\s*function[\s=].*?<\s*/\s*function\s*>',
            '',
            out,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return out.strip()


def _looks_like_library_question(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for hint in LIBRARY_HINTS:
        if hint in lowered:
            return True
    return False


def _looks_like_search_and_save(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    search_words = ("find", "search", "look up", "lookup", "google",
                    "найди", "найти", "поищи", "поиск")
    save_words = ("save", "write", "store", "dump", "put in",
                  "сохрани", "запиши", "в файл")
    has_search = any(w in lowered for w in search_words)
    has_save = any(w in lowered for w in save_words)
    return has_search and has_save


def _user_wants_file_action(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    hints = (
        "save", "save to", "save it", "save this", "save as",
        "create file", "create a file", "make a file", "make file",
        "write to file", "write a file", "write file", "write this to",
        "write it to", "store to", "dump to", "put in",
        "in a file", "into a file", "to a file", "as a file",
        "сохрани", "сохранить", "запиши", "записать",
        "создай файл", "создать файл", "сделай файл",
        "в файл", "в файле", "в файла", "как файл",
    )
    return any(h in lowered for h in hints)


def _content_hash(text: str) -> int:
    s = (text or "").strip()
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xffffffff
    return h


def _extract_tool_names_from_feedback(feedback: str, available: set) -> list:
    if not feedback or not available:
        return []
    found = []
    for m in re.finditer(r'[`\'"]([a-zA-Z_][a-zA-Z0-9_\-]*)[`\'"]', feedback):
        name = m.group(1)
        if name in available and name not in found:
            found.append(name)
    for m in re.finditer(r'\b([a-z_][a-z0-9_]{2,})\b', feedback):
        name = m.group(1)
        if name in available and name not in found:
            found.append(name)
    return found


# ============================================================
# EMPTY-PROMISE DETECTION
# ============================================================

_FILE_ACTION_TOOLS = (
    "create_file", "write_file", "save_file", "append_to_file",
    "append_file", "open_file", "edit_file", "write_to_file",
)

_EMPTY_PROMISE_PATTERNS = (
    "i will save", "i'll save", "i will create", "i'll create",
    "i will write", "i'll write", "i will now save", "i will now create",
    "let me save", "let me create", "let me write",
    "saving the file", "creating the file", "writing the file",
    "now saving", "now creating", "now writing",
    "next, i will save", "next, i'll save", "i am going to save",
    "i am going to create",
    "file has been created", "file has been saved", "file has been written",
    "file created:", "file saved:", "saved to ", "written to ",
    "i have created", "i have saved", "i have written",
    "i've created", "i've saved", "i've written",
    "successfully saved", "successfully created", "successfully written",
    "here is the file", "here's the file", "here is the content",
    "here's the content",
    "я сохраню", "я создам", "я запишу", "сейчас сохраню",
    "сейчас создам", "сейчас запишу", "давай сохраню", "давай создам",
    "сохраню файл", "создам файл", "запишу в файл", "записываю в файл",
    "сохраняю файл", "создаю файл",
    "файл создан", "файл сохранён", "файл сохранен", "файл записан",
    "файл успешно создан", "файл успешно сохранён",
    "я создал файл", "я сохранил файл", "я записал файл",
    "готово, файл", "готово. файл",
    "вот файл", "вот содержимое", "вот что я сохраню",
)


def _looks_like_empty_promise(content: str) -> bool:
    if not content:
        return False
    lowered = content.lower()
    for p in _EMPTY_PROMISE_PATTERNS:
        if p in lowered:
            return True
    return False


# ============================================================
# UNICODE
# ============================================================

def _detect_unicode_support() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" not in enc:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            cp = ctypes.windll.kernel32.GetConsoleOutputCP()
            if cp not in (65001, 1200):
                return False
        except Exception:
            return False
    return True


UNICODE_OK = _detect_unicode_support()


def _sym(u: str, a: str) -> str:
    return u if UNICODE_OK else a


M_TOOL = _sym("⏺", "*")
M_OUT = _sym("⎿", "L")
M_THINK = _sym("✻", "*")
M_OK = _sym("✓", "+")
M_WARN = _sym("!", "!")
M_BAR = _sym("│", "|")
M_BRAND = _sym("◉", "*")


def _brand(state: str = "") -> str:
    color = {
        "":     "\x1b[38;5;141m",
        "dim":  "\x1b[38;5;97m",
        "ok":   "\x1b[38;5;114m",
        "warn": "\x1b[38;5;221m",
        "err":  "\x1b[38;5;203m",
    }.get(state, "\x1b[38;5;141m")
    return f"{color}{M_BRAND}\x1b[0m"


USER_BG = "#1a1a1f"
USER_BAR_FG = "#3a3a42"
USER_TEXT_FG = "#e4e4e7"

THEME = Theme({
    "accent": "bold cyan",
    "dim": "grey58",
    "success": "bold green",
    "warning": "bold yellow",
    "danger": "bold red",
})


def build_schema(tool_map):
    import inspect
    tmap = {int: "integer", float: "number", str: "string",
            bool: "boolean", list: "array", dict: "object"}
    schema = []
    for name, fn in tool_map.items():
        mcp_schema = getattr(fn, "_mcp_schema", None)
        if mcp_schema:
            schema.append(mcp_schema)
            continue
        sig = inspect.signature(fn)
        props, required = {}, []
        for pname, p in sig.parameters.items():
            ann = p.annotation
            props[pname] = {"type": tmap.get(ann, "string") if ann is not inspect.Parameter.empty else "string"}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        meta = getattr(fn, "_tool_meta", {})
        schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta.get("description", fn.__doc__ or ""),
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return schema


class SlashCompleter(Completer):
    def __init__(self, commands, tools):
        self.commands = commands
        self.tools = tools

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        parts = text.split()
        if len(parts) <= 1 and not text.endswith(" "):
            word = parts[0] if parts else "/"
            for cmd, desc in self.commands.items():
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word),
                                     display=cmd, display_meta=desc)
            return
        if parts and parts[0] in ("/enable", "/disable") and len(parts) <= 2:
            prefix = parts[1] if len(parts) == 2 else ""
            for tname in self.tools:
                if tname.startswith(prefix):
                    yield Completion(tname, start_position=-len(prefix))


class AstroCLI:
    def __init__(self, all_tools, tool_map, filter_tools_fn):
        self.console = Console(
            theme=THEME, highlight=False,
            force_terminal=None, legacy_windows=False, soft_wrap=False,
        )

        self.all_tools = all_tools
        self.filter_tools_fn = filter_tools_fn
        self.tool_map = dict(tool_map)
        self.sid = get_current_session()

        try:
            _active = get_active()
            _s = get_provider_settings(_active)
            if _s.get("model"):
                CONFIG["model"] = _s["model"]
        except Exception:
            pass

        self.session_started = time.time()
        self.tool_calls_count = 0
        self.verifications_run = 0
        self.verifications_pushed_back = 0
        self.last_response_time = 0.0
        self._notified_dead_servers = set()

        self._verifier_cache = {}
        self._verify_system_prompt = None

        self._attach_cache = {}
        self._kill_requested = False

        self._repomap_data = None
        self._repomap_loaded_at = 0.0

        self.messages = load_messages(self.sid)
        sys_prompt = get_system_prompt(CONFIG["mode"])
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages = [{"role": "system", "content": sys_prompt}] + self.messages
        else:
            self.messages[0] = {"role": "system", "content": sys_prompt}

        self.commands = {
            "/help":       "show commands",
            "/status":     "current configuration",
            "/settings":   "show settings",
            "/tools":      "list active tools",
            "/enable":     "enable tool: /enable <name>",
            "/disable":    "disable tool: /disable <name>",
            "/model":      "set model: /model <name>",
            "/mode":       "set mode: /mode Standard|Code|Minimal|Creator",
            "/provider":   "set provider: /provider <name>",
            "/workspace":  "set workspace: /workspace <path>",
            "/mcp":        "list MCP servers",
            "/mcphealth":  "show MCP health status",
            "/sessions":   "list all chats",
            "/history":    "show current chat history",
            "/switch":     "switch chat: /switch <n>",
            "/new":        "start a new chat",
            "/rename":     "rename chat: /rename <title>",
            "/clear":      "clear current chat history",
            "/export":     "export chat: /export <file.md>",
            "/save":       "alias for /export",
            "/undo":       "undo last user+assistant pair",
            "/retry":      "regenerate last assistant response",
            "/find":       "find text in current chat: /find <text>",
            "/stats":      "session statistics",
            "/context":    "approximate context size",
            "/system":     "show current system prompt",
            "/verify":     "toggle self-verification: /verify on|off|trivial|passes N",
            "/planning":   "toggle planning mode: /planning on|off",
            "/verifier-model": "set verifier model: /verifier-model <name|off>",
            "/yank":       "save last reply to file: /yank <file>",
            "/attach":     "attach a file: /attach <path>  (or use !- path -! inline)",
            "/debug":      "test create_file round-trip & print signature",
            "/kill":       "interrupt the current agent turn",
            "/repomap":    "show repo map for the current workspace",
            "/repomap-gen": "rebuild repo map from workspace",
            "/scope":      "show which tools are allowed for a text: /scope <text>",
            "/exit":       "quit",
            "/quit":       "quit",
            "/think":      "toggle thinking mode: /think on|off|auto",
        }

        self.prompt_session = PromptSession(
            history=FileHistory(os.path.expanduser("~/.astronaut_history")),
            completer=SlashCompleter(self.commands, list(self.all_tools.keys())),
            style=Style.from_dict({"prompt": "ansibrightblack bold"}),
            complete_while_typing=True,
            multiline=False,
            enable_history_search=False,
            reserve_space_for_menu=6,
        )

    # ---------------- session helpers ----------------

    def _load_session(self, sid):
        self.messages = load_messages(sid)
        sys_prompt = get_system_prompt(CONFIG["mode"])
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages = [{"role": "system", "content": sys_prompt}] + self.messages
        else:
            self.messages[0] = {"role": "system", "content": sys_prompt}
        self._attach_cache.clear()

    def _rebuild_tools(self):
        base = self.filter_tools_fn(CONFIG["mode"], self.all_tools)
        rt = settings_store.load()
        disabled = set(rt.get("disabled_tools", []) or [])
        self.tool_map = {k: v for k, v in base.items() if k not in disabled}

    @contextmanager
    def _status(self, text, spinner="arc"):
        yield

    # ---------------- repo map ----------------

    def _get_repomap(self, force: bool = False):
        ws = CONFIG.get("workspace") or "."
        if force or self._repomap_data is None:
            try:
                self._repomap_data = _repomap.get_or_build(ws, force=force)
                self._repomap_loaded_at = time.time()
            except Exception as e:
                self.console.print(f"[yellow]repomap error: {e}[/yellow]")
                self._repomap_data = {"files": [], "count": 0,
                                      "workspace": str(ws),
                                      "generated_at": time.time()}
        return self._repomap_data

    def _repomap_hint(self, user_text: str) -> str:
        if not _repomap.user_mentions_repo(user_text):
            return ""
        data = self._get_repomap()
        text = _repomap.render_text(data)
        if not text.strip():
            return ""
        return (
            "[REPO MAP — structure of the current workspace]\n"
            "Use this to navigate the codebase. Do NOT re-read the whole\n"
            "tree; the map below already lists every file and its top-level\n"
            "symbols. Read individual files only when you actually need\n"
            "their content.\n\n"
            f"{text}\n"
            "[END REPO MAP]"
        )

    # ---------------- attachment expansion ----------------

    def _prepare_messages_for_llm(self):
        workspace = Path(os.path.expanduser(CONFIG["workspace"])).resolve()
        out = []
        for i, m in enumerate(self.messages):
            if m.get("role") != "user" or not m.get("content"):
                out.append(m)
                continue
            content = m["content"]
            if _ATTACH_OPEN not in content or _ATTACH_CLOSE not in content:
                out.append(m)
                continue
            key = (i, hash(content))
            cached = self._attach_cache.get(key)
            if cached is None:
                expanded, _ = _expand_file_attachments(content, workspace)
                self._attach_cache[key] = expanded
                cached = expanded
            if cached != content:
                mm = dict(m)
                mm["content"] = cached
                out.append(mm)
            else:
                out.append(m)
        return out

    # ---------------- attachment hints ----------------

    def _build_attachments_hint(self, user_text: str) -> str:
        paths = _list_attachment_markers(user_text)
        if not paths:
            return ""
        workspace = Path(os.path.expanduser(CONFIG["workspace"])).resolve()
        lines = []
        for raw in paths:
            try:
                p = _resolve_attach_path(raw, workspace)
                if p.exists() and p.is_file():
                    lines.append(f"  - {p.name}  ({p.stat().st_size} bytes)")
            except Exception:
                pass
        if not lines:
            return ""
        return (
            "[ATTACHMENTS ALREADY LOADED INTO CONTEXT]\n"
            "The following file(s) have been attached by the user, and their\n"
            "FULL CONTENT is already present in the conversation below.\n"
            "Do NOT re-read them from disk. Do NOT recreate them.\n\n"
            "Attached files:\n"
            + "\n".join(lines) + "\n\n"
            "HARD RULES:\n"
            "  1. Do NOT call `open_file` on these files.\n"
            "  2. Do NOT call `create_file` to recreate them.\n"
            "  3. Do NOT call `append_to_file` to add this content back.\n"
            "  4. Only use file tools if the USER explicitly asks you to\n"
            "     modify/create/save something — with NEW content."
        )

    def _print_attachment_summary(self, original_text: str):
        paths = _list_attachment_markers(original_text)
        if not paths:
            return
        workspace = Path(os.path.expanduser(CONFIG["workspace"])).resolve()
        parts = []
        for raw in paths:
            try:
                p = _resolve_attach_path(raw, workspace)
                if p.exists() and p.is_file():
                    size = p.stat().st_size
                    if size >= 1024:
                        parts.append(f"{p.name} ({size/1024:.1f} KB)")
                    else:
                        parts.append(f"{p.name} ({size} B)")
                else:
                    parts.append(f"{raw} (missing)")
            except Exception:
                parts.append(f"{raw} (error)")
        self.console.print(
            f"[grey58]  📎 attached: {', '.join(parts)}[/grey58]"
        )

    def _print_user_message(self, text: str):
        width = self.console.width or 80
        inner_w = max(20, width - 6)
        wrapped_lines = []
        for raw in (text or "").split("\n"):
            if not raw:
                wrapped_lines.append("")
                continue
            parts = textwrap.wrap(
                raw, width=inner_w,
                break_long_words=True, break_on_hyphens=False,
            ) or [""]
            wrapped_lines.extend(parts)
        for line in wrapped_lines:
            pad = max(0, inner_w - len(line))
            t = Text()
            t.append(f"  {M_BAR} ", style=f"bold {USER_BAR_FG} on {USER_BG}")
            t.append(line + " " * pad, style=f"{USER_TEXT_FG} on {USER_BG}")
            self.console.print(t, soft_wrap=True)

    # ---------------- pre-flight hints ----------------

    def _inject_preflight_hints(self, user_text: str):
        attachments_hint = self._build_attachments_hint(user_text)
        if attachments_hint:
            self.messages.append({
                "role": "system",
                "content": attachments_hint,
            })

        repomap_hint = self._repomap_hint(user_text)
        if repomap_hint:
            self.messages.append({
                "role": "system",
                "content": repomap_hint,
            })

        if self._is_analysis_only_request(user_text):
            self.messages.append({
                "role": "system",
                "content": (
                    "[SCOPE LOCK — ANALYSIS ONLY]\n"
                    "The user attached file(s) and asked you to look/analyze\n"
                    "them. You MUST NOT modify, recreate, save, or copy those\n"
                    "files. Answer with your analysis in PLAIN TEXT.\n"
                    "Do NOT call open_file (content is already here).\n"
                    "Do NOT call create_file / append_to_file.\n"
                    "Only read-only tools are allowed this turn."
                ),
            })

        has_context7 = any(
            "resolve-library-id" in n or "query-docs" in n
            for n in self.tool_map.keys()
        )

        if has_context7 and _looks_like_library_question(user_text):
            self.messages.append({
                "role": "system",
                "content": (
                    "[PRE-FLIGHT REMINDER]\n"
                    "The user asks about a library/framework. You MUST:\n"
                    "  1. Call tool `resolve-library-id` with the library name.\n"
                    "  2. Then call `query-docs` with the returned ID and the question.\n"
                    "DO NOT write Python code like `context7(...)` — that does NOTHING.\n"
                    "DO NOT answer from memory — APIs change."
                ),
            })

        if _looks_like_search_and_save(user_text):
            self.messages.append({
                "role": "system",
                "content": (
                    "[WORKFLOW REMINDER — search and save]\n"
                    "The user asks to FIND something and SAVE it to a file.\n"
                    "Follow this workflow EXACTLY:\n"
                    "  1. Call `ddg_search` ONCE. You now have the search results.\n"
                    "  2. Call `create_file` with filename=<the requested file> and "
                    "content=<the search results formatted as Markdown>. "
                    "Copy the actual titles, URLs and snippets into content.\n"
                    "  3. STOP. Do NOT call `fetch_url` on each link.\n"
                    "  4. Do NOT invent arguments for `fetch_url` (like 'destination'). "
                    "`fetch_url` takes ONLY url and max_chars.\n"
                    "  5. Do NOT create an empty file."
                ),
            })

    def _is_analysis_only_request(self, user_text: str) -> bool:
        if not _list_attachment_markers(user_text):
            return False
        lowered = (user_text or "").lower()
        save_intent = (
            "save", "write", "create file", "create a file", "make a file",
            "modify", "edit", "rewrite", "fix the file", "patch",
            "сохрани", "запиши", "создай файл", "измени", "отредактируй",
            "исправь файл", "допиши",
        )
        if any(s in lowered for s in save_intent):
            return False
        analysis_intent = (
            "check", "analyze", "analyse", "review", "explain", "look at",
            "look over", "tell me", "what does", "what is wrong",
            "find bugs", "read", "examine",
            "посмотри", "проверь", "проанализируй", "объясни",
            "что не так", "найди баги", "прочитай", "глянь",
        )
        return any(a in lowered for a in analysis_intent)

    def _tools_allowed_for_turn(self, user_text: str):
        if self._is_analysis_only_request(user_text):
            return {
                "list_directory", "file_info", "open_file",
                "search_workspace", "rag_search", "rag_stats",
                "get_time", "calculate",
            }

        if _user_wants_file_action(user_text) and not self._had_file_tool_this_turn():
            allowed = set()
            for k in self.tool_map.keys():
                if any(f in k for f in _FILE_ACTION_TOOLS):
                    allowed.add(k)
            if any(w in (user_text or "").lower()
                   for w in ("search", "find", "look up", "google",
                             "найди", "поищи", "поиск")):
                for k in self.tool_map.keys():
                    if any(s in k for s in
                           ("search", "fetch", "url", "web")):
                        allowed.add(k)
            for k in self.tool_map.keys():
                if k in ("open_file", "list_directory", "file_info"):
                    allowed.add(k)
            if allowed:
                return allowed

        return None

    def _get_verifier_model(self, default_model: str) -> str:
        rt = settings_store.load()
        return rt.get("verifier_model") or default_model

    # ---------------- verification ----------------

    def _transcript_for_verify(self, max_msgs: int = 30) -> str:
        """Build a transcript for the verifier.

        Starts at the last user message so the verifier sees the whole
        turn. Skips harness guards (duplicate blocks, cancellations)
        because they're not real tool results.
        """
        start = 0
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "user":
                start = i
                break
        recent = [m for m in self.messages[start:]
                  if m.get("role") != "system"]
        if len(recent) > max_msgs:
            recent = recent[-max_msgs:]

        lines = []
        for m in recent:
            role = m.get("role")
            if role == "user":
                lines.append(f"USER: {(m.get('content') or '')[:500]}")
            elif role == "assistant":
                c = (m.get("content") or "").strip()
                tcs = m.get("tool_calls") or []
                if tcs:
                    names = ", ".join(
                        tc["function"]["name"] if "function" in tc else tc.get("name", "?")
                        for tc in tcs
                    )
                    lines.append(f"ASSISTANT: [called: {names}] {c[:200]}")
                elif c:
                    lines.append(f"ASSISTANT: {c[:500]}")
            elif role == "tool":
                if m.get("harness_guard"):
                    continue
                tn = m.get("tool_name", "tool")
                body = str(m.get("content", ""))[:300].replace("\n", " ")
                lines.append(f"TOOL[{tn}]: {body}")
        return "\n".join(lines) or "(empty transcript)"

    def _is_trivial(self, user_text: str) -> bool:
        words = (user_text or "").strip().split()
        if len(words) >= 4:
            return False
        lowered = (user_text or "").lower()
        if any(hint in lowered for hint in LIBRARY_HINTS):
            return False
        if any(m.get("role") == "tool" for m in self.messages):
            return False
        return True

    def _build_verify_prompt(self, strict: bool) -> str:
        if self._verify_system_prompt and not strict:
            return self._verify_system_prompt

        strict_block = ""
        if strict:
            strict_block = (
                "\nEXTRA CHECK (small):\n"
                "  - If the assistant already explained a failure honestly → OK.\n"
                "  - If unsure → OK.\n"
            )

        prompt = (
            "You are a LENIENT verifier. Answer ONE question only:\n"
            "  Did the assistant satisfy the user's LITERAL request?\n\n"

            "RULES (do NOT add your own requirements):\n"
            "  1. Judge ONLY what the user literally asked. Do not demand\n"
            "     files, sources, formats, or extra steps the user did not ask.\n"
            "  2. If the user did NOT ask for a file → do NOT demand create_file.\n"
            "  3. If the user did NOT ask to save → do NOT demand saving.\n"
            "  4. If a tool failed (no results, error, timeout) and the assistant\n"
            "     TOLD the user about it → that IS a complete answer. OK.\n"
            "  5. Honest failure reporting is success. Knowledge from memory with\n"
            "     a clear label is acceptable. OK.\n"
            "  6. If the user asked for a file and the assistant said\n"
            "     'I could not create it' honestly → OK.\n"
            "  7. Lines marked '[HARNESS]' are internal guards, NOT tool results.\n"
            "     Ignore them completely. If you see a real 'TOOL[name]:' line\n"
            "     above the assistant's claim, the tool call happened.\n\n"

            "Return INCOMPLETE only if:\n"
            "  - The assistant claims a file was created/saved AND no real\n"
            "    'TOOL[create_file]:' or 'TOOL[append_to_file]:' line exists.\n"
            "  - The assistant claims a search happened AND no real search\n"
            "    tool line exists.\n"
            "  - The assistant ignored a tool error and pretended it succeeded.\n\n"

            + strict_block +
            "\nWHEN IN DOUBT → OK. Do not nitpick. Do not repeat previous feedback.\n\n"

            "Answer EXACTLY:\n"
            "VERDICT: OK\n"
            "or\n"
            "VERDICT: INCOMPLETE\n"
            "FEEDBACK: <one short sentence>"
        )

        if not strict:
            self._verify_system_prompt = prompt
        return prompt

    def _verify_before_final(self, provider, model, user_text: str,
                             strict: bool = False) -> tuple:
        transcript = self._transcript_for_verify()
        cache_key = hash((user_text, transcript, strict))
        if cache_key in self._verifier_cache:
            cached = self._verifier_cache[cache_key]
            self.console.print(
                f"  [grey58]{M_THINK} verifier cache hit[/grey58]"
            )
            return cached

        sys_prompt = self._build_verify_prompt(strict)
        verify_msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": (
                f"USER TASK:\n{user_text}\n\n"
                f"TRANSCRIPT:\n{transcript}"
            )},
        ]

        verify_model = self._get_verifier_model(model)

        try:
            r = provider.chat(verify_msgs, tools=None, model=verify_model)
            content = (r.content or "").strip()
        except Exception:
            return True, ""

        upper = content.upper()
        if "VERDICT: OK" in upper or upper.strip().startswith("OK"):
            result = (True, "")
            self._verifier_cache[cache_key] = result
            return result

        feedback = ""
        for line in content.splitlines():
            s = line.strip()
            if s.upper().startswith("FEEDBACK:"):
                feedback = s.split(":", 1)[1].strip()
                break
        if not feedback:
            feedback = "Task appears incomplete — review the request and finish it."

        result = (False, feedback)
        self._verifier_cache[cache_key] = result
        return result

    # ---------------- empty-promise helpers ----------------

    def _had_file_tool_this_turn(self) -> bool:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return False
            if m.get("role") == "tool":
                if m.get("harness_guard"):
                    continue
                tn = (m.get("tool_name") or "").lower()
                if any(k in tn for k in _FILE_ACTION_TOOLS):
                    return True
        return False

    # ---------------- planning ----------------

    def _make_plan(self, provider, model, user_text: str):
        rt = settings_store.load()
        if not rt.get("planning_enabled"):
            return []

        tool_names = set(self.tool_map.keys())
        try:
            steps = _planning.generate_plan(
                provider, model,
                user_text=user_text,
                history=self._prepare_messages_for_llm(),
                tool_names=tool_names,
            )
        except Exception as e:
            self.console.print(
                f"  [yellow]{M_WARN} planning failed: "
                f"{type(e).__name__}: {e}[/yellow]"
            )
            return []
        return steps or []

    def _print_plan(self, steps):
        if not steps:
            return
        body = Text()
        for s in steps:
            n = s.get("n", "?")
            title = s.get("title") or ""
            detail = s.get("detail") or ""
            body.append(f"  {n}. ", style="bold #a480ff")
            body.append(f"{title}\n", style="white")
            if detail:
                body.append(f"     {detail}\n", style="grey58")
        self.console.print(Panel(
            body,
            title=f"[bold #a480ff]{M_THINK} Plan[/bold #a480ff] "
                  f"[grey58]({len(steps)} step"
                  f"{'s' if len(steps) != 1 else ''})[/grey58]",
            border_style="#a480ff",
            padding=(1, 1),
            expand=False,
        ))

    def _print_plan_step_marker(self, step_index: int, plan_len: int):
        if not step_index:
            return
        self.console.print(
            f"  [grey58]{M_THINK} step {step_index}/{plan_len}[/grey58]"
        )

    # ---------------- agent loop ----------------

    def _run_agent_loop(self, provider, model, max_steps, user_text: str = "",
                        plan_steps=None):
        rt = settings_store.load()
        verify_enabled = bool(rt.get("verification_enabled", True))
        max_verify_retries = int(rt.get("max_verification_retries", 2))
        max_verify_retries = max(1, min(max_verify_retries, 3))
        min_attempts = max(1, int(rt.get("min_verification_attempts", 1)))
        verify_trivial_skip = bool(rt.get("verify_trivial_skip", True))

        skip_verify = verify_enabled and verify_trivial_skip and self._is_trivial(user_text)

        plan_steps = plan_steps or []
        current_plan_step = 0

        if plan_steps:
            block = _planning.format_for_prompt(plan_steps)
            last_user_i = None
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i].get("role") == "user":
                    last_user_i = i
                    break
            if last_user_i is None:
                self.messages.append({"role": "system", "content": block})
            else:
                self.messages.insert(last_user_i,
                                     {"role": "system", "content": block})

        step = 0
        verify_retries = 0
        recent_calls = []
        finished = False
        fake_code_retries = 0
        empty_promise_retries = 0
        promise_hashes = set()
        verify_response_hashes = set()
        forced_tool_names = None
        all_tool_names = set(self.tool_map.keys())
        tool_counts = {}

        user_wants_file = _user_wants_file_action(user_text)

        self._kill_requested = False

        RETRY_FORBIDDEN_EXACT = {
            "set_goal", "get_goal", "create_plan",
            "run_subagent", "run_parallel_subagents",
            "delegate_file_task",
            "list_skills_tool", "load_skill",
            "rag_index", "rag_stats",
        }

        def _is_retry_forbidden(name: str) -> bool:
            if name in RETRY_FORBIDDEN_EXACT:
                return True
            low = (name or "").lower()
            if "sequential" in low or "thinking" in low or low.endswith("_think"):
                return True
            return False

        def _looks_like_fake_tool_code(text: str) -> bool:
            if not text:
                return False
            for m in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text):
                name = m.group(1)
                for tn in all_tool_names:
                    if name == tn or name == tn.replace("-", "_") or name == tn.replace("-", ""):
                        return True
                if name.lower() in ("context7", "tavily", "firecrawl"):
                    return True
            if re.search(r'\bimport\s+(context7|tavily|firecrawl)\b', text, re.IGNORECASE):
                return True
            return False

        def _force_honest_final(reason: str = ""):
            self.console.print(
                f"  [red]{M_WARN} forcing honest final answer"
                + (f" ({reason})" if reason else "") + "[/red]"
            )
            self.messages.append({
                "role": "system",
                "content": (
                    "[FORCED HONEST FINAL — TOOLS DISABLED]\n"
                    "The system is stopping retries. You must now answer in\n"
                    "PLAIN TEXT ONLY. Do NOT call tools.\n"
                    "Rules:\n"
                    "  1. If you did NOT complete the user's task, open with a\n"
                    "     single plain sentence stating that.\n"
                    "  2. Explain in ONE sentence why (tool failed, no tool\n"
                    "     available, arguments rejected, etc).\n"
                    "  3. Do NOT pretend the task succeeded.\n"
                    "  4. Do NOT paste content as if it had been saved.\n"
                    "  5. Be brief."
                ),
            })
            try:
                final_r = provider.chat(
                    self._prepare_messages_for_llm(),
                    tools=None, model=model,
                )
                final_text = (final_r.content or "").strip() or "(no response)"
            except Exception as e:
                final_text = f"(forced final failed: {e})"
            self.console.print()
            self.console.print(Text("> ", style="grey58"), end="")
            self.console.print(Markdown(final_text))
            self.messages.append({"role": "assistant", "content": final_text})
            append_message(self.sid, {"role": "assistant", "content": final_text})
            self.console.print()

        turn_allowed = self._tools_allowed_for_turn(user_text)
        if turn_allowed is not None:
            preview = ", ".join(sorted(turn_allowed)[:8])
            if len(turn_allowed) > 8:
                preview += ", …"
            self.console.print(
                f"  [grey58]🧭 scope: turn restricted to {len(turn_allowed)} "
                f"tool(s) — {preview}[/grey58]"
            )

        while step < max_steps and not finished:
            if self._kill_requested:
                self.console.print(
                    f"  [yellow]{M_WARN} /kill requested — stopping agent[/yellow]"
                )
                self.messages.append({
                    "role": "system",
                    "content": (
                        "[KILL REQUESTED]\n"
                        "User interrupted the turn. Answer briefly in plain text\n"
                        "with what you managed to do so far, and stop."
                    ),
                })
                try:
                    r = provider.chat(
                        self._prepare_messages_for_llm(),
                        tools=None, model=model,
                    )
                    txt = (r.content or "").strip() or "(interrupted)"
                except Exception:
                    txt = "(interrupted)"
                self.console.print()
                self.console.print(Text("> ", style="grey58"), end="")
                self.console.print(Markdown(txt))
                self.messages.append({"role": "assistant", "content": txt})
                append_message(self.sid, {"role": "assistant", "content": txt})
                return

            if forced_tool_names:
                filtered = {k: v for k, v in self.tool_map.items()
                            if k in forced_tool_names}
                if not filtered:
                    filtered = {k: v for k, v in self.tool_map.items()
                                if not _is_retry_forbidden(k)}
            elif verify_retries > 0:
                filtered = {k: v for k, v in self.tool_map.items()
                            if not _is_retry_forbidden(k)}
            elif turn_allowed is not None:
                filtered = {k: v for k, v in self.tool_map.items()
                            if k in turn_allowed}
            else:
                filtered = self.tool_map
            schema = build_schema(filtered)

            with self._status(
                f"[grey58]{M_THINK} Thinking…[/grey58]", spinner="arc"
            ):
                try:
                    resp = provider.chat(
                        self._prepare_messages_for_llm(),
                        tools=schema if schema else None,
                        model=model,
                        think=self._get_think_param(model),
                    )
                except KeyboardInterrupt:
                    self.console.print("[yellow]^C[/yellow]")
                    return

            if getattr(resp, "thinking", ""):
                self.console.print(
                    f"[grey58]✻ thinking…[/grey58] "
                    f"[grey42]{resp.thinking}[/grey42]"
                )

            if (not resp.tool_calls
                    and resp.content
                    and resp.content.strip().lower().rstrip(".") == "i cannot complete this"):
                self.messages.append({"role": "assistant", "content": resp.content})
                _force_honest_final(reason="model declared it cannot complete")
                return

            if (verify_retries > 0
                    and not resp.tool_calls
                    and resp.content):
                h = _content_hash(resp.content)
                if h in verify_response_hashes:
                    self.console.print(
                        f"  [red]{M_WARN} model stuck on identical response "
                        f"across verifier retries — aborting retries[/red]"
                    )
                    self.messages.append({"role": "assistant",
                                          "content": resp.content or ""})
                    _force_honest_final(reason="repeated identical response")
                    return
                verify_response_hashes.add(h)

            if not resp.tool_calls and resp.content:
                leaked = _parse_leaked_tool_calls(resp.content)
                if leaked:
                    if verify_retries > 0:
                        leaked = [c for c in leaked if not _is_retry_forbidden(c["name"])]
                    if leaked:
                        resp.tool_calls = leaked
                        resp.content = _strip_tool_call_syntax(resp.content)
                        self.console.print(
                            f"  [grey58]{M_WARN} caught leaked tool call — executing[/grey58]"
                        )

            if not resp.tool_calls and resp.content and _looks_like_fake_tool_code(resp.content):
                if fake_code_retries < 2:
                    fake_code_retries += 1
                    self.console.print(
                        f"  [yellow]{M_WARN} fake tool code — push-back "
                        f"({fake_code_retries}/2)[/yellow]"
                    )
                    self.messages.append({
                        "role": "system",
                        "content": (
                            "[CORRECTION] You wrote Python-style code in your TEXT. "
                            "That does NOTHING. To call a tool, return a REAL TOOL CALL. "
                            "No Python code, no imports."
                        ),
                    })
                    continue

            if (not resp.tool_calls
                    and resp.content
                    and user_wants_file
                    and _looks_like_empty_promise(resp.content)
                    and not self._had_file_tool_this_turn()):

                h = _content_hash(resp.content)
                repeated = h in promise_hashes
                promise_hashes.add(h)
                empty_promise_retries += 1

                self.messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                })

                if empty_promise_retries <= 3 and not repeated:
                    self.console.print(
                        f"  [yellow]{M_WARN} empty promise (no tool call) — "
                        f"push-back {empty_promise_retries}/3[/yellow]"
                    )
                    self.messages.append({
                        "role": "system",
                        "content": (
                            "[STOP — YOU DID NOT ACTUALLY DO ANYTHING]\n"
                            "You wrote a promise ('I will save', 'файл создан', "
                            "'saved to X', etc.) or pasted the content into your "
                            "reply WITHOUT calling a tool.\n"
                            "Text is NOT action. Writing the file's content in your "
                            "reply does NOT create the file.\n"
                            "You MUST return a REAL TOOL CALL. Specifically, for "
                            "creating a file, return a tool call to `create_file`:\n"
                            "    tool name: create_file\n"
                            "    arguments:  {\"filename\": \"<name>\", "
                            "\"content\": \"<full text>\"}\n"
                            "Do NOT describe the call. Do NOT paste the content\n"
                            "into your text. Return the actual tool call.\n"
                            "If `create_file` is not available, say so explicitly\n"
                            "IN PLAIN TEXT, without pretending success."
                        ),
                    })
                    continue

                if empty_promise_retries <= 5 and repeated:
                    self.console.print(
                        f"  [yellow]{M_WARN} model repeats same text — "
                        f"escalating push-back {empty_promise_retries}/5[/yellow]"
                    )
                    self.messages.append({
                        "role": "system",
                        "content": (
                            "[ESCALATION]\n"
                            "You ALREADY gave this exact answer and it was "
                            "rejected because no tool was called.\n"
                            "STOP repeating. Either:\n"
                            "  A) Return a REAL tool call:\n"
                            "        create_file(filename=..., content=...)\n"
                            "  B) Or answer in PLAIN TEXT: 'I cannot create the "
                            "file: <one-line reason>'.\n"
                            "No more promises. No more pasting the content."
                        ),
                    })
                    continue

                _force_honest_final(reason="empty-promise retries exhausted")
                return

            if resp.tool_calls:
                verify_response_hashes.clear()

                if verify_retries > 0:
                    allowed = [c for c in resp.tool_calls if not _is_retry_forbidden(c["name"])]
                    blocked = [c for c in resp.tool_calls if _is_retry_forbidden(c["name"])]

                    if not allowed and blocked:
                        names = ", ".join(c["name"] for c in blocked)
                        self.console.print(
                            f"  [yellow]{M_WARN} blocked forbidden meta-tool(s): {names}[/yellow]"
                        )
                        self.messages.append({
                            "role": "assistant",
                            "content": resp.content or "",
                            "tool_calls": [
                                {"function": {"name": c["name"], "arguments": c["arguments"]}}
                                for c in blocked
                            ],
                        })
                        for c in blocked:
                            self.messages.append({
                                "role": "tool",
                                "tool_name": c["name"],
                                "content": (
                                    "[HARNESS] Blocked during verifier retries. "
                                    "This tool was NOT executed."
                                ),
                                "harness_guard": "verifier_block",
                            })
                        step += 1
                        continue
                    else:
                        resp.tool_calls = allowed

                if resp.content and resp.content.strip():
                    self.console.print()
                    self.console.print(Text(resp.content, style="grey70"))

                am = {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in resp.tool_calls
                    ],
                }
                self.messages.append(am)
                append_message(self.sid, am)

                for tc in resp.tool_calls:
                    name = tc["name"]

                    allowed_names = set(filtered.keys())
                    if name not in allowed_names:
                        self.console.print(
                            f"  [red]{M_WARN} scope violation: model tried to "
                            f"call '{name}' but it is not allowed this turn[/red]"
                        )
                        self.messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "function": {
                                    "name": name,
                                    "arguments": tc.get("arguments", {}),
                                }
                            }],
                        })
                        self.messages.append({
                            "role": "tool",
                            "tool_name": name,
                            "content": (
                                f"[HARNESS] Tool '{name}' is not available this turn. "
                                f"The tool was NOT executed. Allowed: "
                                f"{', '.join(sorted(allowed_names))}. "
                                f"Do not retry this tool."
                            ),
                            "harness_guard": "scope_violation",
                        })
                        continue

                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    if tool_counts[name] > 3:
                        self.console.print(
                            f"  [yellow]{M_WARN} loop guard: '{name}' × "
                            f"{tool_counts[name]} — forcing final answer "
                            f"(honesty required)[/yellow]"
                        )
                        self.messages.append({
                            "role": "system",
                            "content": (
                                "[FORCED STOP — HONESTY REQUIRED]\n"
                                f"Tool '{name}' was called {tool_counts[name]} times "
                                "and failed every time.\n"
                                "Tool calls are now disabled. Your next reply is the "
                                "FINAL answer.\n"
                                "RULES — non-negotiable:\n"
                                "  1. Open by stating that the tool FAILED.\n"
                                "  2. Do NOT invent data you did not obtain from a tool.\n"
                                "  3. If you give anything from memory, label it: "
                                "'The tool failed — the following is from my general "
                                "training, NOT from a live source:'\n"
                                "  4. Do not pretend the task succeeded.\n"
                                "  5. Be brief."
                            ),
                        })
                        finished = True
                        break

                    if plan_steps:
                        step_index = _planning.match_step(
                            plan_steps, name, tc.get("arguments", {}),
                            current=current_plan_step,
                        )
                        if step_index and step_index >= current_plan_step:
                            current_plan_step = step_index
                            self._print_plan_step_marker(
                                current_plan_step, len(plan_steps)
                            )

                    self._print_tool_call(name, tc["arguments"])

                    key = (name, json.dumps(tc["arguments"], sort_keys=True, default=str))
                    if key in recent_calls:
                        result = ("[HARNESS] Duplicate call blocked. The tool was "
                                  "NOT executed this time. Check the transcript "
                                  "above for the previous identical call's result.")
                        self.console.print(f"    [grey35]{M_OUT}[/grey35] [grey58]skip: duplicate[/grey58]")
                        tm = {"role": "tool", "tool_name": name,
                              "content": result, "harness_guard": "duplicate"}
                        self.messages.append(tm)
                        continue
                    else:
                        recent_calls.append(key)
                        fn = self.tool_map.get(name)
                        if not fn:
                            result = f"Unknown tool: {name}"
                        elif getattr(fn, "_requires_confirmation", False):
                            if not self._confirm(name, tc["arguments"]):
                                result = ("[HARNESS] Cancelled by user. The tool "
                                          "was NOT executed.")
                                self.console.print(f"    [grey35]{M_OUT}[/grey35] [grey58]cancelled[/grey58]")
                                tm = {"role": "tool", "tool_name": name,
                                      "content": result, "harness_guard": "cancelled"}
                                self.messages.append(tm)
                                continue
                            else:
                                result = self._run_tool(fn, tc["arguments"], name)
                        else:
                            result = self._run_tool(fn, tc["arguments"], name)

                    self._print_tool_result(name, result)
                    tm = {"role": "tool", "tool_name": name, "content": str(result)}
                    self.messages.append(tm)
                    append_message(self.sid, tm)
                    self.tool_calls_count += 1

                if not finished:
                    step += 1
                    self.console.print()
            else:
                if resp.content and resp.content.strip():
                    am_plain = {"role": "assistant", "content": resp.content}
                    self.messages.append(am_plain)
                    append_message(self.sid, am_plain)
                    self.console.print()
                    self.console.print(Markdown(resp.content))

                if verify_enabled and not skip_verify and verify_retries < max_verify_retries:
                    self.verifications_run += 1
                    with self._status(
                        f"[grey58]{M_THINK} Verifying…[/grey58]", spinner="dots12"
                    ):
                        ok, feedback = self._verify_before_final(
                            provider, model, user_text, strict=False
                        )

                    if ok and min_attempts > 1:
                        for pass_idx in range(2, min_attempts + 1):
                            self.verifications_run += 1
                            self.console.print(
                                f"  [grey58]{M_THINK} Verifying "
                                f"(pass {pass_idx}/{min_attempts}, strict)…[/grey58]"
                            )
                            with self._status(
                                f"[grey58]{M_THINK} Verifying "
                                f"(pass {pass_idx}/{min_attempts})…[/grey58]",
                                spinner="dots12",
                            ):
                                ok2, feedback2 = self._verify_before_final(
                                    provider, model, user_text, strict=True
                                )
                            if not ok2:
                                ok = False
                                feedback = feedback2
                                break

                    if not ok:
                        verify_retries += 1
                        self.verifications_pushed_back += 1
                        self.console.print()
                        self.console.print(
                            f"  [yellow]{M_WARN} verifier push-back "
                            f"{verify_retries}/{max_verify_retries}[/yellow]"
                        )
                        self.console.print(
                            f"    [grey35]{M_OUT}[/grey35] [grey70]{feedback}[/grey70]"
                        )

                        if verify_retries >= max_verify_retries:
                            _force_honest_final(reason="verifier retries exhausted")
                            return

                        demanded = _extract_tool_names_from_feedback(
                            feedback, all_tool_names
                        )
                        forced_tool_names = set(demanded) if demanded else None

                        if forced_tool_names:
                            tool_list = ", ".join(sorted(forced_tool_names))
                            self.console.print(
                                f"  [grey58]{M_THINK} restricting next response "
                                f"to tool(s): {tool_list}[/grey58]"
                            )
                            self.messages.append({
                                "role": "system",
                                "content": (
                                    "[FORCED ACTION — VERIFIER DEMAND]\n"
                                    f"Verifier said: {feedback}\n\n"
                                    f"You MUST return a TOOL CALL to one of "
                                    f"these tools: {tool_list}.\n"
                                    "Do NOT write prose. Do NOT apologize. Do NOT "
                                    "paste content into your text. Return the "
                                    "actual tool call with the correct arguments.\n"
                                    "If it is truly impossible, respond with "
                                    "EXACTLY this one line and nothing else:\n"
                                    "    I cannot complete this."
                                ),
                            })
                        else:
                            self.messages.append({
                                "role": "user",
                                "content": (
                                    f"[VERIFIER FEEDBACK] {feedback}\n"
                                    "Continue working. Call tools if needed. "
                                    "Do not just describe — execute. "
                                    "If impossible, say so plainly."
                                ),
                            })
                        step += 1
                        continue
                    else:
                        forced_tool_names = None
                        verify_response_hashes.clear()
                        self.console.print(
                            f"  [green]{M_OK} verifier ok "
                            f"({min_attempts} pass{'es' if min_attempts != 1 else ''})[/green]"
                        )

                elif verify_enabled and verify_retries >= max_verify_retries:
                    _force_honest_final(reason="verifier retries exhausted")
                    return

                finished = True

        if plan_steps and finished:
            self.console.print(
                f"  [green]{M_OK} plan complete "
                f"({len(plan_steps)}/{len(plan_steps)} steps)[/green]"
            )

        if not finished:
            self.console.print(f"[yellow]max steps reached — finalizing[/yellow]")

    # ---------------- streaming final answer ----------------

    def _stream_final(self, provider, model):
        self.console.print()

        dot = _brand("dim")
        self.console.print(f"  {dot}  [grey58]thinking…[/grey58]")
        self.console.print()

        buf = []
        thinking_buf = []
        think = self._get_think_param(model)

        from rich.live import Live
        preview_started = False
        try:
            with Live(console=self.console, refresh_per_second=12,
                      transient=True) as live:
                for chunk in provider.chat(
                    self._prepare_messages_for_llm(),
                    tools=None, stream=True, model=model, think=think,
                ):
                    th = getattr(chunk, "thinking", "") or ""
                    if th:
                        thinking_buf.append(th)
                    if chunk.content:
                        if not preview_started:
                            preview_started = True
                        buf.append(chunk.content)
                        preview = Text("> ", style="grey58")
                        preview.append("".join(buf), style="grey70")
                        live.update(preview)
        except KeyboardInterrupt:
            self.console.print("[yellow]^C[/yellow]")
        except Exception as e:
            self.console.print(f"[red]stream error: {e}[/red]")

        if thinking_buf:
            self.console.print(f"[grey58]{M_THINK} thinking…[/grey58]")
            self.console.print(Text("".join(thinking_buf), style="grey42"))
            self.console.print()

        full = "".join(buf).strip()
        if full:
            self.console.print(Text("> ", style="grey58"), end="")
            self.console.print(Markdown(full))
            self.messages.append({"role": "assistant", "content": full})
            append_message(self.sid, {"role": "assistant", "content": full})
        else:
            self.console.print("[grey58](no response)[/grey58]")

        self.console.print()

    def _send(self, text):
        self._print_user_message(text)
        self._print_attachment_summary(text)

        append_message(self.sid, {"role": "user", "content": text})
        self.messages.append({"role": "user", "content": text})

        self._inject_preflight_hints(text)

        _, cfg, model = self._provider_cfg()
        provider = get_provider(cfg)

        plan_steps = self._make_plan(provider, model, text)
        if plan_steps:
            self._print_plan(plan_steps)
            self.console.print()

        max_steps = int(settings_store.load().get("max_steps", 15))
        self._run_agent_loop(provider, model, max_steps,
                             user_text=text, plan_steps=plan_steps)

        last = self.messages[-1] if self.messages else None
        need_final = not (
            last
            and last.get("role") == "assistant"
            and (last.get("content") or "").strip()
        )
        if need_final:
            self._stream_final(provider, model)

    def _print_tool_call(self, name, args):
        fn = self.tool_map.get(name)
        is_mcp = fn is not None and getattr(fn, "_mcp_schema", None) is not None
        s = json.dumps(args, ensure_ascii=False)
        if len(s) > 120:
            s = s[:117] + "…"
        marker_style = "#0e7490" if is_mcp else "#7c3aed"
        line = Text()
        line.append(f"  {M_TOOL} ", style=f"bold {marker_style}")
        line.append(name, style="bold white")
        line.append(f"({s})", style="grey58")
        self.console.print(line)

    def _print_tool_result(self, name, result):
        s = str(result).strip()
        if not s:
            self.console.print(f"    [grey35]{M_OUT}[/grey35] [grey58](empty)[/grey58]")
            return

        lines = s.split("\n")
        is_diff = any(
            l.startswith(("+++", "---", "@@")) for l in lines[:6]
        ) or any(
            l.startswith(("+ ", "- ")) for l in lines[:6]
        )

        if is_diff:
            self.console.print(f"    [grey35]{M_OUT}[/grey35]")
            shown = lines[:30]
            for line in shown:
                if line.startswith(("+++", "---")):
                    self.console.print(f"      [grey58]{line}[/grey58]")
                elif line.startswith("@@"):
                    self.console.print(f"      [cyan]{line}[/cyan]")
                elif line.startswith("+"):
                    self.console.print(f"      [green]{line}[/green]")
                elif line.startswith("-"):
                    self.console.print(f"      [red]{line}[/red]")
                else:
                    self.console.print(f"      [grey70]{line}[/grey70]")
            if len(lines) > 30:
                self.console.print(f"      [grey58]... (+{len(lines)-30} more lines)[/grey58]")
            return

        low = s.lower()
        is_error = (
            low.startswith("tool error")
            or low.startswith("error")
            or "traceback" in low
            or low.startswith("sandbox rejected")
            or low.startswith("unknown tool")
            or low.startswith("cancelled by user")
            or low.startswith("duplicate call blocked")
        )

        if is_error:
            self.console.print(f"    [red]{M_OUT} ERROR:[/red]")
            for line in lines[:20]:
                self.console.print(f"      [red]{line}[/red]")
            if len(lines) > 20:
                self.console.print(
                    f"      [red]... (+{len(lines)-20} more lines)[/red]"
                )
            return

        first = lines[0]
        if len(first) > 140:
            first = first[:137] + "…"
        extra = f" [grey58](+{len(lines)-1} lines)[/grey58]" if len(lines) > 1 else ""
        self.console.print(f"    [grey35]{M_OUT}[/grey35] [grey70]{first}[/grey70]{extra}")

    def _run_tool(self, fn, args, name):
        t0 = time.time()
        try:
            with open(".astronaut_tool_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tool": name,
                    "args": args,
                }, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

        try:
            result = fn(**args)
        except TypeError as e:
            sig = ""
            try:
                import inspect
                sig = str(inspect.signature(fn))
            except Exception:
                pass
            result = (f"{_brand('err')} Tool error (bad arguments): {e}\n"
                      f"  expected signature: {name}{sig}")
        except KeyboardInterrupt:
            result = "Tool interrupted by user."
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            result = f"Tool error: {type(e).__name__}: {e}\n{tb[-500:]}"

        dt = time.time() - t0

        try:
            with open(".astronaut_tool_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tool": name,
                    "result": str(result)[:2000],
                    "duration": round(dt, 3),
                }, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

        if dt > 0.5:
            self.console.print(f"      [grey35]({dt:.1f}s)[/grey35]")
        return result

    def _confirm(self, name, args):
        self.console.print()
        body = Group(
            Text(f"{M_WARN} Permission required", style="bold yellow"),
            Text(),
            Text(name, style="bold white"),
            Text(),
            Syntax(json.dumps(args, indent=2, ensure_ascii=False, default=str),
                   "json", theme="monokai", word_wrap=True, background_color="default"),
        )
        self.console.print(Panel(body, border_style="yellow", padding=(1, 2)))
        try:
            ans = self.prompt_session.prompt(
                FormattedText([
                    ("fg:ansiyellow bold", "execute? "),
                    ("fg:#7a7a85", "[y/N] "),
                ]),
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        return ans in ("y", "yes")

    def _provider_cfg(self):
        rt = settings_store.load()
        active = get_active()
        preset = PROVIDER_PRESETS.get(active, {})
        s = get_provider_settings(active)
        temp = float(rt.get("temperature", 0.4))
        keep_alive = rt.get("keep_alive", "30m")

        if active == "ollama":
            cfg = {"provider": "ollama", "keep_alive": keep_alive, "temperature": temp}
            model = s.get("model") or CONFIG["model"] or "qwen2.5:7b"
        else:
            cfg = {
                "provider": "openai",
                "api_key": s.get("api_key") or CONFIG.get("api_key"),
                "base_url": s.get("base_url") or preset.get("default_base"),
                "temperature": temp,
            }
            model = s.get("model") or CONFIG["model"] or "gpt-4o-mini"
        return active, cfg, model

    # ---------------- commands ----------------

    def _handle_command(self, text):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        dispatch = {
            "/exit":     lambda: sys.exit(0),
            "/quit":     lambda: sys.exit(0),
            "/help":     lambda: self._cmd_help(),
            "/status":   lambda: self._cmd_status(),
            "/settings": lambda: self._cmd_settings(),
            "/tools":    lambda: self._cmd_tools(),
            "/enable":   lambda: self._cmd_enable_disable(arg, True),
            "/disable":  lambda: self._cmd_enable_disable(arg, False),
            "/model":    lambda: self._cmd_model(arg),
            "/mode":     lambda: self._cmd_mode(arg),
            "/provider": lambda: self._cmd_provider(arg),
            "/workspace": lambda: self._cmd_workspace(arg),
            "/mcp":      lambda: self._cmd_mcp(),
            "/mcphealth": lambda: self._cmd_mcp_health(),
            "/sessions": lambda: self._cmd_sessions(),
            "/history":  lambda: self._cmd_history(),
            "/switch":   lambda: self._cmd_switch(arg),
            "/new":      lambda: self._cmd_new(),
            "/rename":   lambda: self._cmd_rename(arg),
            "/clear":    lambda: self._cmd_clear(),
            "/export":   lambda: self._cmd_export(arg),
            "/save":     lambda: self._cmd_export(arg),
            "/undo":     lambda: self._cmd_undo(),
            "/retry":    lambda: self._cmd_retry(),
            "/find":     lambda: self._cmd_find(arg),
            "/stats":    lambda: self._cmd_stats(),
            "/context":  lambda: self._cmd_context(),
            "/system":   lambda: self._cmd_system(),
            "/verify":   lambda: self._cmd_verify(arg),
            "/planning": lambda: self._cmd_planning(arg),
            "/verifier-model": lambda: self._cmd_verifier_model(arg),
            "/yank":     lambda: self._cmd_yank(arg),
            "/attach":   lambda: self._cmd_attach(arg),
            "/debug":    lambda: self._cmd_debug(),
            "/kill":     lambda: self._cmd_kill(),
            "/repomap":  lambda: self._cmd_repomap(),
            "/repomap-gen": lambda: self._cmd_repomap_gen(),
            "/scope":    lambda: self._cmd_scope(arg),
            "/think":    lambda: self._cmd_think(arg),
        }
        fn = dispatch.get(cmd)
        if fn:
            fn()
        else:
            self.console.print(f"[yellow]unknown command: {cmd}[/yellow]  try [cyan]/help[/cyan]")

    def _cmd_repomap(self):
        data = self._get_repomap()
        text = _repomap.render_text(data)
        self.console.print()
        self.console.print(Rule(
            f"[grey58]repo map · {_repomap.summary_one_line(data)}[/grey58]",
            style="grey35",
        ))
        self.console.print(Syntax(text, "text", theme="monokai",
                                  word_wrap=True, background_color="default"))
        self.console.print(Rule(style="grey35"))
        self.console.print()

    def _cmd_repomap_gen(self):
        self.console.print("[grey58]rebuilding repo map…[/grey58]")
        data = self._get_repomap(force=True)
        self.console.print(
            f"[green]{M_OK}[/green] map: {_repomap.summary_one_line(data)}"
        )

    def _cmd_scope(self, arg):
        if not arg:
            self.console.print("[yellow]usage: /scope <text>[/yellow]")
            return
        allowed = self._tools_allowed_for_turn(arg)
        if allowed is None:
            self.console.print("[green]scope: unrestricted (all tools)[/green]")
            return
        self.console.print(
            f"[cyan]scope: {len(allowed)} tool(s)[/cyan]"
        )
        for t in sorted(allowed):
            self.console.print(f"  [grey58]{t}[/grey58]")
        self.console.print(
            f"[yellow]excluded: {len(self.tool_map) - len(allowed)} tool(s)[/yellow]"
        )

    def _cmd_kill(self):
        self._kill_requested = True
        self.console.print(
            f"[yellow]{M_WARN} kill requested — agent will stop after current step[/yellow]"
        )

    def _cmd_help(self):
        t = Table.grid(padding=(0, 2))
        t.add_column(style="cyan bold", no_wrap=True)
        t.add_column(style="white")
        for c, d in self.commands.items():
            t.add_row(c, d)
        self.console.print(Panel(t, title="commands", border_style="grey35",
                                 padding=(1, 2), expand=False))

    def _cmd_debug(self):
        import inspect
        names = [n for n in self.tool_map.keys()
                 if "create_file" in n or n == "create_file"]
        if not names:
            self.console.print("[red]create_file not in tool_map[/red]")
            avail = ", ".join(sorted(self.tool_map.keys()))
            self.console.print(f"[grey58]available: {avail[:400]}[/grey58]")
            return
        name = names[0]
        fn = self.tool_map[name]
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.items())
        except Exception:
            sig = None
            params = []
        self.console.print(f"[cyan]tool:[/cyan] {name}")
        self.console.print(f"[cyan]signature:[/cyan] {name}{sig if sig else '()'}")
        for pname, p in params:
            ann = p.annotation
            ann_s = getattr(ann, "__name__", str(ann)) if ann is not inspect.Parameter.empty else "?"
            default = "" if p.default is inspect.Parameter.empty else f" = {p.default!r}"
            self.console.print(f"  [grey58]{pname}: {ann_s}{default}[/grey58]")

        kwargs = {}
        param_names = [p[0] for p in params]
        if "filename" in param_names:
            kwargs["filename"] = "_debug_test.txt"
        elif "path" in param_names:
            kwargs["path"] = "_debug_test.txt"
        elif "file" in param_names:
            kwargs["file"] = "_debug_test.txt"
        elif "name" in param_names:
            kwargs["name"] = "_debug_test.txt"
        else:
            self.console.print("[red]cannot determine filename arg[/red]")
            return
        if "content" in param_names:
            kwargs["content"] = "hello from /debug\n"
        elif "text" in param_names:
            kwargs["text"] = "hello from /debug\n"
        elif "data" in param_names:
            kwargs["data"] = "hello from /debug\n"
        else:
            self.console.print("[red]cannot determine content arg[/red]")
            return
        self.console.print(f"[grey58]calling with: {kwargs}[/grey58]")
        try:
            result = fn(**kwargs)
        except Exception as e:
            self.console.print(f"[red]exception: {type(e).__name__}: {e}[/red]")
            import traceback
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
            return
        self.console.print(f"[cyan]result:[/cyan] {result}")

        workspace = pathlib.Path(CONFIG["workspace"]).resolve()
        for candidate in (
            workspace / "_debug_test.txt",
            workspace / "workspace" / "_debug_test.txt",
            pathlib.Path("_debug_test.txt").resolve(),
        ):
            if candidate.exists():
                self.console.print(
                    f"[green]{M_OK}[/green] found on disk: [cyan]{candidate}[/cyan] "
                    f"({candidate.stat().st_size} bytes)"
                )
                try:
                    candidate.unlink()
                    self.console.print("[grey58]cleaned up[/grey58]")
                except Exception:
                    pass
                return
        self.console.print(
            f"[red]file NOT found. looked in:[/red]\n"
            f"[grey58]  {workspace}[/grey58]\n"
            f"[grey58]  {workspace / 'workspace'}[/grey58]\n"
            f"[grey58]  {pathlib.Path('_debug_test.txt').resolve()}[/grey58]"
        )

    def _cmd_attach(self, arg):
        if not arg:
            self.console.print(
                "[grey58]Attach files by wrapping the path in[/grey58] "
                "[cyan]!- ... -![/cyan] "
                "[grey58]inline in any message.[/grey58]\n"
                "[grey58]Example:[/grey58] "
                "[cyan]review this code !- main.py -![/cyan]\n"
                "[grey58]Relative paths resolve against the workspace; "
                "absolute paths and[/grey58] [cyan]~[/cyan] [grey58]work too. "
                f"Max {_ATTACH_MAX_BYTES // 1000} KB per file.[/grey58]"
            )
            return
        workspace = Path(os.path.expanduser(CONFIG["workspace"])).resolve()
        try:
            p = _resolve_attach_path(arg, workspace)
        except Exception as e:
            self.console.print(f"[red]bad path: {e}[/red]")
            return
        if not p.exists():
            self.console.print(f"[red]not found:[/red] {p}")
            return
        if not p.is_file():
            self.console.print(f"[red]not a file:[/red] {p}")
            return
        size = p.stat().st_size
        lines = p.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        self.console.print(
            f"[green]{M_OK}[/green] [cyan]{p}[/cyan]\n"
            f"[grey58]  {size} bytes · ~{lines} lines[/grey58]\n"
            f"[grey58]  to attach: [/grey58][cyan]!- {arg} -![/cyan]"
        )

    def _get_think_param(self, model: str):
        rt = settings_store.load()
        mode = rt.get("thinking_mode", "auto")
        if mode == "on":
            return True
        if mode == "off":
            return False
        return None

    def _cmd_mcp_health(self):
        mgr = get_mcp_manager()
        status = mgr.status()
        if not status:
            self.console.print("[grey58]no MCP servers configured[/grey58]")
            return
        t = Table(border_style="grey35", show_header=True, header_style="cyan", expand=False)
        t.add_column("name", style="white")
        t.add_column("status")
        t.add_column("tools", style="grey58")
        t.add_column("detail", style="grey58")
        for s in status:
            if s.get("dead"):
                st = f"[red]{M_WARN} dead[/red]"
                detail = s.get("dead_reason", "")[:60]
            elif s["connected"]:
                st = f"[green]{M_OK} connected[/green]"
                detail = ""
            elif not s["enabled"]:
                st = "[grey58]○ disabled[/grey58]"
                detail = ""
            else:
                st = "[yellow]○ offline[/yellow]"
                detail = ""
            t.add_row(s["name"], st, str(s["tool_count"]), detail)
        self.console.print(t)

    def _cmd_think(self, arg):
        rt = settings_store.load()
        a = (arg or "").lower()
        _, _, model = self._provider_cfg()

        if a in ("on", "true", "1", "yes"):
            rt["thinking_mode"] = "on"
            settings_store.save(rt)
            self.console.print(f"[green]{M_OK}[/green] thinking -> [bold green]on[/bold green]")
        elif a in ("off", "false", "0", "no"):
            rt["thinking_mode"] = "off"
            settings_store.save(rt)
            self.console.print(f"[green]{M_OK}[/green] thinking -> [bold yellow]off[/bold yellow]")
        elif a == "auto":
            rt["thinking_mode"] = "auto"
            settings_store.save(rt)
            self.console.print(f"[green]{M_OK}[/green] thinking -> [bold]auto[/bold]")
        else:
            supports = model_supports_thinking(model)
            state = rt.get("thinking_mode", "auto")
            self.console.print(
                f"model: [cyan]{model}[/cyan]  ·  "
                f"supports thinking: [bold]{'yes' if supports else 'no'}[/bold]  ·  "
                f"mode: [bold]{state}[/bold]"
            )
            if not supports:
                self.console.print(
                    "[grey58]this model does not advertise 'thinking' capability[/grey58]"
                )
            self.console.print("[grey58]usage: /think on|off|auto[/grey58]")

    def _cmd_verify(self, arg):
        rt = settings_store.load()
        a = (arg or "").lower()
        if a in ("on", "true", "1", "yes"):
            rt["verification_enabled"] = True
            settings_store.save(rt)
            self.console.print(f"[green]{M_OK}[/green] verifier [bold green]on[/bold green]")
        elif a in ("off", "false", "0", "no"):
            rt["verification_enabled"] = False
            settings_store.save(rt)
            self.console.print(f"[green]{M_OK}[/green] verifier [bold yellow]off[/bold yellow]")
        elif a == "trivial":
            cur = bool(rt.get("verify_trivial_skip", True))
            rt["verify_trivial_skip"] = not cur
            settings_store.save(rt)
            state = "on" if not cur else "off"
            self.console.print(f"[green]{M_OK}[/green] trivial-skip -> [bold]{state}[/bold]")
        elif a.startswith("passes"):
            parts = a.split()
            if len(parts) == 2 and parts[1].isdigit():
                n = max(1, min(5, int(parts[1])))
                rt["min_verification_attempts"] = n
                settings_store.save(rt)
                self.console.print(f"[green]{M_OK}[/green] min_verification_attempts -> [bold]{n}[/bold]")
            else:
                self.console.print("[grey58]usage: /verify passes 1..5[/grey58]")
        else:
            state = "on" if rt.get("verification_enabled", True) else "off"
            skip = "on" if rt.get("verify_trivial_skip", True) else "off"
            passes = rt.get("min_verification_attempts", 1)
            vmodel = rt.get("verifier_model") or "(main)"
            self.console.print(
                f"verifier: [bold]{state}[/bold] · trivial-skip: [bold]{skip}[/bold] · "
                f"passes: [bold]{passes}[/bold] · model: [cyan]{vmodel}[/cyan]"
            )
            self.console.print("[grey58]usage: /verify on|off|trivial|passes N[/grey58]")

    def _cmd_planning(self, arg):
        rt = settings_store.load()
        a = (arg or "").lower()
        if a in ("on", "true", "1", "yes"):
            rt["planning_enabled"] = True
            settings_store.save(rt)
            self.console.print(
                f"[green]{M_OK}[/green] planning [bold green]on[/bold green] "
                f"[grey58](agent writes a plan first)[/grey58]"
            )
        elif a in ("off", "false", "0", "no"):
            rt["planning_enabled"] = False
            settings_store.save(rt)
            self.console.print(
                f"[green]{M_OK}[/green] planning [bold yellow]off[/bold yellow]"
            )
        else:
            state = "on" if rt.get("planning_enabled") else "off"
            color = "green" if state == "on" else "yellow"
            self.console.print(
                f"planning: [bold {color}]{state}[/bold {color}]  "
                f"[grey58]usage: /planning on|off[/grey58]"
            )

    def _cmd_verifier_model(self, arg):
        rt = settings_store.load()
        a = (arg or "").strip()
        if not a:
            current = rt.get("verifier_model") or "(main model)"
            self.console.print(f"verifier model: [cyan]{current}[/cyan]")
            self.console.print(
                "[grey58]usage: /verifier-model <name|off>[/grey58]\n"
                "[grey58]example: /verifier-model granite4.1:8b[/grey58]"
            )
            return
        if a.lower() in ("off", "none", "main", "default", "-"):
            rt["verifier_model"] = ""
            settings_store.save(rt)
            self.console.print(
                f"[green]{M_OK}[/green] verifier model -> [bold]main model[/bold]"
            )
        else:
            rt["verifier_model"] = a
            settings_store.save(rt)
            self.console.print(
                f"[green]{M_OK}[/green] verifier model -> [cyan]{a}[/cyan]"
            )
            self.console.print(
                "[grey58]the verifier will use this model on the next turn[/grey58]"
            )

    def _cmd_yank(self, arg):
        if not arg:
            self.console.print("[yellow]usage: /yank <file>[/yellow]")
            return
        last = None
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                last = m
                break
        if not last:
            self.console.print("[yellow]no assistant reply to yank[/yellow]")
            return
        try:
            pathlib.Path(arg).write_text(last["content"], encoding="utf-8")
            self.console.print(f"[green]{M_OK}[/green] saved to [cyan]{arg}[/cyan]")
        except Exception as e:
            self.console.print(f"[red]failed: {e}[/red]")

    def _cmd_history(self):
        for m in self.messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                continue
            if role == "user":
                self.console.print()
                self.console.print(f"[grey58]▌ {content[:200]}[/grey58]")
            elif role == "assistant":
                if m.get("tool_calls") and not content.strip():
                    continue
                if not content.strip():
                    continue
                self.console.print()
                self.console.print("[grey58]> [/grey58]", end="")
                self.console.print(Markdown(content))
            elif role == "tool":
                if m.get("harness_guard"):
                    continue
                tn = m.get("tool_name", "tool")
                preview = str(content).replace("\n", " ")[:100]
                self.console.print(
                    f"  [grey35]{M_OUT}[/grey35] [grey58]{tn}[/grey58] "
                    f"[grey35]{preview}[/grey35]"
                )

    def _cmd_status(self):
        rt = settings_store.load()
        info = Table.grid(padding=(0, 2))
        info.add_column(style="grey58", no_wrap=True)
        info.add_column(style="white")
        info.add_row("provider", get_active())
        _, _, model = self._provider_cfg()
        info.add_row("model", model)
        info.add_row("mode", CONFIG["mode"])
        info.add_row("workspace", CONFIG["workspace"])
        info.add_row("session", self.sid)
        info.add_row("messages", str(max(0, len(self.messages) - 1)))
        info.add_row("tools", f"{len(self.tool_map)} active / {len(self.all_tools)} total")
        info.add_row("verifier", "on" if rt.get("verification_enabled", True) else "off")
        info.add_row("planning", "on" if rt.get("planning_enabled") else "off")
        info.add_row("verifier model", rt.get("verifier_model") or "(main)")
        info.add_row("trivial-skip", "on" if rt.get("verify_trivial_skip", True) else "off")
        info.add_row("verify passes", str(rt.get("min_verification_attempts", 1)))
        info.add_row("repomap", _repomap.summary_one_line(self._repomap_data))
        self.console.print(Panel(info, title="status", border_style="grey35",
                                 padding=(1, 2), expand=False))

    def _cmd_settings(self):
        rt = settings_store.load()
        info = Table.grid(padding=(0, 2))
        info.add_column(style="grey58", no_wrap=True)
        info.add_column(style="white")
        info.add_row("temperature", str(rt.get("temperature")))
        info.add_row("max_steps", str(rt.get("max_steps")))
        info.add_row("keep_alive", str(rt.get("keep_alive")))
        info.add_row("history_max", str(rt.get("history_max_messages")))
        info.add_row("ai_titles", str(rt.get("ai_titles")))
        info.add_row("verification_enabled", str(rt.get("verification_enabled")))
        info.add_row("planning_enabled", str(rt.get("planning_enabled")))
        info.add_row("max_verification_retries", str(rt.get("max_verification_retries")))
        info.add_row("min_verification_attempts", str(rt.get("min_verification_attempts")))
        info.add_row("verify_trivial_skip", str(rt.get("verify_trivial_skip")))
        info.add_row("verifier_model", str(rt.get("verifier_model") or "(main)"))
        disabled = rt.get("disabled_tools") or []
        info.add_row("disabled_tools", ", ".join(disabled) if disabled else "(none)")
        sp = (rt.get("system_prompt") or "(mode default)")
        info.add_row("system_prompt", sp[:80] + ("..." if len(sp) > 80 else ""))
        self.console.print(Panel(info, title="settings", border_style="grey35",
                                 padding=(1, 2), expand=False))

    def _cmd_tools(self):
        t = Table(border_style="grey35", show_header=True, header_style="cyan", expand=False)
        t.add_column("tool", style="white")
        t.add_column("source", style="grey58")
        t.add_column("category", style="grey58")
        t.add_column("", style="yellow", width=1)
        for name, fn in sorted(self.tool_map.items()):
            meta = getattr(fn, "_tool_meta", {})
            src = "MCP" if getattr(fn, "_mcp_schema", None) else "native"
            cat = meta.get("category", "general")
            warn = M_WARN if getattr(fn, "_requires_confirmation", False) else ""
            t.add_row(name, src, cat, warn)
        self.console.print(t)
        self.console.print(f"[grey58]{len(self.tool_map)} active tools[/grey58]")

    def _cmd_enable_disable(self, arg, enable):
        if not arg:
            self.console.print(f"[yellow]usage: /{'enable' if enable else 'disable'} <tool>[/yellow]")
            return
        if arg not in self.all_tools and arg not in self.tool_map:
            self.console.print(f"[red]tool '{arg}' not found[/red]")
            return
        rt = settings_store.load()
        disabled = set(rt.get("disabled_tools", []) or [])
        if enable:
            disabled.discard(arg)
        else:
            disabled.add(arg)
        settings_store.save({**rt, "disabled_tools": sorted(disabled)})
        self._rebuild_tools()
        self.console.print(f"[green]{M_OK}[/green] [cyan]{arg}[/cyan] {'enabled' if enable else 'disabled'}")

    def _cmd_model(self, arg):
        _, _, model = self._provider_cfg()
        if not arg:
            self.console.print(f"current model: [cyan]{model}[/cyan]")
            return
        CONFIG["model"] = arg
        save_provider(get_active(), model=arg)
        try:
            from _shared import invalidate_context_cache
            invalidate_context_cache()
        except Exception:
            pass
        self.console.print(f"[green]{M_OK}[/green] model -> [cyan]{arg}[/cyan]")
        self.console.print(
            f"[grey58]saved to .astronaut_providers.json "
            f"(provider: {get_active()})[/grey58]"
        )

    def _cmd_mode(self, arg):
        if arg not in ("Standard", "Code", "Minimal", "Creator"):
            self.console.print("[yellow]usage: /mode Standard|Code|Minimal|Creator[/yellow]")
            return
        CONFIG["mode"] = arg
        self._rebuild_tools()
        sp = get_system_prompt(arg)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": sp}
        self.console.print(f"[green]{M_OK}[/green] mode -> [yellow]{arg}[/yellow] [grey58]({len(self.tool_map)} tools)[/grey58]")

    def _cmd_provider(self, arg):
        if not arg:
            self.console.print(f"current: [cyan]{get_active()}[/cyan]")
            self.console.print("[grey58]available: " + ", ".join(PROVIDER_PRESETS.keys()) + "[/grey58]")
            return
        if arg not in PROVIDER_PRESETS:
            self.console.print(f"[red]unknown provider[/red] [grey58]({', '.join(PROVIDER_PRESETS.keys())})[/grey58]")
            return
        set_active(arg)
        try:
            s = get_provider_settings(arg)
            if s.get("model"):
                CONFIG["model"] = s["model"]
            else:
                preset = PROVIDER_PRESETS.get(arg, {})
                defaults = preset.get("default_models", [])
                if defaults:
                    CONFIG["model"] = defaults[0]
        except Exception:
            pass
        try:
            from _shared import invalidate_context_cache
            invalidate_context_cache()
        except Exception:
            pass
        self.console.print(f"[green]{M_OK}[/green] provider -> [cyan]{arg}[/cyan]")
        self.console.print(f"[grey58]model -> [cyan]{CONFIG['model']}[/cyan][/grey58]")

    def _cmd_workspace(self, arg):
        if not arg:
            self.console.print(f"current: [cyan]{CONFIG['workspace']}[/cyan]")
            return
        p = pathlib.Path(arg).expanduser().resolve()
        if not p.exists():
            self.console.print(f"[red]path does not exist:[/red] {p}")
            return
        if not p.is_dir():
            self.console.print(f"[red]not a directory:[/red] {p}")
            return
        CONFIG["workspace"] = str(p)
        try:
            rt = settings_store.load()
            rt["workspace"] = str(p)
            settings_store.save(rt)
        except Exception as e:
            self.console.print(f"[yellow]could not save workspace: {e}[/yellow]")
        self._repomap_data = None
        self.console.print(f"[green]{M_OK}[/green] workspace -> [cyan]{p}[/cyan]")

    def _cmd_mcp(self):
        status = get_mcp_manager().status()
        if not status:
            self.console.print("[grey58]no MCP servers configured[/grey58]")
            return
        t = Table(border_style="grey35", show_header=True, header_style="cyan", expand=False)
        t.add_column("name", style="white")
        t.add_column("status")
        t.add_column("tools", style="grey58")
        for s in status:
            if s.get("dead"):
                st = f"[red]{M_WARN} dead[/red]"
            elif s["connected"]:
                st = f"[green]{M_OK} connected[/green]"
            elif not s["enabled"]:
                st = "[grey58]○ disabled[/grey58]"
            else:
                st = "[yellow]○ offline[/yellow]"
            t.add_row(s["name"], st, str(s["tool_count"]))
        self.console.print(t)

    def _cmd_sessions(self):
        sessions = list_sessions()
        t = Table(border_style="grey35", show_header=True, header_style="cyan", expand=False)
        t.add_column("#", style="grey58", width=3)
        t.add_column("title", style="white")
        t.add_column("updated", style="grey58")
        marker = _sym("▸", ">")
        for i, s in enumerate(sessions, 1):
            title = s.get("title") or "New chat"
            if s["id"] == self.sid:
                title = f"[cyan]{marker}[/cyan] {title}"
            t.add_row(str(i), title, s.get("updated", ""))
        self.console.print(t)

    def _cmd_switch(self, arg):
        try:
            idx = int(arg) - 1
        except ValueError:
            self.console.print("[yellow]usage: /switch <number>[/yellow]")
            return
        sessions = list_sessions()
        if idx < 0 or idx >= len(sessions):
            self.console.print("[red]invalid index[/red]")
            return
        self.sid = sessions[idx]["id"]
        set_current_session(self.sid)
        self._load_session(self.sid)
        title = sessions[idx].get("title", "")
        self.console.print(f"[green]{M_OK}[/green] switched to [grey58]{title}[/grey58]")
        self._cmd_history()

    def _cmd_new(self):
        self.sid = create_session()
        set_current_session(self.sid)
        sp = get_system_prompt(CONFIG["mode"])
        self.messages = [{"role": "system", "content": sp}]
        append_message(self.sid, self.messages[0])
        self._attach_cache.clear()
        self.console.print(f"[green]{M_OK}[/green] new chat")

    def _cmd_rename(self, arg):
        if not arg:
            self.console.print("[yellow]usage: /rename <title>[/yellow]")
            return
        update_meta(self.sid, title=arg)
        self.console.print(f"[green]{M_OK}[/green] renamed")

    def _cmd_clear(self):
        clear_messages(self.sid)
        sp = get_system_prompt(CONFIG["mode"])
        self.messages = [{"role": "system", "content": sp}]
        append_message(self.sid, self.messages[0])
        self._attach_cache.clear()
        self.console.print(f"[green]{M_OK}[/green] chat cleared")

    def _cmd_export(self, arg):
        path = arg or f"{self.sid}.md"
        lines = [f"# Chat {self.sid}\n"]
        for m in self.messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "system":
                continue
            if role == "user":
                lines.append(f"## You\n\n{content}\n")
            elif role == "assistant":
                lines.append(f"## Bot\n\n{content}\n")
            elif role == "tool":
                if m.get("harness_guard"):
                    continue
                lines.append(f"### tool: `{m.get('tool_name')}`\n\n```\n{content}\n```\n")
        pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.console.print(f"[green]{M_OK}[/green] exported to [cyan]{path}[/cyan]")

    def _cmd_undo(self):
        msgs = [m for m in self.messages if m.get("role") != "system"]
        if len(msgs) < 2:
            self.console.print("[yellow]nothing to undo[/yellow]")
            return
        last_user = -1
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                last_user = i
                break
        if last_user < 0:
            self.console.print("[yellow]nothing to undo[/yellow]")
            return
        keep = msgs[:last_user]
        sys_msg = next((m for m in self.messages if m.get("role") == "system"), None)
        self.messages = ([sys_msg] if sys_msg else []) + keep
        clear_messages(self.sid)
        for m in self.messages:
            append_message(self.sid, m)
        self._attach_cache.clear()
        self.console.print(f"[green]{M_OK}[/green] undone last exchange ({len(msgs) - len(keep)} messages removed)")

    def _cmd_retry(self):
        msgs = self.messages
        last_user = -1
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                last_user = i
                break
        if last_user < 0:
            self.console.print("[yellow]no user message to retry[/yellow]")
            return
        self.messages = msgs[: last_user + 1]
        clear_messages(self.sid)
        for m in self.messages:
            append_message(self.sid, m)
        self._attach_cache.clear()
        user_text = msgs[last_user].get("content", "")
        self.console.print(f"[grey58]retrying:[/grey58] {user_text[:80]}")
        self._inject_preflight_hints(user_text)
        _, cfg, model = self._provider_cfg()
        provider = get_provider(cfg)

        plan_steps = self._make_plan(provider, model, user_text)
        if plan_steps:
            self._print_plan(plan_steps)
            self.console.print()

        max_steps = int(settings_store.load().get("max_steps", 15))
        self._run_agent_loop(provider, model, max_steps,
                             user_text=user_text, plan_steps=plan_steps)

        last = self.messages[-1] if self.messages else None
        need_final = not (
            last
            and last.get("role") == "assistant"
            and (last.get("content") or "").strip()
        )
        if need_final:
            self._stream_final(provider, model)

    def _cmd_find(self, arg):
        if not arg:
            self.console.print("[yellow]usage: /find <text>[/yellow]")
            return
        needle = arg.lower()
        hits = []
        for i, m in enumerate(self.messages):
            if m.get("role") == "system":
                continue
            content = (m.get("content") or "").lower()
            if needle in content:
                role = m.get("role")
                preview = (m.get("content") or "").replace("\n", " ")[:120]
                hits.append(f"[grey58]#{i}[/grey58] [cyan]{role}[/cyan]  {preview}")
        if not hits:
            self.console.print(f"[grey58]no matches for[/grey58] [cyan]{arg}[/cyan]")
            return
        self.console.print(f"[grey58]matches ({len(hits)}):[/grey58]")
        for h in hits[:30]:
            self.console.print("  " + h)

    def _cmd_stats(self):
        n_user = sum(1 for m in self.messages if m.get("role") == "user")
        n_bot = sum(1 for m in self.messages if m.get("role") == "assistant")
        n_tool = sum(1 for m in self.messages
                     if m.get("role") == "tool" and not m.get("harness_guard"))
        elapsed = int(time.time() - self.session_started)
        mins, secs = divmod(elapsed, 60)
        info = Table.grid(padding=(0, 2))
        info.add_column(style="grey58", no_wrap=True)
        info.add_column(style="white")
        info.add_row("session", self.sid)
        info.add_row("uptime", f"{mins}m {secs}s")
        info.add_row("user messages", str(n_user))
        info.add_row("bot replies", str(n_bot))
        info.add_row("tool calls (session)", str(self.tool_calls_count))
        info.add_row("tool calls (chat)", str(n_tool))
        info.add_row("verifications run", str(self.verifications_run))
        info.add_row("verifier push-backs", str(self.verifications_pushed_back))
        info.add_row("last response", f"{self.last_response_time:.1f}s")
        _, _, model = self._provider_cfg()
        info.add_row("model", model)
        info.add_row("provider", get_active())
        self.console.print(Panel(info, title="stats", border_style="grey35",
                                 padding=(1, 2), expand=False))

    def _cmd_context(self):
        total_chars = sum(len(str(m.get("content") or "")) for m in self.messages)
        for m in self.messages:
            if m.get("tool_calls"):
                total_chars += len(json.dumps(m["tool_calls"], ensure_ascii=False))
        approx_tokens = total_chars // 4
        limit = int(settings_store.load().get("history_max_messages", 40))
        info = Table.grid(padding=(0, 2))
        info.add_column(style="grey58", no_wrap=True)
        info.add_column(style="white")
        info.add_row("messages", str(len(self.messages)))
        info.add_row("total chars", f"{total_chars:,}")
        info.add_row("~ tokens", f"{approx_tokens:,}")
        info.add_row("history max", f"{limit} msgs")
        self.console.print(Panel(info, title="context", border_style="grey35",
                                 padding=(1, 2), expand=False))

    def _cmd_system(self):
        sp = ""
        if self.messages and self.messages[0].get("role") == "system":
            sp = self.messages[0].get("content", "")
        rt = settings_store.load()
        custom = rt.get("system_prompt") or ""
        self.console.print()
        self.console.print(Rule("[grey58]system prompt[/grey58]", style="grey35"))
        if custom:
            self.console.print("[yellow]-- custom override --[/yellow]")
            self.console.print(Markdown(custom))
            self.console.print()
        self.console.print("[grey58]-- effective --[/grey58]")
        self.console.print(Markdown(sp or "(empty)"))
        self.console.print(Rule(style="grey35"))
        self.console.print()

    # ---------------- banner & REPL ----------------

    def _print_banner(self):
        _, _, model = self._provider_cfg()
        try:
            provider = get_active()
        except Exception:
            provider = CONFIG.get("provider", "?")
        ws = CONFIG["workspace"]
        dot = _brand()
        self.console.print()
        self.console.print(
            f"  {dot}  [bold #a480ff]Astronaut AI[/bold #a480ff]"
            f"  [grey58]·[/grey58]  "
            f"[white]{provider}[/white][grey58]:[/grey58][cyan]{model}[/cyan]"
        )
        self.console.print(f"  [grey58]workspace[/grey58]  [white]{ws}[/white]")
        self.console.print(
            "  [grey58]type[/grey58] [cyan]/help[/cyan] "
            "[grey58]· attach files with[/grey58] [cyan]!- path -![/cyan] "
            "[grey58]· [/grey58][cyan]Ctrl+C[/cyan] [grey58]to interrupt[/grey58]"
        )
        self.console.print()

    def run(self):
        self._print_banner()
        while True:
            try:
                text = self.prompt_session.prompt(
                    FormattedText([("fg:#7c3aed bold", "❯ ")]),
                )
            except KeyboardInterrupt:
                self.console.print()
                continue
            except EOFError:
                break

            text = (text or "").strip()
            if not text:
                continue

            if text.startswith("/"):
                if text.split(maxsplit=1)[0].lower() in ("/exit", "/quit", "/q"):
                    break
                self._handle_command(text)
                continue

            try:
                self._send(text)
            except KeyboardInterrupt:
                self.console.print("[yellow]^C[/yellow]")
                continue
            except Exception as e:
                self.console.print(f"[red]error: {e}[/red]")
                continue

        self.console.print("[grey58]bye.[/grey58]")


# ============================================================
# entry point
# ============================================================

def run_cli(all_tools, tool_map, filter_tools_fn):
    cli = AstroCLI(all_tools, tool_map, filter_tools_fn)
    cli.run()