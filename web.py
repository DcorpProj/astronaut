# web.py
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
import repomap as _repomap

from flask import Flask, Response, jsonify, render_template, request, send_file
import mimetypes

from config import (
    CONFIG, get_system_prompt, PROVIDER_PRESETS, load_agents_md, model_supports_thinking
)
from history import (
    list_sessions, get_meta, update_meta, delete_session,
    create_session, append_message, load_messages, clear_messages,
    get_current_session, set_current_session,
    truncate_after, replace_message, last_user_index,
    find_user_index_for_assistant,
)
from providers import get_provider
from providers_config import (
    list_providers, get_active, set_active,
    save_provider, get_provider_settings, delete_provider,
)
import settings as settings_store
import planning
from mcp_client import (
    get_manager as get_mcp_manager,
    load_config as mcp_load,
    save_config as mcp_save,
)


# ════════════════════════════════════════════════════════════
# PINNED FILES + GIT STATE
# ════════════════════════════════════════════════════════════

_PINS_PATH = Path(".astronaut_pins.json")
_MAX_PINS  = 10


def _ws_key() -> str:
    try:
        return str(Path(CONFIG["workspace"]).expanduser().resolve())
    except Exception:
        return str(CONFIG.get("workspace") or ".")


def _load_pins() -> dict:
    if _PINS_PATH.exists():
        try:
            data = json.loads(_PINS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"workspaces": {}}


def _save_pins(data: dict) -> dict:
    try:
        _PINS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return data


def _ws_pins() -> dict:
    data = _load_pins()
    ws = _ws_key()
    workspaces = data.setdefault("workspaces", {})
    entry = workspaces.setdefault(ws, {})
    entry.setdefault("files", [])
    entry.setdefault("git", {"branch": ""})
    return entry


def _update_ws_pins(mutator):
    data = _load_pins()
    ws = _ws_key()
    workspaces = data.setdefault("workspaces", {})
    entry = workspaces.setdefault(ws, {})
    entry.setdefault("files", [])
    entry.setdefault("git", {"branch": ""})
    mutator(entry)
    _save_pins(data)
    return entry

def _sync_config_from_providers():
    try:
        active = get_active()
        s = get_provider_settings(active)
        if s.get("model"):
            CONFIG["model"] = s["model"]
        else:
            preset = PROVIDER_PRESETS.get(active, {})
            defaults = preset.get("default_models", [])
            if defaults:
                CONFIG["model"] = defaults[0]
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# ATTACHMENTS
# ════════════════════════════════════════════════════════════

_FILE_ATTACH_RE   = re.compile(r'!-\s*([^\n\r]+?)\s*-!')
_ATTACH_OPEN      = "!-"
_ATTACH_CLOSE     = "-!"
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
            lang = p.suffix.lstrip(".").lower() or "text"
            return (
                f"\n--- ATTACHED FILE: {p.name} ({size} bytes) ---\n"
                f"```{lang}\n{content}\n```\n"
                f"--- END ATTACHED FILE: {p.name} ---\n"
            )
        except Exception as e:
            return f"[attachment error: '{raw}': {e}]"

    expanded = _FILE_ATTACH_RE.sub(_repl, text)
    return expanded, attached


_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".rag_index", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target", ".idea", ".vscode",
    "sessions", "skills",
}


# ════════════════════════════════════════════════════════════
# SCHEMA
# ════════════════════════════════════════════════════════════

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
            props[pname] = {"type": tmap.get(ann, "string")
                            if ann is not inspect.Parameter.empty else "string"}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        meta = getattr(fn, "_tool_meta", {})
        schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta.get("description", fn.__doc__ or ""),
                "parameters": {"type": "object", "properties": props,
                               "required": required},
            },
        })
    return schema


def _human_size(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _ollama_base_url():
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not base.startswith("http"):
        base = "http://" + base
    return base.rstrip("/")


_CONFIRM_WAITERS = {}
_CONFIRM_LOCK    = threading.Lock()


# ════════════════════════════════════════════════════════════
# WEB-LEVEL GUARDS
# ════════════════════════════════════════════════════════════

_FILE_ACTION_TOOLS = (
    "create_file", "write_file", "save_file", "append_to_file",
    "append_file", "open_file", "edit_file", "write_to_file",
)

_EMPTY_PROMISE_PATTERNS = (
    "i will save", "i'll save", "i will create", "i'll create",
    "i will write", "i'll write", "saving the file", "creating the file",
    "file has been created", "file has been saved", "file created:",
    "file saved:", "saved to ", "i have created", "i have saved",
    "successfully saved", "successfully created",
    "файл создан", "файл сохранён", "файл сохранен", "сохраню файл",
)


def _looks_like_empty_promise(text):
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in _EMPTY_PROMISE_PATTERNS)


def _had_file_tool_recently(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return False
        if m.get("role") == "tool":
            if m.get("harness_guard"):
                continue
            tn = (m.get("tool_name") or "").lower()
            if any(k in tn for k in _FILE_ACTION_TOOLS):
                return True
    return False


def _user_wants_file_action(text):
    if not text:
        return False
    low = text.lower()
    return any(h in low for h in (
        "save", "save to", "save it", "save this", "create file",
        "write to file", "write a file", "in a file", "into a file",
        "сохрани", "запиши", "создай файл", "в файл",
    ))


def _looks_like_fake_tool_code(text, tools):
    if not text:
        return False
    for m in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text):
        name = m.group(1)
        if name in tools:
            return True
        if name.lower() in ("context7", "tavily", "firecrawl"):
            return True
    if re.search(r'\bimport\s+(context7|tavily|firecrawl)\b',
                 text, re.IGNORECASE):
        return True
    return False


# ════════════════════════════════════════════════════════════
# SESSION
# ════════════════════════════════════════════════════════════

class Session:
    def __init__(self, sid, all_tools, filter_fn):
        self.sid = sid
        self.all_tools = all_tools
        self.filter_fn = filter_fn
        self.lock = threading.Lock()
        self.input_queue = queue.Queue()
        self.subscribers = []
        self.events = []
        self.mode = CONFIG["mode"]
        self.disabled_tools = set()
        self.workspace = str(Path(CONFIG["workspace"]).resolve())
        self.busy = False
        self.current = None
        self.tools = self._compute_tools()
        self._cancel = threading.Event()
        self._plan_current_step = 0
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def cancel(self):
        self._cancel.set()

    def _compute_tools(self):
        base = self.filter_fn(self.mode, self.all_tools)
        return {k: v for k, v in base.items() if k not in self.disabled_tools}

    def recompute_tools(self):
        with self.lock:
            self.tools = self._compute_tools()

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def emit(self, ev):
        with self.lock:
            self.events.append(ev)
            if len(self.events) > 2000:
                self.events = self.events[-1000:]
            for q in list(self.subscribers):
                q.put(ev)

    def enqueue(self, task):
        self.input_queue.put(task)
        label = task.get("text") or task.get("kind", "task")
        self.emit({"type": "queued", "text": label})
        self.emit({"type": "queue", "size": self.input_queue.qsize()})

    def _run(self):
        while True:
            task = self.input_queue.get()
            if task is None:
                return
            try:
                self._process(task)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.emit({"type": "error", "message": str(e)})
                self.emit({"type": "done", "queue_size": self.input_queue.qsize()})

    def _runtime_settings(self):
        return settings_store.load()

    def _resolve_system_prompt(self, runtime):
        custom = (runtime.get("system_prompt") or "").strip()
        if custom:
            return custom
        return get_system_prompt(self.mode)

    def _get_think_param(self, runtime):
        mode = runtime.get("thinking_mode", "auto")
        if mode == "on":
            return True
        if mode == "off":
            return False
        return None

    def _build_provider_cfg(self, runtime):
        active = get_active()
        preset = PROVIDER_PRESETS.get(active, {})
        settings = get_provider_settings(active)
        temperature = float(runtime.get("temperature", 0.4))
        keep_alive = runtime.get("keep_alive", "30m")

        if active == "ollama":
            cfg = {"provider": "ollama",
                   "keep_alive": keep_alive,
                   "temperature": temperature}
            model = settings.get("model") or CONFIG["model"] or "qwen2.5:7b"
        else:
            cfg = {"provider": "openai",
                   "api_key": settings.get("api_key") or CONFIG.get("api_key"),
                   "base_url": settings.get("base_url") or preset.get("default_base"),
                   "temperature": temperature}
            model = settings.get("model") or CONFIG["model"] or "gpt-4o-mini"
        return active, cfg, model

    def _apply_think_switch(self, messages, think, model):
        if think is None:
            return messages
        if not model_supports_thinking(model):
            return messages
        marker = " /think" if think else " /no_think"
        out = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content") or ""
                if "/think" not in c and "/no_think" not in c:
                    out.append({**m, "content": c + marker})
                else:
                    out.append(m)
            else:
                out.append(m)
        return out

    def _expand_attachments_in_messages(self, messages):
        ws = Path(CONFIG["workspace"]).resolve()
        out = []
        for m in messages:
            if m.get("role") != "user" or not m.get("content"):
                out.append(m)
                continue
            c = m["content"]
            if _ATTACH_OPEN not in c or _ATTACH_CLOSE not in c:
                out.append(m)
                continue
            expanded, attached = _expand_file_attachments(c, ws)
            if attached:
                self.emit({"type": "attachments_used",
                           "files": [os.path.basename(p) for p in attached]})
            if expanded != c:
                mm = dict(m)
                mm["content"] = expanded
                out.append(mm)
            else:
                out.append(m)
        return out

    def _process(self, task):
        self.busy = True
        self._cancel.clear()
        self._plan_current_step = 0
        kind = task.get("kind", "chat")
        self.current = task.get("text") or kind
        recent_calls = []
        plan_steps = []
        fake_code_retries = 0
        empty_promise_retries = 0
        user_wants_file = _user_wants_file_action(self.current or "")

        try:
            runtime = self._runtime_settings()
            think = self._get_think_param(runtime)

            self.emit({"type": "queue", "size": self.input_queue.qsize()})
            self.emit({"type": "status", "status": "thinking"})

            if kind == "chat":
                text = (task.get("text") or "").strip()
                if text:
                    append_message(self.sid, {"role": "user", "content": text})
                    self.emit({"type": "user_message", "text": text})

            elif kind == "regenerate":
                msg_index = task.get("message_index", -1)
                if msg_index >= 0:
                    msgs = load_messages(self.sid)
                    ui = (msg_index
                          if msgs[msg_index].get("role") == "user"
                          else find_user_index_for_assistant(self.sid, msg_index))
                    if ui >= 0:
                        truncate_after(self.sid, ui)
                        self.emit({"type": "history_truncated", "keep_up_to": ui})
                else:
                    ui = last_user_index(self.sid)
                    if ui >= 0:
                        truncate_after(self.sid, ui)
                        self.emit({"type": "history_truncated", "keep_up_to": ui})
                self.emit({"type": "regenerating"})

            elif kind == "edit":
                msg_index = task.get("message_index", -1)
                new_content = (task.get("new_content") or "").strip()
                if msg_index >= 0 and new_content:
                    replace_message(self.sid, msg_index, new_content)
                    self.emit({"type": "history_truncated", "keep_up_to": msg_index})
                    self.emit({"type": "edited", "index": msg_index,
                               "content": new_content})

            if runtime.get("ai_titles", True):
                meta = get_meta(self.sid)
                if meta and meta.get("title", "New chat") in ("New chat", ""):
                    first_user = next(
                        (m for m in load_messages(self.sid)
                         if m.get("role") == "user"), None)
                    if first_user:
                        title = self._gen_title(first_user["content"], runtime)
                        if title:
                            update_meta(self.sid, title=title)
                            self.emit({"type": "title", "sid": self.sid,
                                       "title": title})

            history_max = int(runtime.get("history_max_messages", 40))
            messages = load_messages(self.sid, tail=history_max)
            messages = self._expand_attachments_in_messages(messages)

            sys_prompt = self._resolve_system_prompt(runtime)
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": sys_prompt}] + messages
            else:
                messages[0] = {"role": "system", "content": sys_prompt}

            try:
                agents_block, agents_meta = load_agents_md(self.workspace)
                if agents_block:
                    insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                    messages.insert(insert_at,
                                    {"role": "system", "content": agents_block})
                    self.emit({"type": "agents_loaded", "files": agents_meta})
            except Exception:
                pass

            provider_name, provider_cfg, model = self._build_provider_cfg(runtime)
            self.emit({"type": "provider_active",
                       "provider": provider_name, "model": model})

            planning_on = bool(runtime.get("planning_enabled", False))
            if planning_on and kind in ("chat", "regenerate", "edit"):
                self.emit({"type": "status", "status": "planning"})

                plan_user_text = ""
                if kind == "chat":
                    plan_user_text = (task.get("text") or "").strip()
                if not plan_user_text:
                    for m in reversed(messages):
                        if m.get("role") == "user" and m.get("content"):
                            plan_user_text = m["content"]
                            break
                if not plan_user_text:
                    plan_user_text = self.current or ""

                tool_names = set(self.tools.keys())

                plan_provider = None
                try:
                    plan_provider = get_provider(provider_cfg)
                except Exception as e:
                    self.emit({"type": "plan_generated", "steps": [],
                               "note": f"planner unavailable: {e}"})
                else:
                    try:
                        plan_steps = planning.generate_plan(
                            plan_provider, model,
                            user_text=plan_user_text,
                            history=messages,
                            tool_names=tool_names,
                        )
                    except Exception as e:
                        plan_steps = []
                        print(f"[planning] generate_plan raised: "
                              f"{type(e).__name__}: {e}", flush=True)
                        self.emit({"type": "plan_generated", "steps": [],
                                   "note": f"planner failed: {type(e).__name__}: {e}"})

                if plan_provider is not None:
                    if plan_steps:
                        self.emit({"type": "plan_generated", "steps": plan_steps})
                        block = planning.format_for_prompt(plan_steps)
                        last_user_i = None
                        for i in range(len(messages) - 1, -1, -1):
                            if messages[i].get("role") == "user":
                                last_user_i = i
                                break
                        if last_user_i is None:
                            messages.append({"role": "system", "content": block})
                        else:
                            messages.insert(last_user_i,
                                            {"role": "system", "content": block})
                    else:
                        self.emit({"type": "plan_generated", "steps": [],
                                   "note": "no plan needed for this task"})
                self.emit({"type": "status", "status": "thinking"})

            messages = self._apply_think_switch(messages, think, model)

            provider = get_provider(provider_cfg)
            step = 0
            final_done = False
            final_answer = None
            tools = self.tools
            max_steps = int(runtime.get("max_steps", 15))
            cancelled = False

            while step < max_steps and not final_done:
                if self._cancel.is_set():
                    cancelled = True
                    break

                schema = build_schema(tools)
                resp = provider.chat(
                    messages,
                    tools=schema if schema else None,
                    model=model,
                    think=think,
                )

                if (not resp.tool_calls and resp.content
                        and _looks_like_fake_tool_code(resp.content, tools)):
                    if fake_code_retries < 2:
                        fake_code_retries += 1
                        self.emit({"type": "status",
                                   "status": f"correcting (fake code {fake_code_retries}/2)"})
                        messages.append({"role": "assistant",
                                         "content": resp.content})
                        messages.append({
                            "role": "system",
                            "content": (
                                "[CORRECTION] You wrote Python-style code in "
                                "your TEXT. That does nothing. To call a tool, "
                                "return a REAL TOOL CALL. No Python code, no "
                                "imports, no fake function calls."
                            ),
                        })
                        continue

                if (not resp.tool_calls and resp.content
                        and user_wants_file
                        and _looks_like_empty_promise(resp.content)
                        and not _had_file_tool_recently(messages)):
                    if empty_promise_retries < 3:
                        empty_promise_retries += 1
                        self.emit({"type": "status",
                                   "status": f"correcting (empty promise {empty_promise_retries}/3)"})
                        messages.append({"role": "assistant",
                                         "content": resp.content})
                        messages.append({
                            "role": "system",
                            "content": (
                                "[STOP — YOU DID NOT ACTUALLY DO ANYTHING]\n"
                                "You promised to save a file but did not "
                                "call create_file. Text is not action.\n"
                                "Return a real tool call to create_file with "
                                "filename and content arguments."
                            ),
                        })
                        continue

                if resp.tool_calls:
                    if resp.content:
                        self.emit({"type": "plan", "content": resp.content})

                    am = {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": [
                            {"function": {"name": tc["name"],
                                          "arguments": tc["arguments"]}}
                            for tc in resp.tool_calls
                        ],
                    }
                    messages.append(am)
                    append_message(self.sid, am)

                    for i, tc in enumerate(resp.tool_calls):
                        if self._cancel.is_set():
                            cancelled = True
                            break

                        step_index = 0
                        if plan_steps:
                            step_index = planning.match_step(
                                plan_steps, tc["name"], tc["arguments"],
                                current=self._plan_current_step,
                            )
                            if step_index and step_index >= self._plan_current_step:
                                self._plan_current_step = step_index

                        call_id = f"tc_{step}_{i}_{int(time.time()*1000)}"
                        self.emit({"type": "tool_call",
                                   "id": call_id,
                                   "name": tc["name"],
                                   "args": tc["arguments"],
                                   "step_index": step_index})

                        try:
                            key = (tc["name"],
                                   json.dumps(tc["arguments"], sort_keys=True))
                        except Exception:
                            key = (tc["name"], str(tc["arguments"]))

                        if key in recent_calls:
                            # Harness guard — not a real tool result.
                            result = ("[HARNESS] Duplicate call blocked. "
                                      "The tool was NOT executed this time. "
                                      "The previous identical call is in the "
                                      "transcript above — check its result.")
                            tm = {"role": "tool", "tool_name": tc["name"],
                                  "content": result,
                                  "harness_guard": "duplicate"}
                            messages.append(tm)
                            self.emit({"type": "tool_result",
                                       "id": call_id,
                                       "name": tc["name"], "result": result})
                            continue
                        recent_calls.append(key)

                        fn = tools.get(tc["name"])
                        if not fn:
                            result = f"Unknown tool: {tc['name']}"
                        elif getattr(fn, "_requires_confirmation", False):
                            cid = f"c_{step}_{tc['name']}_{int(time.time()*1000)}"
                            self.emit({"type": "status", "status": "asking"})
                            self.emit({"type": "confirm_request",
                                       "id": cid,
                                       "name": tc["name"],
                                       "args": tc["arguments"]})
                            ans = _wait_confirm(self.sid, cid, timeout=600)
                            self.emit({"type": "status", "status": "thinking"})
                            if ans == "y":
                                try:
                                    result = fn(**tc["arguments"])
                                except Exception as e:
                                    result = f"Tool error: {e}"
                            else:
                                result = "[HARNESS] Cancelled by user. The tool was NOT executed."
                                tm = {"role": "tool", "tool_name": tc["name"],
                                      "content": result,
                                      "harness_guard": "cancelled"}
                                messages.append(tm)
                                self.emit({"type": "tool_result",
                                           "id": call_id,
                                           "name": tc["name"], "result": result})
                                continue
                        else:
                            try:
                                result = fn(**tc["arguments"])
                            except Exception as e:
                                result = f"Tool error: {e}"

                        tm = {"role": "tool", "tool_name": tc["name"],
                              "content": str(result)}
                        messages.append(tm)
                        append_message(self.sid, tm)

                        preview = str(result)[:600] + ("..." if len(str(result)) > 600 else "")
                        self.emit({"type": "tool_result",
                                   "id": call_id,
                                   "name": tc["name"], "result": preview})

                    if cancelled:
                        break
                    step += 1
                else:
                    self.emit({"type": "status", "status": "typing"})
                    full = ""
                    try:
                        for chunk in provider.chat(
                            messages, tools=None, stream=True,
                            model=model, think=think,
                        ):
                            if self._cancel.is_set():
                                cancelled = True
                                break
                            th = getattr(chunk, "thinking", "") or ""
                            if th and think is not False:
                                self.emit({"type": "thinking", "content": th})
                            if chunk.content:
                                full += chunk.content
                                self.emit({"type": "chunk", "content": chunk.content})
                    except Exception as e:
                        if not cancelled:
                            self.emit({"type": "error", "message": str(e)})
                    final_answer = full
                    break

            if cancelled:
                self.emit({"type": "stopped"})
                if plan_steps:
                    self.emit({"type": "plan_aborted"})
            elif final_answer is None:
                self.emit({"type": "status", "status": "typing"})
                r = provider.chat(messages, tools=None, model=model, think=think)
                final_answer = r.content
                self.emit({"type": "chunk", "content": final_answer})
                am = {"role": "assistant", "content": final_answer}
                messages.append(am)
                append_message(self.sid, am)
            else:
                am = {"role": "assistant", "content": final_answer}
                messages.append(am)
                append_message(self.sid, am)

            if plan_steps and not cancelled:
                self.emit({"type": "plan_completed"})

            if kind == "regenerate" and not cancelled:
                self.emit({"type": "regenerated"})

        finally:
            self.busy = False
            self.current = None
            self.emit({"type": "status", "status": "idle"})
            self.emit({"type": "done", "queue_size": self.input_queue.qsize()})

    def _gen_title(self, first_message, runtime):
        try:
            _, cfg, model = self._build_provider_cfg(runtime)
            provider = get_provider(cfg)
            msgs = [
                {"role": "system",
                 "content": "Generate a short chat title (3-6 words). "
                            "Reply ONLY with the title."},
                {"role": "user", "content": first_message[:500]},
            ]
            resp = provider.chat(msgs, tools=None, model=model)
            t = resp.content.strip().strip('"').strip("'").split("\n")[0][:60]
            return t or None
        except Exception:
            return None


def _wait_confirm(sid, cid, timeout=600):
    ev = threading.Event()
    holder = {"value": None}
    with _CONFIRM_LOCK:
        _CONFIRM_WAITERS[(sid, cid)] = (ev, holder)
    ev.wait(timeout=timeout)
    return holder["value"] or "n"


# ════════════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════════════

def create_app(tool_map, all_tools, load_plugins_fn, filter_tools_fn):
    _sync_config_from_providers()
    app = Flask(__name__)
    sessions = {}
    sessions_lock = threading.Lock()

    def get_session_obj(sid):
        with sessions_lock:
            if sid not in sessions:
                sessions[sid] = Session(sid, all_tools, filter_tools_fn)
            return sessions[sid]

    def _reload_mcp():
        mgr = get_mcp_manager()
        mgr.reload()
        for name in [n for n, f in all_tools.items()
                     if getattr(f, "_mcp_schema", None)]:
            all_tools.pop(name, None)
        for name, fn in mgr.get_tools().items():
            all_tools.setdefault(name, fn)
        with sessions_lock:
            for s in sessions.values():
                s.recompute_tools()

    @app.route("/")
    def index():
        if not list_sessions():
            create_session()
        sid = get_current_session()
        set_current_session(sid)
        return render_template(
            "index.html",
            model=CONFIG["model"],
            mode=CONFIG["mode"],
            provider=CONFIG["provider"],
            workspace=str(Path(CONFIG["workspace"]).resolve()),
            current_sid=sid,
        )

    @app.route("/api/files/list")
    def api_files_list():
        ws = Path(CONFIG["workspace"]).resolve()
        if not ws.exists() or not ws.is_dir():
            return jsonify([])

        q      = (request.args.get("q") or "").strip().lower()
        limit  = max(1, min(int(request.args.get("limit", 300)), 2000))
        max_sz = 500_000

        out = []
        for p in ws.rglob("*"):
            try:
                if any(part in _IGNORE_DIRS for part in p.parts):
                    continue
                if not p.is_file():
                    continue
                sz = p.stat().st_size
                if sz > max_sz:
                    continue
                rel = str(p.relative_to(ws)).replace("\\", "/")
                if q and q not in rel.lower():
                    continue
                out.append({"path": rel, "size": sz})
                if len(out) >= limit * 4:
                    break
            except (OSError, PermissionError):
                continue

        out.sort(key=lambda x: x["path"])
        return jsonify(out[:limit])

        # ═══ REPOMAP ═══

    @app.route("/api/repomap")
    def api_repomap():
        """Return the current repomap as JSON. Query params:
             refresh=1   force rebuild
             graph=1     return node/edge graph instead of raw map
             tree=1      return directory tree view
        """
        ws = Path(CONFIG["workspace"]).resolve()
        force = request.args.get("refresh", "0") == "1"
        want_graph = request.args.get("graph", "0") == "1"
        want_tree  = request.args.get("tree",  "0") == "1"

        try:
            data = _repomap.get_or_build(str(ws), force=force)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if want_graph:
            return jsonify(_repomap.as_graph(data))
        if want_tree:
            return jsonify(_repomap.as_tree(data))
        return jsonify(data)

    @app.route("/api/repomap/regen", methods=["POST"])
    def api_repomap_regen():
        ws = Path(CONFIG["workspace"]).resolve()
        try:
            data = _repomap.build_map(ws)
            _repomap.save_map(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({
            "ok": True,
            "count": data.get("count", 0),
            "workspace": data.get("workspace"),
        })

    @app.route("/api/repomap/summaries", methods=["POST"])
    def api_repomap_summaries():
        """Generate one-line summaries for every file. Uses the active
        provider/model. Slow (one call per file) but cached by mtime."""
        body = request.get_json(force=True, silent=True) or {}
        ws = Path(CONFIG["workspace"]).resolve()
        active = get_active()
        preset = PROVIDER_PRESETS.get(active, {})
        settings = get_provider_settings(active)
        try:
            max_files = max(1, min(int(body.get("max_files", 60)), 500))
        except (ValueError, TypeError):
            max_files = 60
        
        runtime = settings_store.load()
        if active == "ollama":
            cfg = {
                "provider": "ollama",
                "keep_alive": runtime.get("keep_alive", "30m"),
                "temperature": 0.2,
            }
            model = settings.get("model") or CONFIG["model"] or "qwen2.5:7b"
        else:
            cfg = {
                "provider": "openai",
                "api_key": settings.get("api_key") or CONFIG.get("api_key"),
                "base_url": settings.get("base_url") or preset.get("default_base"),
                "temperature": 0.2,
            }
            model = settings.get("model") or CONFIG["model"] or "gpt-4o-mini"

        try:
            provider = get_provider(cfg)
        except Exception as e:
            return jsonify({"error": f"provider init failed: {e}"}), 500

        try:
            data = _repomap.build_map(
                ws, with_summaries=True, provider=provider, model=model, max_summaries=max_files
            )
            _repomap.save_map(data)
        except Exception as e:
            import traceback
            return jsonify({
                "error": str(e),
                "trace": traceback.format_exc(),
            }), 500

        n = sum(1 for f in data.get("files", []) if f.get("summary"))
        return jsonify({
            "ok": True,
            "summarized": n,
            "total": data.get("count", 0),
        })

    @app.route("/api/repomap/file")
    def api_repomap_file():
        """Read a single file's content for the preview pane."""
        rel = (request.args.get("path") or "").strip().replace("\\", "/")
        if not rel:
            return jsonify({"error": "path required"}), 400
        ws = Path(CONFIG["workspace"]).resolve()
        try:
            target = (ws / rel).resolve()
            if not str(target).startswith(str(ws)):
                return jsonify({"error": "forbidden"}), 403
            if not target.exists() or not target.is_file():
                return jsonify({"error": "not found"}), 404
            size = target.stat().st_size
            if size > 200_000:
                return jsonify({
                    "error": f"file too large ({size} bytes)",
                    "size": size,
                }), 400
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify({
            "path": rel,
            "size": size,
            "content": content,
        })

    @app.route("/api/repomap/raw")
    def api_repomap_raw():
        """Serve raw file bytes with correct MIME type (images, PDF, audio, video)."""
        rel = (request.args.get("path") or "").strip().replace("\\", "/").lstrip("/")
        if not rel or Path(rel).is_absolute():
            return jsonify({"error": "invalid path"}), 400

        ws = Path(CONFIG["workspace"]).resolve()
        try:
            target = (ws / rel).resolve()
            target.relative_to(ws)
        except (ValueError, OSError):
            return jsonify({"error": "forbidden"}), 403

        if not target.is_file():
            return jsonify({"error": "not found"}), 404

        try:
            size = target.stat().st_size
        except OSError:
            return jsonify({"error": "unreadable"}), 500

        if size > 25 * 1024 * 1024:
            return jsonify({"error": f"too large ({size} B)"}), 400

        mime, _ = mimetypes.guess_type(str(target))
        return send_file(
            str(target),
            mimetype=mime or "application/octet-stream",
            as_attachment=False,
            conditional=True,
        )
    
    @app.route("/api/pins")
    def api_pins_get():
        return jsonify(_ws_pins())

    @app.route("/api/pins/add", methods=["POST"])
    def api_pins_add():
        data = request.get_json(force=True) or {}
        path = (data.get("path") or "").strip().replace("\\", "/")
        if not path:
            return jsonify({"error": "path required"}), 400

        err = {"msg": None}

        def _mut(entry):
            files = entry.setdefault("files", [])
            if path in files:
                return
            if len(files) >= _MAX_PINS:
                err["msg"] = f"max {_MAX_PINS} pins"
                return
            files.append(path)

        entry = _update_ws_pins(_mut)
        if err["msg"]:
            return jsonify({"error": err["msg"], "files": entry.get("files", [])}), 400
        return jsonify({"ok": True, "files": entry.get("files", [])})

    @app.route("/api/pins/remove", methods=["POST"])
    def api_pins_remove():
        data = request.get_json(force=True) or {}
        path = (data.get("path") or "").strip().replace("\\", "/")

        def _mut(entry):
            entry["files"] = [f for f in entry.get("files", []) if f != path]

        entry = _update_ws_pins(_mut)
        return jsonify({"ok": True, "files": entry.get("files", [])})

    @app.route("/api/pins/clear", methods=["POST"])
    def api_pins_clear():
        def _mut(entry):
            entry["files"] = []
        entry = _update_ws_pins(_mut)
        return jsonify({"ok": True, "files": []})

    @app.route("/api/git/status")
    def api_git_status():
        ws = Path(CONFIG["workspace"]).resolve()
        branches, current, error = [], "", ""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(ws), capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                return jsonify({"is_repo": False, "branches": [],
                                "current": "", "error": "not a git repo"})
        except Exception as e:
            return jsonify({"is_repo": False, "branches": [],
                            "current": "", "error": str(e)})

        try:
            r2 = subprocess.run(
                ["git", "branch", "--format=%(refname:short)"],
                cwd=str(ws), capture_output=True, text=True, timeout=10,
            )
            if r2.returncode == 0:
                branches = [b.strip() for b in r2.stdout.splitlines() if b.strip()]
            else:
                error = r2.stderr.strip()
        except Exception as e:
            error = str(e)

        try:
            r3 = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(ws), capture_output=True, text=True, timeout=5,
            )
            current = r3.stdout.strip() if r3.returncode == 0 else ""
        except Exception:
            current = ""

        return jsonify({
            "is_repo": True,
            "branches": branches,
            "current": current,
            "error": error,
        })

    @app.route("/api/git/checkout", methods=["POST"])
    def api_git_checkout():
        data = request.get_json(force=True) or {}
        branch = (data.get("branch") or "").strip()
        if not branch:
            return jsonify({"error": "branch required"}), 400
        ws = Path(CONFIG["workspace"]).resolve()
        try:
            r = subprocess.run(
                ["git", "checkout", branch],
                cwd=str(ws), capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return jsonify({"error": r.stderr.strip() or "checkout failed"})
            def _mut(entry):
                entry.setdefault("git", {})["branch"] = branch
            _update_ws_pins(_mut)
            return jsonify({"ok": True, "current": branch,
                            "output": r.stdout.strip()})
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/api/sessions")
    def api_sessions():
        return jsonify(list_sessions())

    @app.route("/api/sessions", methods=["POST"])
    def api_sessions_create():
        sid = create_session()
        set_current_session(sid)
        return jsonify({"id": sid, "title": "New chat",
                        "created": "", "updated": ""})

    @app.route("/api/sessions/<sid>", methods=["DELETE"])
    def api_sessions_delete(sid):
        delete_session(sid)
        with sessions_lock:
            sessions.pop(sid, None)
        return jsonify({"ok": True})

    @app.route("/api/sessions/<sid>/messages")
    def api_sessions_messages(sid):
        return jsonify(load_messages(sid))

    @app.route("/api/sessions/<sid>/rename", methods=["POST"])
    def api_sessions_rename(sid):
        data = request.get_json(force=True) or {}
        title = (data.get("title") or "").strip()
        if not title:
            return jsonify({"error": "title required"}), 400
        update_meta(sid, title=title)
        return jsonify({"ok": True})

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json(force=True) or {}
        sid  = data.get("sid")
        text = (data.get("message") or "").strip()
        if not sid or not text:
            return jsonify({"error": "sid and message required"}), 400
        set_current_session(sid)
        s = get_session_obj(sid)
        s.enqueue({"kind": "chat", "text": text})
        return jsonify({"ok": True})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        data = request.get_json(force=True) or {}
        sid = data.get("sid")
        if not sid:
            return jsonify({"error": "sid required"}), 400
        s = get_session_obj(sid)
        s.cancel()
        return jsonify({"ok": True})

    @app.route("/api/regenerate", methods=["POST"])
    def api_regenerate():
        data = request.get_json(force=True) or {}
        sid = data.get("sid")
        if not sid:
            return jsonify({"error": "sid required"}), 400
        s = get_session_obj(sid)
        s.enqueue({"kind": "regenerate",
                   "message_index": data.get("message_index", -1)})
        return jsonify({"ok": True})

    @app.route("/api/edit_message", methods=["POST"])
    def api_edit_message():
        data = request.get_json(force=True) or {}
        sid  = data.get("sid")
        message_index = data.get("message_index", -1)
        new_content   = (data.get("content") or "").strip()
        if not sid or message_index < 0 or not new_content:
            return jsonify({"error": "sid, message_index, content required"}), 400
        s = get_session_obj(sid)
        s.enqueue({"kind": "edit",
                   "message_index": message_index,
                   "new_content": new_content})
        return jsonify({"ok": True})

    @app.route("/api/stream/<sid>")
    def api_stream(sid):
        s = get_session_obj(sid)
        q = s.subscribe()
        backlog = list(s.events)[-50:]

        def gen():
            try:
                for ev in backlog:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                while True:
                    try:
                        ev = q.get(timeout=25)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            finally:
                s.unsubscribe(q)

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    @app.route("/api/confirm", methods=["POST"])
    def api_confirm():
        data = request.get_json(force=True) or {}
        key  = (data.get("sid"), data.get("id"))
        ans  = data.get("answer", "n")
        with _CONFIRM_LOCK:
            if key in _CONFIRM_WAITERS:
                ev, holder = _CONFIRM_WAITERS.pop(key)
                holder["value"] = ans
                ev.set()
                return jsonify({"ok": True})
        return jsonify({"ok": False}), 404

    @app.route("/api/settings")
    def api_settings_get():
        return jsonify(settings_store.load())

    @app.route("/api/settings", methods=["POST"])
    def api_settings_save():
        data = request.get_json(force=True) or {}
        allowed = {
            "system_prompt", "temperature", "max_steps", "keep_alive",
            "history_max_messages", "ai_titles",
            "verification_enabled", "thinking_mode",
            "planning_enabled",
            "workspace",
        }
        payload = {k: v for k, v in data.items() if k in allowed}
        try:
            if "temperature" in payload:
                payload["temperature"] = float(payload["temperature"])
            if "max_steps" in payload:
                payload["max_steps"] = int(payload["max_steps"])
            if "history_max_messages" in payload:
                payload["history_max_messages"] = int(payload["history_max_messages"])
            if "ai_titles" in payload:
                payload["ai_titles"] = bool(payload["ai_titles"])
            if "verification_enabled" in payload:
                payload["verification_enabled"] = bool(payload["verification_enabled"])
            if "planning_enabled" in payload:
                payload["planning_enabled"] = bool(payload["planning_enabled"])
            if "thinking_mode" in payload:
                v = str(payload["thinking_mode"]).lower()
                payload["thinking_mode"] = v if v in ("auto", "on", "off") else "auto"
            if "workspace" in payload:
                payload["workspace"] = str(payload["workspace"]).strip()
                if payload["workspace"]:
                    CONFIG["workspace"] = payload["workspace"]
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"invalid value: {e}"}), 400
        merged = settings_store.save(payload)
        return jsonify({"ok": True, "settings": merged})

    @app.route("/api/model/capabilities")
    def api_model_capabilities():
        name = (request.args.get("model") or "").strip()
        if not name:
            return jsonify({"thinking": False})
        active = get_active()
        if active != "ollama":
            return jsonify({"thinking": False, "provider": active})
        try:
            supported = bool(model_supports_thinking(name))
        except Exception:
            supported = False
        return jsonify({"thinking": supported, "provider": active})

    @app.route("/api/mcp/servers")
    def api_mcp_servers():
        cfg = mcp_load()
        status = get_mcp_manager().status()
        status_map = {s["name"]: s for s in status}
        out = []
        for name, server in cfg.get("servers", {}).items():
            st = status_map.get(name, {})
            out.append({
                "name": name,
                "command": server.get("command", ""),
                "args": server.get("args", []),
                "env": server.get("env", {}),
                "enabled": server.get("enabled", True),
                "connected": st.get("connected", False),
                "tools": st.get("tools", []),
                "tool_count": st.get("tool_count", 0),
            })
        return jsonify(out)

    @app.route("/api/mcp/servers", methods=["POST"])
    def api_mcp_save():
        data = request.get_json(force=True) or {}
        name = (data.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        command = (data.get("command") or "").strip()
        if not command:
            return jsonify({"error": "command required"}), 400
        cfg = mcp_load()
        servers = cfg.setdefault("servers", {})
        servers[name] = {
            "command": command,
            "args": data.get("args") or [],
            "env":  data.get("env")  or {},
            "enabled": bool(data.get("enabled", True)),
        }
        mcp_save(cfg)
        _reload_mcp()
        return jsonify({"ok": True})

    @app.route("/api/mcp/servers/<name>", methods=["DELETE"])
    def api_mcp_delete(name):
        cfg = mcp_load()
        servers = cfg.get("servers", {})
        servers.pop(name, None)
        mcp_save(cfg)
        _reload_mcp()
        return jsonify({"ok": True})

    @app.route("/api/mcp/reload", methods=["POST"])
    def api_mcp_reload():
        _reload_mcp()
        return jsonify({"ok": True, "status": get_mcp_manager().status()})

    @app.route("/api/providers")
    def api_providers():
        stored = list_providers()
        active = get_active()
        out = []
        for name, preset in PROVIDER_PRESETS.items():
            entry = stored.get(name, {})
            out.append({
                "name": name,
                "label": preset["label"],
                "needs_key": preset["needs_key"],
                "needs_base": preset["needs_base"],
                "default_base": preset["default_base"],
                "default_models": preset["default_models"],
                "api_key_set": bool(entry.get("api_key")),
                "api_key_masked": ("*" * 8 + entry["api_key"][-4:])
                                  if entry.get("api_key") else "",
                "base_url": entry.get("base_url", preset["default_base"]),
                "model": entry.get("model", ""),
                "models": entry.get("models", preset["default_models"]),
                "active": name == active,
            })
        return jsonify(out)

    @app.route("/api/providers/active", methods=["POST"])
    def api_providers_active():
        data = request.get_json(force=True) or {}
        name = data.get("name")
        if name not in PROVIDER_PRESETS:
            return jsonify({"error": "unknown provider"}), 400
        set_active(name)
        s = get_provider_settings(name)
        if s.get("model"):
            CONFIG["model"] = s["model"]
        else:
            preset = PROVIDER_PRESETS.get(name, {})
            defaults = preset.get("default_models", [])
            if defaults:
                CONFIG["model"] = defaults[0]
        return jsonify({"ok": True, "active": name, "model": CONFIG["model"]})

    @app.route("/api/providers/save", methods=["POST"])
    def api_providers_save():
        data = request.get_json(force=True) or {}
        name = data.get("name")
        if name not in PROVIDER_PRESETS:
            return jsonify({"error": "unknown provider"}), 400
        save_provider(
            name,
            api_key=data.get("api_key") if data.get("api_key") is not None else None,
            base_url=data.get("base_url"),
            model=data.get("model"),
            models=data.get("models"),
        )
        if name == get_active() and data.get("model"):
            CONFIG["model"] = data["model"]
        return jsonify({"ok": True, "model": CONFIG.get("model")})

    @app.route("/api/providers/<name>", methods=["DELETE"])
    def api_providers_delete(name):
        delete_provider(name)
        return jsonify({"ok": True})

    @app.route("/api/ollama/models")
    def api_ollama_models():
        try:
            import requests
            base = _ollama_base_url()
            r = requests.get(f"{base}/api/tags", timeout=5)
            r.raise_for_status()
            data = r.json()
            raw = data.get("models", [])
            out = []
            for m in raw:
                if not isinstance(m, dict):
                    continue
                name = m.get("name") or m.get("model") or ""
                if not name:
                    continue
                name = str(name)
                size = m.get("size", 0) or 0
                remote_host = m.get("remote_host")
                remote_model = m.get("remote_model")
                nl = name.lower()
                is_cloud = bool(
                    remote_host or remote_model
                    or nl.endswith(":cloud")
                    or nl.endswith("-cloud")
                    or nl.endswith("cloud")
                    or ":cloud" in nl
                    or "-cloud" in nl
                )
                out.append({
                    "name": name,
                    "cloud": is_cloud,
                    "thinking": model_supports_thinking(name),
                    "size": int(size) if size else 0,
                    "size_human": "" if is_cloud else _human_size(size),
                })
            out.sort(key=lambda x: (x["cloud"], x["name"].lower()))
            return jsonify(out)
        except Exception as e:
            import traceback
            return jsonify({
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
                "base_url": _ollama_base_url(),
            }), 500

    @app.route("/api/tools")
    def api_tools():
        sid = request.args.get("sid") or "default"
        s = get_session_obj(sid)
        base = filter_tools_fn(s.mode, all_tools)
        out = []
        for name, fn in sorted(base.items()):
            meta = getattr(fn, "_tool_meta", {})
            out.append({
                "name": name,
                "category": meta.get("category", "general"),
                "description": meta.get("description", ""),
                "confirm": bool(getattr(fn, "_requires_confirmation", False)),
                "enabled": name not in s.disabled_tools,
                "source": "mcp" if getattr(fn, "_mcp_schema", None) else "native",
                "mcp_server": meta.get("mcp_server"),
            })
        return jsonify(out)

    @app.route("/api/tools/toggle", methods=["POST"])
    def api_tools_toggle():
        data = request.get_json(force=True) or {}
        s = get_session_obj(data.get("sid"))
        name = data.get("name")
        if bool(data.get("enabled", True)):
            s.disabled_tools.discard(name)
        else:
            s.disabled_tools.add(name)
        s.recompute_tools()
        return jsonify({"ok": True, "enabled_count": len(s.tools)})

    @app.route("/api/tools/toggle_all", methods=["POST"])
    def api_tools_toggle_all():
        data = request.get_json(force=True) or {}
        s = get_session_obj(data.get("sid"))
        enabled = bool(data.get("enabled", True))
        base = filter_tools_fn(s.mode, all_tools)
        s.disabled_tools = set() if enabled else set(base.keys())
        s.recompute_tools()
        return jsonify({"ok": True, "enabled_count": len(s.tools)})

    @app.route("/api/mode", methods=["POST"])
    def api_mode():
        data = request.get_json(force=True) or {}
        mode = data.get("mode")
        if mode not in ("Standard", "Code", "Minimal", "Creator"):
            return jsonify({"error": "invalid mode"}), 400
        s = get_session_obj(data.get("sid"))
        s.mode = mode
        CONFIG["mode"] = mode
        s.recompute_tools()
        return jsonify({"ok": True, "tools": list(s.tools.keys())})

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        data = request.get_json(force=True) or {}
        clear_messages(data.get("sid"))
        return jsonify({"ok": True})

    @app.route("/api/workspace", methods=["POST"])
    def api_workspace_set():
        data = request.get_json(force=True) or {}
        new_ws = (data.get("path") or "").strip()
        if not new_ws:
            return jsonify({"error": "path required"}), 400
        try:
            p = Path(new_ws).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return jsonify({"error": f"cannot use path: {e}"}), 400
        CONFIG["workspace"] = str(p)
        try:
            rt = settings_store.load()
            rt["workspace"] = str(p)
            settings_store.save(rt)
        except Exception as e:
            print(f"[workspace] save failed: {e}", flush=True)
        s = get_session_obj(data.get("sid"))
        s.workspace = str(p)
        return jsonify({"ok": True, "workspace": str(p)})

    @app.route("/api/shutdown", methods=["POST"])
    def api_shutdown():
        server = app.config.get("_server")
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()
            return jsonify({"ok": True, "action": "shutting down"})
        return jsonify({"ok": True, "action": "no server handle"})

    return app